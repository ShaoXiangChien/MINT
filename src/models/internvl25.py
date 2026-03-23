"""InternVL2.5-8B model adapter.

InternVL2.5 maintains the same "ViT-MLP-LLM" architecture as InternVL2,
with the following key components for the 8B variant:
  - Vision encoder : InternViT-300M-448px-V2_5
  - Language model : InternLM2.5-7B-Chat
  - Projector      : randomly-initialised MLP (pixel-unshuffle, 4× reduction)

Module paths (verified against OpenGVLab/InternVL2_5-8B):
  - Decoder layers : model.language_model.model.layers[i]
  - Final norm     : model.language_model.model.norm
  - Vision layers  : model.vision_model.encoder.layers[i]
  - IMG_CONTEXT token id : resolved dynamically from tokenizer

HuggingFace model id: OpenGVLab/InternVL2_5-8B
"""

import warnings

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image

import transformers

from .base import BaseModelAdapter, ModelAndTokenizer
from src.utils.tokens import set_requires_grad

# ImageNet normalisation constants used by InternVL
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# IMG_CONTEXT token id — resolved dynamically from the tokenizer at runtime.
# Do NOT hardcode this: InternVL2 uses 151859 but InternVL2.5-8B uses 92546.
# The constant below is only a last-resort fallback and should never be reached.
_IMG_CONTEXT_TOKEN_ID_FALLBACK = 92546

# Keys that must be stripped before calling mt.model(**inputs) in capture_decoder_hs.
# image_flags MUST be kept for forward() — InternVLChatModel.forward() calls
# image_flags.squeeze(-1) unconditionally, so passing None crashes.
# _raw_prompt is our own metadata key, not accepted by forward().
_FORWARD_STRIP_KEYS = ("_raw_prompt",)

# Keys that must be stripped before calling mt.model.generate().
# GenerationMixin.generate() validates all kwargs and rejects image_flags
# because it doesn't appear in language_model.forward()'s signature.
# pixel_values is also not accepted by language_model.generate() directly —
# we handle image embedding injection via the capture/patch hooks instead.
_GENERATE_STRIP_KEYS = ("_raw_prompt", "image_flags", "pixel_values")


class InternVL25Adapter(BaseModelAdapter):
    """Adapter for InternVL2.5 family of models (tested on InternVL2_5-8B)."""

    # Override _NON_FORWARD_KEYS to only strip _raw_prompt from forward() calls.
    # image_flags must NOT be stripped from forward() — see _FORWARD_STRIP_KEYS above.
    _NON_FORWARD_KEYS = _FORWARD_STRIP_KEYS

    def load_model(self, model_path, device):
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "eager"

        model = transformers.AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation=attn_impl,
            device_map="auto",
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True,
        )
        set_requires_grad(False, model)
        model.eval()

        # InternVL initialises img_context_token_id to None and only sets it
        # inside model.chat().  We must set it here so that model.forward()
        # correctly replaces IMG_CONTEXT token embeddings with ViT features.
        img_ctx_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        if img_ctx_id is None or img_ctx_id == tokenizer.unk_token_id:
            img_ctx_id = _IMG_CONTEXT_TOKEN_ID_FALLBACK
        model.img_context_token_id = img_ctx_id

        # transformers >= 4.50 removed GenerationMixin from PreTrainedModel's
        # default inheritance chain.  InternLM2ForCausalLM uses trust_remote_code
        # and was not updated, so it loses .generate().  Patch it in at runtime.
        from transformers import GenerationMixin, GenerationConfig
        lm_cls = type(model.language_model)
        if not issubclass(lm_cls, GenerationMixin):
            lm_cls.__bases__ = lm_cls.__bases__ + (GenerationMixin,)
        # GenerationMixin.generate() reads self.generation_config; if it is None
        # (not set by from_pretrained because the class lacked GenerationMixin),
        # _prepare_generation_config crashes with 'NoneType has no _from_model_config'.
        if getattr(model.language_model, "generation_config", None) is None:
            model.language_model.generation_config = GenerationConfig(
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        mt = ModelAndTokenizer(
            model_name=model_path,
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
        return mt

    def prepare_inputs(self, prompt, image, mt):
        """Build model inputs for InternVL2.5.

        InternVL2.5 uses a custom tokenizer (InternLM2.5) where '<IMG_CONTEXT>'
        is a special token.  We must NOT rely on string tokenization of
        '<IMG_CONTEXT>' because the tokenizer may split it into subword pieces.
        Instead we tokenize only the text parts and then manually splice in the
        correct number of IMG_CONTEXT token IDs.
        """
        pixel_values = None
        num_patches = 0

        if image is not None:
            pixel_values = _load_image(image).to(torch.bfloat16).to(mt.device)
            num_patches = pixel_values.size(0)

        # Tokenize the text prompt (no image placeholder in the string)
        text_ids = mt.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True
        ).input_ids[0]  # shape: (text_len,)

        if pixel_values is not None:
            # Get the actual IMG_CONTEXT token id.
            img_ctx_id = getattr(mt.model, "img_context_token_id", None)
            if img_ctx_id is None:
                img_ctx_id = mt.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
            if img_ctx_id is None or img_ctx_id == mt.tokenizer.unk_token_id:
                img_ctx_id = _IMG_CONTEXT_TOKEN_ID_FALLBACK  # last-resort fallback
            num_img_tokens = num_patches * 256  # 256 tokens per 448×448 tile

            # Build: [BOS] [IMG_CONTEXT * N] [newline] [text tokens]
            # This mirrors what model.chat() does internally.
            img_token_ids = torch.full(
                (num_img_tokens,), img_ctx_id, dtype=torch.long
            )
            newline_id = mt.tokenizer.convert_tokens_to_ids("\n")
            if newline_id == mt.tokenizer.unk_token_id:
                newline_id = mt.tokenizer.convert_tokens_to_ids("</s>")
            newline_tensor = torch.tensor([newline_id], dtype=torch.long)

            # Prepend image tokens before the text (skip BOS from text_ids if present)
            bos_id = mt.tokenizer.bos_token_id
            if bos_id is not None and len(text_ids) > 0 and text_ids[0] == bos_id:
                combined = torch.cat([
                    text_ids[:1],        # BOS
                    img_token_ids,       # IMG_CONTEXT * N
                    newline_tensor,      # \n
                    text_ids[1:],        # rest of text
                ])
            else:
                combined = torch.cat([img_token_ids, newline_tensor, text_ids])

            input_ids = combined.unsqueeze(0).to(mt.device)
        else:
            input_ids = text_ids.unsqueeze(0).to(mt.device)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids).to(mt.device),
            "_raw_prompt": prompt,  # metadata only, stripped before forward/generate
        }
        if pixel_values is not None:
            inputs["pixel_values"] = pixel_values
            inputs["image_flags"] = torch.tensor(
                [1] * num_patches, dtype=torch.long,
            ).to(mt.device)
        return inputs

    # -- Architecture accessors ----------------------------------------

    def get_decoder_layer(self, mt, layer_idx):
        return mt.model.language_model.model.layers[layer_idx]

    def get_vision_layer(self, mt, layer_idx):
        return mt.model.vision_model.encoder.layers[layer_idx]

    def get_final_norm(self, mt):
        return mt.model.language_model.model.norm

    def num_decoder_layers(self, mt):
        return len(mt.model.language_model.model.layers)

    def num_vision_layers(self, mt):
        return len(mt.model.vision_model.encoder.layers)

    # -- Generation ----------------------------------------------------

    def generate(self, mt, inputs, max_new_tokens=20):
        """Generate by calling language_model.generate() with inputs_embeds.

        We cannot pass pixel_values or image_flags to language_model.generate()
        because GenerationMixin validates kwargs and rejects unknown keys.
        Instead, we pre-compute the input embeddings (with ViT features already
        injected) via a forward pass through InternVLChatModel, then pass those
        embeddings directly to language_model.generate().

        This also ensures our decoder hooks (attached to language_model layers)
        fire correctly during generation.
        """
        # Step 1: compute input_embeds with ViT features injected via forward().
        # get_forward_inputs() strips _raw_prompt but keeps pixel_values + image_flags.
        forward_inputs = self.get_forward_inputs(inputs)
        with torch.no_grad():
            # Run one forward pass to get the input embeddings with image tokens replaced.
            # We hook into language_model to capture inputs_embeds.
            embeds_storage = {}
            def _capture_embeds(module, args, kwargs_inner):
                if "inputs_embeds" in kwargs_inner:
                    embeds_storage["inputs_embeds"] = kwargs_inner["inputs_embeds"].detach().clone()
                elif len(args) > 0:
                    # inputs_embeds may be passed positionally
                    pass
            hook = mt.model.language_model.register_forward_pre_hook(
                _capture_embeds, with_kwargs=True
            )
            try:
                mt.model(**forward_inputs)
            except Exception:
                pass  # we only need the embeddings, ignore any output errors
            finally:
                hook.remove()

        if "inputs_embeds" not in embeds_storage:
            # Fallback: compute embeddings manually using just the language model
            with torch.no_grad():
                inputs_embeds = mt.model.language_model.get_input_embeddings()(
                    inputs["input_ids"]
                )
        else:
            inputs_embeds = embeds_storage["inputs_embeds"]

        # Step 2: generate using language_model with pre-computed embeddings.
        with torch.no_grad():
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*pad_token_id.*")
                warnings.filterwarnings("ignore", message=".*GenerationMixin.*")
                output_ids = mt.model.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=mt.tokenizer.eos_token_id,
                )
        return mt.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        )

    # -- Image token range ---------------------------------------------

    def find_image_token_range(self, mt, inputs):
        """Return (start, end) indices of IMG_CONTEXT tokens in input_ids."""
        ids = inputs["input_ids"][0]
        # Resolve the token ID dynamically from the tokenizer each time,
        # since it differs between InternVL2 (151859) and InternVL2.5 (92546).
        img_ctx_id = mt.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        if img_ctx_id is None or img_ctx_id == mt.tokenizer.unk_token_id:
            img_ctx_id = _IMG_CONTEXT_TOKEN_ID_FALLBACK
        img_positions = torch.where(ids == img_ctx_id)[0]
        if len(img_positions) == 0:
            raise ValueError(
                f"No IMG_CONTEXT tokens (id={img_ctx_id}) found in input_ids. "
                "Check that the image placeholder was correctly inserted into the prompt."
            )
        return int(img_positions[0]), int(img_positions[-1]) + 1


# ======================================================================
# InternVL image preprocessing utilities (identical to InternVL2)
# ======================================================================

def _build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_ar = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target_ar)
        if diff < best_ratio_diff:
            best_ratio_diff = diff
            best_ratio = ratio
        elif diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_w, orig_h = image.size
    aspect = orig_w / orig_h

    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1)
         for i in range(1, n + 1) for j in range(1, n + 1)
         if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1],
    )
    best = _find_closest_aspect_ratio(aspect, target_ratios, orig_w, orig_h, image_size)
    tw, th = image_size * best[0], image_size * best[1]
    blocks = best[0] * best[1]

    resized = image.resize((tw, th), Image.LANCZOS)
    crops = []
    for idx in range(blocks):
        col = idx % (tw // image_size)
        row = idx // (tw // image_size)
        box = (col * image_size, row * image_size,
               (col + 1) * image_size, (row + 1) * image_size)
        crops.append(resized.crop(box))
    if use_thumbnail and len(crops) > 1:
        crops.append(image.resize((image_size, image_size), Image.LANCZOS))
    return crops


def _load_image(image, input_size=448, max_num=12):
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        image = image.convert("RGB")
    transform = _build_transform(input_size)
    tiles = _dynamic_preprocess(image, image_size=input_size, max_num=max_num)
    return torch.stack([transform(t) for t in tiles])
