"""MiniCPM-V 2.6 (8B) model adapter.

Architecture overview:
  - Top-level class: MiniCPMV  (custom_code, loaded via AutoModel)
  - Language backbone: Qwen2ForCausalLM stored at  mt.model.llm
  - Vision encoder:   SiglipVisionTransformer at   mt.model.vpm
  - Resampler:        Resampler at                 mt.model.resampler
                      (compresses SigLIP patches → 64 query tokens per tile)

Module paths:
  - Decoder layers:   mt.model.llm.model.layers[i]
  - Vision layers:    mt.model.vpm.encoder.layers[i]   (27 layers; last dropped if drop_vision_last_layer=True)
  - Final LM norm:    mt.model.llm.model.norm

Image token handling:
  MiniCPM-V does NOT use a single image_token_id in input_ids.
  Instead, the processor returns an `image_bound` tensor of shape (B, N, 2)
  where each row is [start_idx, end_idx) of a resampled image token block.
  Each image produces `config.query_num` (=64) consecutive tokens in input_ids.

  For MINT patching purposes we treat the entire span from
  image_bound[0][0][0] to image_bound[0][-1][1] as the image token range.

  NOTE: MiniCPM-V uses a custom forward interface (data dict) rather than
  standard HuggingFace kwargs. The adapter wraps this transparently.

Default HF model path: openbmb/MiniCPM-V-2_6
"""

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from .base import BaseModelAdapter, ModelAndTokenizer
from src.utils.tokens import set_requires_grad


class MiniCPMAdapter(BaseModelAdapter):
    """Adapter for MiniCPM-V 2.6 (8B)."""

    def load_model(self, model_path, device):
        # MiniCPM-V uses trust_remote_code because it ships custom modeling code
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=torch.float16,
            device_map=device,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        set_requires_grad(False, model)
        model.eval()

        mt = ModelAndTokenizer(
            model_name=model_path,
            model=model,
            processor=tokenizer,   # MiniCPM-V uses tokenizer directly
            device=device,
        )
        mt.tokenizer = tokenizer
        return mt

    def prepare_inputs(self, prompt, image, mt):
        """Build inputs using MiniCPM-V's native chat interface.

        Returns a dict with keys: input_ids, pixel_values, tgt_sizes, image_bound.
        All tensors are moved to mt.device.
        """
        msgs = [{"role": "user", "content": [image, prompt]}]

        # model.build_conversation_input_ids is the official preprocessing API
        inputs = mt.model.build_conversation_input_ids(
            mt.tokenizer,
            query=prompt,
            history=[],
            images=[image],
            return_tensors="pt",
        )
        # Move all tensor values to device
        inputs = {
            k: v.to(mt.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        return inputs

    # -- Architecture accessors ----------------------------------------

    def get_decoder_layer(self, mt, layer_idx):
        # Language backbone is Qwen2ForCausalLM stored at .llm
        # Its transformer layers are at .llm.model.layers
        return mt.model.llm.model.layers[layer_idx]

    def get_vision_layer(self, mt, layer_idx):
        # SigLIP vision encoder layers (27 by default)
        return mt.model.vpm.encoder.layers[layer_idx]

    def get_final_norm(self, mt):
        return mt.model.llm.model.norm

    def num_decoder_layers(self, mt):
        return len(mt.model.llm.model.layers)

    def num_vision_layers(self, mt):
        return len(mt.model.vpm.encoder.layers)

    # -- Generation ----------------------------------------------------

    def generate(self, mt, inputs, max_new_tokens=20):
        """Run generation using MiniCPM-V's native generate interface."""
        with torch.no_grad():
            output_ids = mt.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        # Decode only the newly generated tokens (strip the prompt)
        input_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[:, input_len:]
        return mt.tokenizer.decode(new_tokens[0], skip_special_tokens=True)

    # -- Image token range ---------------------------------------------

    def find_image_token_range(self, mt, inputs):
        """Return (start, end) of the image token block in input_ids.

        MiniCPM-V uses image_bound: a list of (B, N, 2) tensors where each
        row is [start_idx, end_idx) of a resampled image block.
        We return the span covering all image blocks for the first sample.

        TODO: Verify with actual model output that image_bound is present
        and has the expected shape. If prepare_inputs does not return
        image_bound, you may need to call model.build_conversation_input_ids
        with a different argument or extract it from the processor output.
        """
        image_bound = inputs.get("image_bound", None)
        if image_bound is None:
            raise ValueError(
                "image_bound not found in inputs. "
                "MiniCPM-V requires image_bound to locate image tokens. "
                "Check that prepare_inputs returns image_bound correctly."
            )
        # image_bound shape: (B, N, 2) or list of tensors
        if isinstance(image_bound, (list, tuple)):
            bounds = image_bound[0]  # first sample
        else:
            bounds = image_bound[0]

        if len(bounds) == 0:
            raise ValueError("image_bound is empty — no image tokens found.")

        start = int(bounds[0][0])
        end = int(bounds[-1][1])
        return start, end
