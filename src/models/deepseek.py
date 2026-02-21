"""DeepSeek-VL2-Tiny model adapter."""

import torch
from PIL import Image

from .base import BaseModelAdapter, ModelAndTokenizer
from src.utils.tokens import set_requires_grad


class DeepSeekAdapter(BaseModelAdapter):
    """Adapter for the DeepSeek-VL2 family of models."""

    # Image token id used by DeepSeek-VL2
    IMAGE_TOKEN_ID = 128815

    def load_model(self, model_path, device):
        import transformers

        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype="auto", device_map=device,
            trust_remote_code=True,
        )
        processor = transformers.AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True,
        )
        set_requires_grad(False, model)
        model.eval()

        mt = ModelAndTokenizer(
            model_name=model_path,
            model=model,
            tokenizer=getattr(processor, "tokenizer", None),
            processor=processor,
            device=device,
        )
        return mt

    def prepare_inputs(self, prompt, image, mt):
        """Build inputs using the DeepSeek conversation format."""
        if image is not None:
            conversation = [
                {"role": "<|User|>",
                 "content": f"<image>\n{prompt}",
                 "images": [None]},
                {"role": "<|Assistant|>", "content": ""},
            ]
            inputs = mt.processor(
                conversations=conversation, images=[image],
                return_tensors="pt",
            ).to(mt.device)
        else:
            placeholder = Image.new("RGB", (1, 1), (0, 0, 0))
            conversation = [
                {"role": "<|User|>", "content": prompt},
                {"role": "<|Assistant|>", "content": ""},
            ]
            inputs = mt.processor(
                conversations=conversation, images=[placeholder],
                return_tensors="pt",
            ).to(mt.device)
        return inputs

    # -- Architecture accessors ----------------------------------------

    def get_decoder_layer(self, mt, layer_idx):
        if hasattr(mt.model, "language_model"):
            return mt.model.language_model.model.layers[layer_idx]
        return mt.model.model.layers[layer_idx]

    def get_vision_layer(self, mt, layer_idx):
        return mt.model.vision.blocks[layer_idx]

    def get_final_norm(self, mt):
        if hasattr(mt.model, "language_model"):
            return mt.model.language_model.model.norm
        return mt.model.model.norm

    def num_decoder_layers(self, mt):
        if hasattr(mt.model, "language_model"):
            return len(mt.model.language_model.model.layers)
        return len(mt.model.model.layers)

    def num_vision_layers(self, mt):
        return len(mt.model.vision.blocks)

    # -- Generation ----------------------------------------------------

    def generate(self, mt, inputs, max_new_tokens=20):
        with torch.no_grad():
            # DeepSeek-VL2 uses prepare_inputs_embeds for generation
            if hasattr(mt.model, "prepare_inputs_embeds"):
                inputs_embeds = mt.model.prepare_inputs_embeds(**inputs)
                output_ids = mt.model.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    pad_token_id=mt.processor.tokenizer.eos_token_id,
                )
            else:
                output_ids = mt.model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                )
        return mt.processor.tokenizer.decode(
            output_ids[0], skip_special_tokens=True,
        )

    # -- Image token range ---------------------------------------------

    def find_image_token_range(self, mt, inputs):
        ids = inputs["input_ids"][0]
        img_positions = torch.where(ids == self.IMAGE_TOKEN_ID)[0]
        if len(img_positions) == 0:
            raise ValueError("No image tokens found in input_ids")
        return int(img_positions[0]), int(img_positions[-1]) + 1
