"""Qwen2.5-VL-7B model adapter.

Architecture differences from Qwen2-VL:
  - Class: Qwen2_5_VLForConditionalGeneration  (was Qwen2VLForConditionalGeneration)
  - Vision blocks: mt.model.model.visual.blocks  (NEW: extra .model hop; visual lives inside Qwen2_5_VLModel)
  - Decoder layers: mt.model.model.language_model.layers  (NEW: extra .language_model hop vs Qwen2-VL)
    Qwen2_5_VLForConditionalGeneration.model = Qwen2_5_VLModel
    Qwen2_5_VLModel.language_model           = Qwen2_5_VLTextModel
    Qwen2_5_VLTextModel.layers               = nn.ModuleList of decoder layers
  - Image token ID: 151655                      (was 151652/151653 start/end pair)
    Qwen2.5-VL uses a single image_token_id instead of start/end pair.
    The full image region is: first occurrence of 151655 → last occurrence of 151655 + 1.
  - Processor: same AutoProcessor API, but uses qwen_vl_utils for vision info.
  - Default HF model: Qwen/Qwen2.5-VL-7B-Instruct
"""

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from .base import BaseModelAdapter, ModelAndTokenizer
from src.utils.tokens import set_requires_grad

# Qwen2.5-VL uses a single image token id (not a start/end pair like Qwen2-VL)
_IMAGE_TOKEN_ID = 151655


class Qwen25Adapter(BaseModelAdapter):
    """Adapter for the Qwen2.5-VL family of models."""

    def load_model(self, model_path, device):
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.float16, device_map=device,
        )
        processor = AutoProcessor.from_pretrained(model_path)
        set_requires_grad(False, model)
        model.eval()

        mt = ModelAndTokenizer(
            model_name=model_path,
            model=model,
            processor=processor,
            device=device,
        )
        mt.tokenizer = processor.tokenizer
        return mt

    def prepare_inputs(self, prompt, image, mt):
        """Build inputs using the Qwen2.5-VL chat template."""
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]
        text = mt.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = mt.processor(
            text=[text], images=image_inputs,
            padding=True, return_tensors="pt",
        )
        return inputs.to(mt.device)

    # -- Architecture accessors ----------------------------------------

    def get_decoder_layer(self, mt, layer_idx):
        # Qwen2_5_VLForConditionalGeneration -> .model (Qwen2_5_VLModel)
        # -> .language_model (Qwen2_5_VLTextModel) -> .layers[i]
        return mt.model.model.language_model.layers[layer_idx]

    def get_vision_layer(self, mt, layer_idx):
        # Qwen2_5_VLForConditionalGeneration -> .model (Qwen2_5_VLModel) -> .visual -> .blocks[i]
        return mt.model.model.visual.blocks[layer_idx]

    def get_final_norm(self, mt):
        return mt.model.model.language_model.norm

    def num_decoder_layers(self, mt):
        return len(mt.model.model.language_model.layers)

    def num_vision_layers(self, mt):
        return len(mt.model.model.visual.blocks)

    # -- Generation ----------------------------------------------------

    def generate(self, mt, inputs, max_new_tokens=20):
        with torch.no_grad():
            output_ids = mt.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return mt.processor.batch_decode(
            [o[len(i):] for i, o in zip(inputs["input_ids"], output_ids)],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    # -- Image token range ---------------------------------------------

    def find_image_token_range(self, mt, inputs):
        """Return (start, end) index of the image token block in input_ids.

        Qwen2.5-VL uses a single image_token_id (151655) repeated for every
        image patch, so we take the first and last occurrence.
        """
        ids = inputs["input_ids"][0]
        img_token_id = getattr(mt.model.config, "image_token_id", _IMAGE_TOKEN_ID)
        positions = torch.where(ids == img_token_id)[0]
        if len(positions) == 0:
            raise ValueError(
                f"No image tokens (id={img_token_id}) found in input_ids. "
                "Check that the image was correctly preprocessed."
            )
        return int(positions[0]), int(positions[-1]) + 1
