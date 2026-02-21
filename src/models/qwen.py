"""Qwen2-VL-7B model adapter."""

import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from .base import BaseModelAdapter, ModelAndTokenizer
from src.utils.tokens import set_requires_grad

# Qwen2-VL image start / end token ids
_IMG_START_TOKEN_ID = 151652
_IMG_END_TOKEN_ID = 151653


class QwenAdapter(BaseModelAdapter):
    """Adapter for the Qwen2-VL family of models."""

    def load_model(self, model_path, device):
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map=device,
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
        """Build inputs using the Qwen2-VL chat template."""
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
        return mt.model.model.layers[layer_idx]

    def get_vision_layer(self, mt, layer_idx):
        return mt.model.visual.blocks[layer_idx]

    def get_final_norm(self, mt):
        return mt.model.model.norm

    def num_decoder_layers(self, mt):
        return len(mt.model.model.layers)

    def num_vision_layers(self, mt):
        return len(mt.model.visual.blocks)

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
        ids = inputs["input_ids"][0]
        start_positions = torch.where(ids == _IMG_START_TOKEN_ID)[0]
        end_positions = torch.where(ids == _IMG_END_TOKEN_ID)[0]
        if len(start_positions) == 0 or len(end_positions) == 0:
            raise ValueError("No image tokens found in input_ids")
        return int(start_positions[0]), int(end_positions[0]) + 1
