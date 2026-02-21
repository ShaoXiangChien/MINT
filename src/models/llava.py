"""LLaVA-1.5-7B model adapter."""

import torch
from PIL import Image

from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model

from .base import BaseModelAdapter, ModelAndTokenizer
from src.utils.tokens import set_requires_grad

import re


class LLaVAAdapter(BaseModelAdapter):
    """Adapter for the LLaVA-1.5 family of models."""

    def load_model(self, model_path, device):
        torch.cuda.set_device(device)
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=model_name,
            device_map={"": device},
        )
        model = model.to(device)
        set_requires_grad(False, model)
        model.eval()

        mt = ModelAndTokenizer(
            model_name=model_path,
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            device=device,
        )
        return mt

    def prepare_inputs(self, prompt, image, mt):
        qs = self._format_prompt(prompt, mt.model, mt.model.name_or_path
                                 if hasattr(mt.model, "name_or_path") else "v1")
        input_ids = tokenizer_image_token(
            qs, mt.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(mt.device)

        images_tensor = process_images(
            [image], mt.image_processor, mt.model.config
        ).to(mt.device, dtype=torch.float16)

        return {
            "input_ids": input_ids,
            "images": images_tensor,
            "image_sizes": [image.size],
        }

    # -- Architecture accessors ----------------------------------------

    def get_decoder_layer(self, mt, layer_idx):
        return mt.model.model.layers[layer_idx]

    def get_vision_layer(self, mt, layer_idx):
        return mt.model.model.vision_tower.vision_tower.vision_model.encoder.layers[layer_idx]

    def get_final_norm(self, mt):
        return mt.model.model.norm

    def num_decoder_layers(self, mt):
        return len(mt.model.model.layers)

    def num_vision_layers(self, mt):
        return len(mt.model.model.vision_tower.vision_tower.vision_model.encoder.layers)

    # -- Generation ----------------------------------------------------

    def generate(self, mt, inputs, max_new_tokens=20):
        with torch.no_grad():
            output_ids = mt.model.generate(
                inputs["input_ids"],
                images=inputs["images"],
                image_sizes=inputs["image_sizes"],
                max_new_tokens=max_new_tokens,
                pad_token_id=mt.tokenizer.pad_token_id,
            )
        return mt.tokenizer.decode(output_ids[0][len(inputs["input_ids"][0]):],
                                   skip_special_tokens=True)

    # -- Image token range ---------------------------------------------

    def find_image_token_range(self, mt, inputs):
        ids = inputs["input_ids"][0]
        img_positions = torch.where(ids == IMAGE_TOKEN_INDEX)[0]
        if len(img_positions) == 0:
            raise ValueError("No image tokens found in input_ids")
        return int(img_positions[0]), int(img_positions[-1]) + 1

    # -- Helpers -------------------------------------------------------

    @staticmethod
    def _format_prompt(prompt, model, model_path):
        """Wrap a plain-text prompt into the LLaVA conversation template."""
        qs = prompt
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        if "llama-2" in model_path.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in model_path.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in model_path.lower():
            conv_mode = "chatml_direct"
        elif "v1" in model_path.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_path.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()
