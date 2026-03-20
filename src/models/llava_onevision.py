"""LLaVA-OneVision model adapter."""

import torch
from PIL import Image
from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor

from .base import BaseModelAdapter, ModelAndTokenizer
from src.utils.tokens import set_requires_grad

# LLaVA-OneVision specific image token ID
# The default image token index is 151646 according to the configuration
_IMAGE_TOKEN_ID = 151646

class LLaVAOneVisionAdapter(BaseModelAdapter):
    """Adapter for the LLaVA-OneVision family of models.
    
    TODO(User): This adapter has been implemented based on the model architecture, 
    but it requires GPU inference to fully validate the exact token alignment and 
    generation behavior.
    """

    def load_model(self, model_path, device):
        # TODO(User): LLaVA-OneVision requires significant VRAM.
        # Make sure your device has enough memory (e.g., 16GB+ for 7B model in fp16).
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype=torch.float16, 
            device_map=device,
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
        """Build inputs using the LLaVA-OneVision chat template."""
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        
        # Apply chat template
        text_prompt = mt.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        
        # Process the full multimodal input
        inputs = mt.processor(
            images=image, 
            text=text_prompt, 
            return_tensors="pt"
        )
        
        return inputs.to(mt.device, torch.float16)

    # -- Architecture accessors ----------------------------------------

    def get_decoder_layer(self, mt, layer_idx):
        # LLaVA-OneVision wraps Qwen2 under model.language_model.model
        return mt.model.model.language_model.model.layers[layer_idx]

    def get_vision_layer(self, mt, layer_idx):
        # SigLIP vision tower path
        return mt.model.model.vision_tower.vision_model.encoder.layers[layer_idx]

    def get_final_norm(self, mt):
        return mt.model.model.language_model.model.norm

    def num_decoder_layers(self, mt):
        return len(mt.model.model.language_model.model.layers)

    def num_vision_layers(self, mt):
        return len(mt.model.model.vision_tower.vision_model.encoder.layers)

    # -- Generation ----------------------------------------------------

    def generate(self, mt, inputs, max_new_tokens=20):
        with torch.no_grad():
            output_ids = mt.model.generate(**inputs, max_new_tokens=max_new_tokens)
            
        # Decode only the newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0][input_len:]
        
        return mt.processor.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    # -- Image token range ---------------------------------------------

    def find_image_token_range(self, mt, inputs):
        """Find the start and end index of image tokens in the input sequence.
        
        TODO(User): This logic assumes contiguous image tokens. If LLaVA-OneVision
        interleaves image tokens with newline tokens or uses multiple chunks 
        (like anyres-9), this logic might need refinement during inference validation.
        """
        ids = inputs["input_ids"][0]
        
        # The model config specifies 151646 as the image token index
        image_token_id = mt.model.config.image_token_index
        
        img_positions = torch.where(ids == image_token_id)[0]
        
        if len(img_positions) == 0:
            raise ValueError(f"No image tokens (ID {image_token_id}) found in input_ids")
            
        return int(img_positions[0]), int(img_positions[-1]) + 1
