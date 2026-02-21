import os
import sys
from pathlib import Path

# Get the absolute path of the current script
current_script_path = Path(__file__).parent.absolute()
# Add the LLaVA directory to Python path
sys.path.append("/home/sc305/VLM/LLaVA")
# Change to LLaVA directory
os.chdir("/home/sc305/VLM/LLaVA")

from PIL import Image
import numpy as np
import torch
from llava.constants import IMAGE_TOKEN_INDEX
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from patchscopes_utils import inspect_vision_in_lm
import torch
from general_utils import ModelAndTokenizer, process_images
import json
from tqdm import tqdm
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates
import re

PATH_PREFIX = "/home/sc305/VLM/Final Experiments/spatial-relationship/"

def load_vlm(model_path, device):
    # Force PyTorch to use only the specified device
    torch.cuda.set_device(device)
    
    # Clear any existing CUDA cache
    torch.cuda.empty_cache()
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path),
        device=device
    )

    # Comprehensive device enforcement
    model = model.to(device)
    
    # Recursively move ALL model components to target device
    def enforce_device_recursive(module, target_device):
        """Aggressively enforce device placement for all components"""
        module.to(target_device)
        
        # Move all parameters
        for param in module.parameters():
            param.data = param.data.to(target_device)
            if param.grad is not None:
                param.grad = param.grad.to(target_device)
        
        # Move all buffers
        for buffer in module.buffers():
            buffer.data = buffer.data.to(target_device)
        
        # Recursively handle children
        for child in module.children():
            enforce_device_recursive(child, target_device)
    
    enforce_device_recursive(model, device)

    # Additional verification with error raising
    def verify_device_strict(module, target_device, path=""):
        """Strict device verification that raises errors"""
        device_type = target_device.split(':')[0] if ':' in target_device else target_device
        device_index = int(target_device.split(':')[1]) if ':' in target_device else 0
        
        for name, param in module.named_parameters(recurse=False):
            if param.device.type != device_type or param.device.index != device_index:
                raise RuntimeError(f"Parameter {path}.{name} on {param.device}, expected {target_device}")
        
        for name, buffer in module.named_buffers(recurse=False):
            if buffer.device.type != device_type or buffer.device.index != device_index:
                raise RuntimeError(f"Buffer {path}.{name} on {buffer.device}, expected {target_device}")
        
        for name, child in module.named_children():
            verify_device_strict(child, target_device, f"{path}.{name}" if path else name)

    # Verify all components are on correct device
    verify_device_strict(model, device)
    
    dtype = torch.float16

    mt = ModelAndTokenizer(
        model_path,
        model=model,
        tokenizer=tokenizer,
        low_cpu_mem_usage=False,
        image_processor=image_processor,
        device=device,
        torch_dtype=dtype
    )

    # Ensure model is in eval mode and all components are properly placed
    mt.model.eval()
    
    # Add device enforcement method to mt
    def enforce_tensor_device(tensor):
        """Helper method to ensure tensor is on correct device"""
        return tensor.to(device) if tensor is not None else None
    
    mt.enforce_tensor_device = enforce_tensor_device
    
    return mt

def load_cate_mapping(mapping_path):
    with open(mapping_path, "r") as f:
        category_map = json.load(f)

    return category_map

def text_prompt_to_qs(model, model_path, prompt):
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

def inference(mt, prompt, images):
    image_tensor = process_images(images, mt.image_processor, mt.model.config).to(mt.device, dtype=torch.float16)
    image_tensor = image_tensor.to(mt.device, dtype=torch.float16)
    image_sizes = [image.size for image in images]

    input_ids_target = (
        tokenizer_image_token(prompt, mt.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(mt.device)
    )

    inp = {
        "input_ids": input_ids_target,
        "images": image_tensor,
        "image_sizes": image_sizes
    }

    with torch.no_grad():
        output_ids = mt.model.generate(
            inp["input_ids"],
            images=inp["images"],
            image_sizes=inp["image_sizes"],
            do_sample=False,
            temperature=0.7,
            max_new_tokens=10,
            pad_token_id=mt.tokenizer.eos_token_id
            )
    output = mt.tokenizer.decode(output_ids[0][1:])

    return output

if __name__ == "__main__":
    # CRITICAL: Set device explicitly and ensure CUDA_VISIBLE_DEVICES consistency
    import os
    
    # Force single GPU usage if you want to avoid multi-GPU issues
    # Uncomment the next line to use only GPU 0
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Set the default device for all tensor operations
    torch.cuda.set_device(device)
    
    results_path = "/home/sc305/VLM/Final Experiments/spatial-relationship/results/llava_baseline_results.json"
    model_path = "liuhaotian/llava-v1.5-7b"

    with open("/home/sc305/VLM/Final Experiments/spatial-relationship/data/controlled_images_dataset.json") as f:
        data = json.load(f)
    # Load vision-language model with enhanced device management
    mt = load_vlm(model_path, device)
   
    exp_results = []
    pbar = tqdm(data)
    count = 0

    for item in pbar:
        source_image = Image.open(PATH_PREFIX + item["image_path"])
        target_image = Image.new('RGB', source_image.size, (0, 0, 0))
        
        for idx, caption in enumerate(item["caption_options"]):
            prompt_source = text_prompt_to_qs(mt.model, model_path, "")
            prompt_target = text_prompt_to_qs(mt.model, model_path, f"Is {caption}? only answer YES or NO")

            result = inference(mt, prompt_target, [source_image])
            prediction = 1 if "yes" in result.lower() else 0
            correct = prediction == 1 if idx == 0 else prediction == 0
            
            exp_results.append({
                "image_id": count,
                "label": "yes" if idx == 0 else "no",
                "prediction": prediction,
                "correct": correct,
                "result": result
            })


        count += 1
        
        # Periodic cleanup
        if count % 10 == 0:
            torch.cuda.empty_cache()

    with open(results_path, "w") as f:
        json.dump(exp_results, f)
    
    
    
    
