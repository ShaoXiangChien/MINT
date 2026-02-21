import os
import sys
from pathlib import Path

# Get the absolute path of the current script
CURRENT_SCRIPT_PATH = Path(__file__).parent.absolute()
# Add the LLaVA directory to Python path
sys.path.append("/home/sc305/VLM/LLaVA")
# Change to LLaVA directory
os.chdir("/home/sc305/VLM/LLaVA")


from PIL import Image
import numpy as np
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
import torch
from general_utils import ModelAndTokenizer
from patchscopes_utils import inspect_vision_in_lm
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
import glob

PATH_PREFIX = "/home/sc305/VLM/Final Experiments/negation/"


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

def load_category_mapping(mapping_path):
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


def get_category_from_filename(filename):
    """Extract category ID from filename like '1000_000000508602.jpg'"""
    base = os.path.splitext(filename)[0]
    category_id = base.split("_")[0] if "_" in base else "0"
    return category_id


def evaluate_response(response, category):
    """Evaluate if the response indicates presence (yes) or absence (no)"""
    response_lower = response.lower().strip()
    
    # Check for explicit yes/no answers
    if "yes" in response_lower and "no" not in response_lower:
        return 1  # Presence detected
    elif "no" in response_lower and "yes" not in response_lower:
        return 0  # Absence detected
    else:
        # Fallback: check if category is mentioned positively
        if category.lower() in response_lower:
            return 1
        else:
            return 0





if __name__ == "__main__":
    # Device and model paths
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_path = "liuhaotian/llava-v1.5-7b"
    results_path = os.path.join(CURRENT_SCRIPT_PATH, "../llava_results.json")


    # Data paths
    dataset_dir = os.path.join(CURRENT_SCRIPT_PATH, "../../test_images_split/dataset")  # Directory with source images
    test_images_dir = os.path.join(CURRENT_SCRIPT_PATH, "../../test_images_split/test_images")  # Directory with target images
    
    category_mapping_path = os.path.join(CURRENT_SCRIPT_PATH, "../../test_images_split/neg_object_map.json")
    # Load category mapping
    category_mapping = load_category_mapping(category_mapping_path)
    
    # Get all image files from dataset directory
    image_files = []
    if os.path.exists(dataset_dir):
        image_files = glob.glob(os.path.join(dataset_dir, "*.jpg")) + glob.glob(os.path.join(dataset_dir, "*.png"))
    
    if not image_files:
        print(f"No images found in {dataset_dir}")
        print("Available directories:")
        for root, dirs, files in os.walk("."):
            if any(f.endswith(('.jpg', '.png')) for f in files):
                print(f"  {root}: {len([f for f in files if f.endswith(('.jpg', '.png'))])} images")
        exit(1)

    # Model
    mt = load_vlm(model_path, device)
    target_image = Image.open(os.path.join(CURRENT_SCRIPT_PATH, "../../test_images_split/test_images/blank_384x384.png")).resize((384, 384))

    # Load existing results if available
    exp_results = []
    if os.path.exists(results_path):
        try:
            with open(results_path, "r") as f:
                exp_results = json.load(f)
            print(f"Loaded {len(exp_results)} existing results from {results_path}")
        except Exception as e:
            print(f"Error loading existing results: {e}")
            exp_results = []
    
    # Get list of already processed filenames
    processed_files = set([result['filename'] for result in exp_results])
    print(f"Already processed {len(processed_files)} images")
    
    # Filter out already processed images
    remaining_images = [img for img in image_files if os.path.basename(img) not in processed_files]
    print(f"Remaining images to process: {len(remaining_images)}")
    
    sample_id = len(exp_results)
    
    pbar = tqdm(remaining_images)

    for img_path in pbar:
        print(f"Processing {img_path}")
        # Extract filename from path
        fname = os.path.basename(img_path)
        
        # Extract category from filename (assuming format like "01_000000108864.jpg")
        # Get the number part before underscore
        base_name = os.path.splitext(fname)[0]
        if '_' in base_name:
            category_id = base_name.split('_')[0]
        else:
            category_id = "0"  # Default category
        
        category = category_mapping.get(category_id, "unknown")
        prompt_source = text_prompt_to_qs(mt.model, model_path, "")
        prompt_target = text_prompt_to_qs(mt.model, model_path, f"Is there a {category} in the image? Answer yes or no.")

        try:
            # Load source image from file path
            source_image = Image.open(img_path).resize((384, 384))
            
            # Layer sweep for patching
            layer_results = []
            for layer_source in range(0, 32, 3):
                source_layer_results = []
                for layer_target in range(0, 32, 3):
                    result = inspect_vision_in_lm(
                        mt,
                        prompt_source,
                        prompt_target,
                        [source_image],  # Source image from dataset
                        [target_image],  # Blank target image
                        layer_source,
                        layer_target,
                    )
                    
    
                    # Check if response indicates presence (yes) or absence (no)
                    response_lower = result.lower().strip()
                    if "yes" in response_lower and "no" not in response_lower:
                        prediction = 1  # Presence detected
                    elif "no" in response_lower and "yes" not in response_lower:
                        prediction = 0  # Absence detected
                    else:
                        # Fallback: check if category is mentioned positively
                        prediction = 1 if category in response_lower else 0
                    source_layer_results.append(prediction)
                layer_results.append(source_layer_results)

            # Save results for this sample
            exp_results.append({
                "sample_id": sample_id,
                "filename": fname,
                "category": category,
                "results": layer_results
            })

            # Save after each sample
            with open(results_path, "w") as f:
                json.dump(exp_results, f)
            
            sample_id += 1 

        except Exception as e:
            print(f"Error processing {fname}: {e}")
            continue

    print(f"Successfully processed {len(exp_results)} samples")
    
    # Final save
    with open(results_path, "w") as f:
        json.dump(exp_results, f)
