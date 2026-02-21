"""
LLaVA Activation Patching Experiment

This script performs activation patching experiments on LLaVA model,
extracting hidden states from a source image and patching them into
a target (blank) image to test information transfer across layers.
"""

import os
import sys
from pathlib import Path

# Get the absolute path of the current script
current_script_path = Path(__file__).parent.absolute()
# Add the LLaVA directory to Python path
sys.path.append("/home/sc305/VLM/LLaVA")
# Change to LLaVA directory
os.chdir("/home/sc305/VLM/LLaVA")

import torch
import json
import re
from PIL import Image
from tqdm import tqdm
from datasets import load_from_disk

# LLaVA specific imports
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token, process_images
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates

# Custom utility
from general_utils import ModelAndTokenizer


def load_vlm(model_path, device):
    """
    Load LLaVA vision-language model.
    
    Args:
        model_path: Path or HuggingFace model name for LLaVA
        device: Device to load the model on (e.g., "cuda:0")
    
    Returns:
        mt: ModelAndTokenizer wrapper containing the model and related components
    """
    # Set the CUDA device
    torch.cuda.set_device(device)
    
    # Load the pretrained LLaVA model
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path),
        device_map={"": device}
    )
    
    # Move model to device and set dtype
    model = model.to(device)
    dtype = torch.float16
    
    # Create ModelAndTokenizer wrapper
    mt = ModelAndTokenizer(
        model_path,
        model=model,
        tokenizer=tokenizer,
        low_cpu_mem_usage=False,
        image_processor=image_processor,
        device=device,
        torch_dtype=dtype
    )
    
    mt.model.eval()
    return mt


def load_cate_mapping(mapping_path):
    """
    Load category ID to name mapping from JSON file.
    
    Args:
        mapping_path: Path to the JSON mapping file
    
    Returns:
        category_map: Dictionary mapping category IDs to names
    """
    with open(mapping_path, "r") as f:
        category_map = json.load(f)
    return category_map


def text_prompt_to_qs(prompt, model, model_path):
    """
    Convert a text prompt to the format expected by LLaVA.
    
    This function adds the image token and wraps the prompt in the
    appropriate conversation template based on the model type.
    
    Args:
        prompt: The text prompt to convert
        model: The LLaVA model (to check config)
        model_path: Model path (to determine conversation mode)
    
    Returns:
        Formatted prompt string ready for tokenization
    """
    qs = prompt
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    
    # Add image placeholder if not present
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
    
    # Determine conversation mode based on model path
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
    
    # Build conversation from template
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    
    return conv.get_prompt()


def prepare_inputs(prompt, image, mt, model_path):
    """
    Prepare inputs for LLaVA model.
    
    Args:
        prompt: Text prompt (will be formatted with text_prompt_to_qs)
        image: PIL Image
        mt: ModelAndTokenizer wrapper
        model_path: Model path for conversation mode selection
    
    Returns:
        Dictionary containing input_ids, images tensor, and image_sizes
    """
    # Format the prompt
    formatted_prompt = text_prompt_to_qs(prompt, mt.model, model_path)
    
    # Process the image
    images = [image]
    images_tensor = process_images(
        images, 
        mt.image_processor, 
        mt.model.config
    ).to(mt.device, dtype=torch.float16)
    
    # Get image sizes
    image_sizes = [img.size for img in images]
    
    # Tokenize the prompt
    input_ids = tokenizer_image_token(
        formatted_prompt, 
        mt.tokenizer, 
        IMAGE_TOKEN_INDEX, 
        return_tensors="pt"
    ).unsqueeze(0).to(mt.device)
    
    return {
        "input_ids": input_ids,
        "images": images_tensor,
        "image_sizes": image_sizes
    }


def get_tail_length(input_ids, tokenizer):
    """
    Calculate the tail length (text tokens after image tokens).
    
    The tail includes the question text and "ASSISTANT:" token.
    Token 3624 corresponds to "Is" which marks the start of the question.
    
    Args:
        input_ids: Tokenized input tensor
        tokenizer: Tokenizer to decode tokens
    
    Returns:
        tail_len: Number of tokens from "Is" to the end
    """
    # Token 3624 is "Is" in LLaVA's tokenizer
    is_token_id = 3624
    
    # Find the position of "Is" token
    is_positions = torch.where(input_ids[0] == is_token_id)[0]
    
    if len(is_positions) > 0:
        is_pos = is_positions[0].item()
        tail_len = len(input_ids[0]) - is_pos
    else:
        # Fallback: use last 19 tokens (typical for short prompts)
        tail_len = 19
    
    return tail_len


def capture_emb(mt, inp_source, layer_source):
    """
    Capture hidden state embeddings from a specific layer.
    
    Args:
        mt: ModelAndTokenizer wrapper
        inp_source: Input dictionary for the source image
        layer_source: Layer index to capture activations from
    
    Returns:
        Tensor containing the layer's output hidden states
    """
    extracted_activations = {}
    
    def capture_hs(module, input, output):
        nonlocal extracted_activations
        if "layer_output" not in extracted_activations:
            # Clone to avoid reference issues
            extracted_activations["layer_output"] = output[0].clone()
    
    # Register hook on the specified layer
    # Note: LLaVA uses model.model.layers (not model.language_model.layers like Qwen)
    capture_hook = mt.model.model.layers[layer_source].register_forward_hook(capture_hs)
    
    with torch.no_grad():
        _ = mt.model(**inp_source)
    
    capture_hook.remove()
    
    return extracted_activations["layer_output"]


def patch_text(mt, inp_source, inp_target, layer_target, cache_hs):
    """
    Patch hidden states from source into target at a specific layer.
    
    This function replaces the tail tokens (question + ASSISTANT:) in the
    target's hidden states with those from the source, then generates output.
    
    Args:
        mt: ModelAndTokenizer wrapper
        inp_source: Input dictionary for source (needed for tail_len calculation)
        inp_target: Input dictionary for the target image
        layer_target: Layer index to apply the patch
        cache_hs: Cached hidden states from the source
    
    Returns:
        Generated text output after patching
    """
    patched = False
    
    # Pre-calculate tail length
    tail_len = get_tail_length(inp_source["input_ids"], mt.tokenizer)
    
    def patch_hs(module, input, output):
        nonlocal patched
        
        # Only patch on first forward pass (not during generation)
        if patched:
            return output
        
        # Patch the tail tokens (question + ASSISTANT:)
        for i in range(tail_len):
            output[0][0][(-1) * i] = cache_hs[0][(-1) * i]
        
        patched = True
        return output
    
    # Register hook on the target layer
    patch_hook = mt.model.model.layers[layer_target].register_forward_hook(patch_hs)
    
    with torch.no_grad():
        output_ids = mt.model.generate(
            inp_target["input_ids"],
            images=inp_target["images"],
            image_sizes=inp_target["image_sizes"],
            max_new_tokens=20,
            pad_token_id=mt.tokenizer.pad_token_id,
        )
    
    patch_hook.remove()
    
    # Decode output (skip the first token which is typically BOS)
    output = mt.tokenizer.decode(output_ids[0][1:], skip_special_tokens=True)
    
    return output


if __name__ == "__main__":
    # ===== Configuration =====
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    model_path = "liuhaotian/llava-v1.5-7b"
    
    # Paths for data and results
    data_path = "../data/full_sample"
    mapping_path = "../data/instances_category_map.json"
    results_path = str(current_script_path / "results" / "llava_results.json")
    
    # Ensure results directory exists
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    # ===== Load Model and Data =====
    print("Loading LLaVA model...")
    mt = load_vlm(model_path, device)
    
    print("Loading dataset...")
    sample_trainset = load_from_disk(data_path)
    category_mapping = load_cate_mapping(mapping_path)
    
    # Create blank target image
    target_image = Image.new("RGB", (2, 2), (0, 0, 0))
    
    # ===== Run Experiment =====
    exp_results = []
    num_layers = 32  # LLaVA-v1.5-7B has 32 layers
    layer_step = 2   # Sample every 2nd layer
    
    pbar = tqdm(sample_trainset)
    count = 0
    
    for sample in pbar:
        results = []
        
        # Get category name from mapping
        category = category_mapping[str(sample["annotations"]["category_id"][0])]
        prompt = f"Is there a {category} in the image? only answer with yes or no"
        
        # Prepare inputs for source (real image) and target (blank image)
        try:
            inp_source = prepare_inputs(prompt, sample["image"], mt, model_path)
            inp_target = prepare_inputs(prompt, target_image, mt, model_path)
        except Exception as e:
            print(f"Error preparing inputs for sample {count}: {e}")
            count += 1
            continue
        
        # Run patching experiment across layer combinations
        try:
            for layer_source in range(0, num_layers, layer_step):
                layer_results = []
                
                # Capture embeddings from source layer
                cache_hs = capture_emb(mt, inp_source, layer_source)
                
                for layer_target in range(0, num_layers, layer_step):
                    # Patch and generate
                    result = patch_text(mt, inp_source, inp_target, layer_target, cache_hs)
                    
                    # Check if category is mentioned (success) or "yes" is in response
                    prediction = 1 if ("yes" in result.lower() or category.lower() in result.lower()) else 0
                    layer_results.append(prediction)
                
                results.append(layer_results)
                
        except Exception as e:
            print(f"Error processing sample {count}: {e}")
            count += 1
            continue
        
        # Save intermediate results periodically
        if count % 100 == 0 and count > 0:
            with open(results_path, "w") as f:
                json.dump(exp_results, f, indent=2)
            pbar.set_description(f"Saved {len(exp_results)} results")
        
        # Store results if valid
        if len(results) > 0:
            exp_results.append({
                "sample_id": count,
                "category": category,
                "results": results
            })
        
        count += 1
    
    # ===== Save Final Results =====
    print(f"\nExperiment complete!")
    print(f"Successfully processed samples: {len(exp_results)}")
    
    with open(results_path, "w") as f:
        json.dump(exp_results, f, indent=2)
    
    print(f"Results saved to: {results_path}")

