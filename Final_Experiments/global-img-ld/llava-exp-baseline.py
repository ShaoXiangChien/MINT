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
from llava.mm_utils import tokenizer_image_token, process_images
from llava.constants import IMAGE_TOKEN_INDEX
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
import torch
from general_utils import ModelAndTokenizer
import math
import json
from datasets import load_from_disk
import cv2
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


def load_vlm(model_path, device):
    """
    Load the LLaVA vision-language model.
    This function is identical to the original experiment for consistency.
    
    The model loading process includes:
    - Loading the pretrained LLaVA model with tokenizer and image processor
    - Moving all model components to the specified device
    - Setting up the ModelAndTokenizer wrapper for easier access
    """
    # Load vision-language model
    torch.cuda.set_device(device)
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path),
        device=device
    )

    # 更徹底地確保所有模型組件都在正確的設備上
    model = model.to(device)
    
    # 遷移視覺模型
    if hasattr(model, 'model'):
        if hasattr(model.model, 'vision_tower'):
            # 確保主視覺塔在正確設備上
            model.model.vision_tower = model.model.vision_tower.to(device)
            
            # 遷移內部視覺塔
            if hasattr(model.model.vision_tower, 'vision_tower'):
                model.model.vision_tower.vision_tower = model.model.vision_tower.vision_tower.to(device)
                
                # 遷移視覺模型的所有子模塊
                if hasattr(model.model.vision_tower.vision_tower, 'vision_model'):
                    for module in model.model.vision_tower.vision_tower.vision_model.modules():
                        module.to(device)
                        
                # 遷移視覺投影器
                if hasattr(model.model.vision_tower, 'mm_projector'):
                    model.model.vision_tower.mm_projector = model.model.vision_tower.mm_projector.to(device)

    # 驗證設備遷移
    def verify_device(module, target_device):
        for name, child in module.named_children():
            for param in child.parameters():
                assert param.device.type == target_device.split(':')[0] and \
                       param.device.index == int(target_device.split(':')[1]), \
                       f"Parameter in {name} found on {param.device}, should be on {target_device}"
            verify_device(child, target_device)

    # 驗證所有組件都在正確的設備上
    try:
        verify_device(model, device)
    except AssertionError as e:
        print(f"Warning: Device verification failed: {e}")
        # 可以選擇是否在這裡 raise 錯誤
    
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

    mt.model.eval()
    return mt


def generate_baseline_response(mt, prompt, image, max_new_tokens=100, temperature=None, verbose=False):
    """
    Generate a baseline response from the LLaVA model WITHOUT any patching or layer intervention.
    
    This is the key difference from the original experiment:
    - No layer-wise patching or inspection using patchscopes
    - Direct generation from the model using the given image and prompt
    - Mimics standard VLM inference behavior that users would normally experience
    
    Args:
        mt: ModelAndTokenizer object containing the LLaVA model, tokenizer, and image processor
        prompt: Text prompt formatted for LLaVA conversation
        image: PIL Image to process
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (None for greedy decoding)
        verbose: Whether to print debug information
        
    Returns:
        Generated text response from the model
    """
    try:
        # Process the image using LLaVA's image processor
        # This step converts the PIL image into the format expected by the model
        image_tensor = process_images([image], mt.image_processor, mt.model.config).to(mt.device, dtype=torch.float16)
        
        # Tokenize the input prompt
        # The prompt should already be formatted with conversation template
        input_ids = tokenizer_image_token(prompt, mt.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(mt.device)
        
        
        if verbose:
            print(f"Input prompt: {prompt}")
            print(f"Input shape: {input_ids.shape}, Image shape: {image_tensor.shape}")
        
        # Generate response using standard model inference
        # This is pure model generation - what the model naturally produces
        # without any internal modifications or patching
        with torch.no_grad():
            output_ids = mt.model.generate(
                input_ids,
                images=image_tensor,  # Add batch dimension
                image_sizes=[image.size],
                max_new_tokens=max_new_tokens,
                pad_token_id=mt.tokenizer.pad_token_id,
                temperature=temperature if temperature is not None else 1.0,
                do_sample=(temperature is not None and temperature > 0)
            )
        
        # Decode only the new tokens (excluding the input prompt)
        # This gives us just the model's response
        output_text = mt.tokenizer.decode(output_ids[0][1:], skip_special_tokens=True)
        
        if verbose:
            print(f"Generated baseline output: {output_text}")
        
        return output_text.strip()
        
    except Exception as e:
        print(f"Error in baseline generation: {e}")
        return ""


def load_cate_mapping(mapping_path):
    """
    Load category mapping from JSON file.
    This function is identical to the original experiment.
    
    The category mapping translates numerical category IDs 
    from the dataset into human-readable category names.
    """
    with open(mapping_path, "r") as f:
        category_map = json.load(f)
    return category_map


def text_prompt_to_qs(model, model_path, prompt):
    """
    Convert text prompt to proper LLaVA conversation format.
    This function is identical to the original experiment.
    
    LLaVA uses specific conversation templates that format the input
    with proper image tokens and conversation structure.
    
    Args:
        model: The LLaVA model (used to get config)
        model_path: Path to the model (used to determine conversation mode)
        prompt: Raw text prompt
        
    Returns:
        Formatted prompt string ready for tokenization
    """
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


if __name__ == "__main__":
    # Set device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Output directory for baseline results
    result_dir = "/home/sc305/VLM/Final Experiments/global-img-ld/lg_results/llava-baseline-results.json"
    model_path = "liuhaotian/llava-v1.5-7b"

    # Load dataset and category mapping (same as original)
    sample_trainset = load_from_disk("../data/full_sample")
    category_mapping = load_cate_mapping("../data/instances_category_map.json")

    # Load vision-language model
    mt = load_vlm(model_path, device)
    
    # Load target image (kept for consistency, though not used in baseline)
    target_image = Image.open("../DeepSeek-VL2/images/apple.png")

    exp_results = []
    pbar = tqdm(sample_trainset)
    count = 0

    print("Starting LLaVA baseline experiment...")
    print("This experiment generates responses without any layer patching or patchscopes intervention.")
    print("Each sample will be processed once, representing the model's natural behavior.")
    print(f"Using model: {model_path}")
    print(f"Device: {device}")
    
    for sample in pbar:
        # Get category for this sample
        category = category_mapping[str(sample["annotations"]["category_id"][0])]
        
        # Use the same prompt format as the original experiment
        prompt = f"What is in the image? {category} or apple"
        
        # Convert to proper LLaVA conversation format
        formatted_prompt = text_prompt_to_qs(mt.model, model_path, prompt)

        try:
            # BASELINE APPROACH: Single response generation without any patching
            # Unlike the original experiment which uses inspect_vision_in_lm with layer manipulation,
            # the baseline simply asks the model what it sees in the image using standard inference
            result = generate_baseline_response(mt, formatted_prompt, sample["image"])
            
            # Determine if the model correctly identified the category
            # This uses the same evaluation logic as the original experiment
            prediction = 1 if category in result.lower() else 0
            
            # Store results in a format that's comparable to the original experiment
            # Note: For baseline, we don't have layer-wise results, so we store a single prediction
            exp_results.append({
                "sample_id": count,
                "category": category,
                "prompt": prompt,
                "formatted_prompt": formatted_prompt,
                "model_response": result,
                "prediction": prediction,
                "baseline": True  # Flag to indicate this is baseline data
            })
            
        except Exception as e:
            print(f"Error processing sample {count}: {e}")
            continue
        
        # Save results after each sample (same approach as original)
        with open(result_dir, "w") as f:
            json.dump(exp_results, f, indent=2)

        count += 1
        correct_predictions = sum(r['prediction'] for r in exp_results)
        pbar.set_description(f"Processed: {count}, Correct: {correct_predictions}")

    # Final results summary
    total_samples = len(exp_results)
    correct_predictions = sum(r['prediction'] for r in exp_results)
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0

    print(f"\n=== LLAVA BASELINE EXPERIMENT RESULTS ===")
    print(f"Total samples processed: {total_samples}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Baseline accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")

    # Save final results
    with open(result_dir, "w") as f:
        json.dump(exp_results, f, indent=2)
    
    print(f"Results saved to: {result_dir}")
