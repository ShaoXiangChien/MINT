"""
Baseline Experiment for LLaVA Model

Purpose: Test if the model hallucinates or has language bias toward answering "yes"
         when shown a blank image.

Setup:
- Image input: blank image (2x2 black pixels)
- Text input: "Is there a {category} in the image? only answer with yes or no"
- Expected correct answer: "no" (since blank image has no objects)

Metrics:
- prediction = 1 if "yes" in response (hallucination)
- prediction = 0 if "no" in response (correct)
- accuracy = percentage of "no" responses (correct behavior)
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
    """Load category ID to name mapping from JSON file."""
    with open(mapping_path, "r") as f:
        category_map = json.load(f)
    return category_map


def text_prompt_to_qs(prompt, model, model_path):
    """
    Convert a text prompt to the format expected by LLaVA.
    
    This function adds the image token and wraps the prompt in the
    appropriate conversation template based on the model type.
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


def generate_response(mt, inp):
    """
    Generate a response from the model without any patching.
    This is the baseline behavior - just ask the model directly.
    """
    with torch.no_grad():
        output_ids = mt.model.generate(
            inp["input_ids"],
            images=inp["images"],
            image_sizes=inp["image_sizes"],
            max_new_tokens=20,
            pad_token_id=mt.tokenizer.pad_token_id,
        )
    
    # Decode output (skip the first token which is typically BOS)
    output = mt.tokenizer.decode(output_ids[0][1:], skip_special_tokens=True)
    
    return output


def parse_prediction(response_text):
    """
    Parse the model's response to determine prediction.
    
    Returns:
        prediction (int): 1 if "yes" in response (hallucination), 0 otherwise
        is_correct (bool): True if model correctly answered "no"
    """
    response_lower = response_text.lower().strip()
    
    # Check if "yes" is in the response
    has_yes = "yes" in response_lower
    has_no = "no" in response_lower
    
    # prediction = 1 means the model said "yes" (hallucination)
    # prediction = 0 means the model did not say "yes"
    if has_yes and not has_no:
        prediction = 1  # Clear yes - hallucination
    elif has_no and not has_yes:
        prediction = 0  # Clear no - correct
    elif has_yes and has_no:
        # Both present, check which comes first
        prediction = 1 if response_lower.find("yes") < response_lower.find("no") else 0
    else:
        # Neither yes nor no - unclear response
        prediction = -1  # Unclear
    
    # For a blank image, correct answer is "no", so prediction=0 is correct
    is_correct = (prediction == 0)
    
    return prediction, is_correct


if __name__ == "__main__":
    # ==================== Configuration ====================
    device = "cuda:5" if torch.cuda.is_available() else "cpu"
    model_path = "liuhaotian/llava-v1.5-7b"
    
    # Results path
    results_path = "/home/sc305/VLM/Final Experiments/text-patching-ld/results/llava_baseline_results.json"
    
    # Ensure results directory exists
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    # Data paths
    data_path = "../data/full_sample"
    mapping_path = "../data/instances_category_map.json"
    
    sample_trainset = load_from_disk(data_path)
    category_mapping = load_cate_mapping(mapping_path)
    
    # ==================== Load Model ====================
    print("Loading LLaVA model...")
    mt = load_vlm(model_path, device)
    print("Model loaded successfully!")
    
    # Create a blank image (2x2 black pixels, same as llava-exp.py)
    blank_image = Image.new("RGB", (2, 2), (0, 0, 0))
    
    # ==================== Run Baseline Experiment ====================
    exp_results = []
    correct_count = 0
    yes_count = 0
    no_count = 0
    unclear_count = 0
    
    pbar = tqdm(sample_trainset, desc="Running baseline")
    count = 0
    
    for sample in pbar:
        try:
            # Get the category for this sample
            category = category_mapping[str(sample["annotations"]["category_id"][0])]
            
            # Create the prompt
            prompt = f"Is there a {category} in the image? only answer with yes or no"
            
            # Prepare inputs with blank image
            inp = prepare_inputs(prompt, blank_image, mt, model_path)
            
            # Generate response from model
            response_text = generate_response(mt, inp)
            
            # Parse the prediction
            prediction, is_correct = parse_prediction(response_text)
            
            # Update counters
            if prediction == 1:
                yes_count += 1
            elif prediction == 0:
                no_count += 1
                correct_count += 1
            else:
                unclear_count += 1
            
            # Store the result with both response text and prediction
            exp_results.append({
                "sample_id": count,
                "category": category,
                "prompt": prompt,
                "response": response_text,
                "prediction": prediction,  # 1=yes(hallucination), 0=no(correct), -1=unclear
                "is_correct": is_correct
            })
            
            # Update progress bar with running accuracy
            current_accuracy = correct_count / (count + 1) * 100 if count >= 0 else 0
            pbar.set_postfix({
                "acc": f"{current_accuracy:.1f}%",
                "yes": yes_count,
                "no": no_count
            })
            
        except Exception as e:
            print(f"\nError processing sample {count}: {e}")
            exp_results.append({
                "sample_id": count,
                "category": category if 'category' in locals() else "unknown",
                "prompt": prompt if 'prompt' in locals() else "unknown",
                "response": f"ERROR: {str(e)}",
                "prediction": -2,  # Error indicator
                "is_correct": False
            })
        
        # Save results periodically
        if count % 100 == 0:
            with open(results_path, "w") as f:
                json.dump({
                    "metadata": {
                        "model": model_path,
                        "experiment": "baseline",
                        "description": "Testing hallucination/language bias with blank image",
                        "total_samples": count + 1,
                        "correct_count": correct_count,
                        "yes_count": yes_count,
                        "no_count": no_count,
                        "unclear_count": unclear_count
                    },
                    "results": exp_results
                }, f, indent=2)
        
        count += 1
    
    # ==================== Calculate Final Statistics ====================
    total_valid = yes_count + no_count  # Exclude unclear responses
    accuracy = (no_count / total_valid * 100) if total_valid > 0 else 0
    hallucination_rate = (yes_count / total_valid * 100) if total_valid > 0 else 0
    
    print("\n" + "="*60)
    print("BASELINE EXPERIMENT RESULTS")
    print("="*60)
    print(f"Total samples processed: {count}")
    print(f"Valid responses (yes/no): {total_valid}")
    print(f"Unclear responses: {unclear_count}")
    print("-"*60)
    print(f"'Yes' responses (hallucination): {yes_count} ({hallucination_rate:.2f}%)")
    print(f"'No' responses (correct): {no_count} ({accuracy:.2f}%)")
    print("-"*60)
    print(f"ACCURACY (correct 'no' responses): {accuracy:.2f}%")
    print(f"HALLUCINATION RATE: {hallucination_rate:.2f}%")
    print("="*60)
    
    # ==================== Save Final Results ====================
    final_results = {
        "metadata": {
            "model": model_path,
            "experiment": "baseline",
            "description": "Testing hallucination/language bias with blank image",
            "total_samples": count,
            "valid_samples": total_valid,
            "correct_count": no_count,
            "yes_count": yes_count,
            "no_count": no_count,
            "unclear_count": unclear_count,
            "accuracy": accuracy,
            "hallucination_rate": hallucination_rate
        },
        "results": exp_results
    }
    
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")

