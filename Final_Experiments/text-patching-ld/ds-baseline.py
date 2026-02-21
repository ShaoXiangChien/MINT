"""
Baseline Experiment for DeepSeek-VL2 Model

Purpose: Test if the model hallucinates or has language bias toward answering "yes"
         when shown a blank image (None input).

Setup:
- Image input: None (blank/no image)
- Text input: "Is there a {category} in the image? only answer with yes or no"
- Expected correct answer: "no" (since there's no image/blank image)

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
# Add the DeepSeek-VL2 directory to Python path
sys.path.append("/home/sc305/VLM/DeepSeek-VL2")
# Change to DeepSeek-VL2 directory
os.chdir("/home/sc305/VLM/DeepSeek-VL2")

import torch
from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor
from general_utils import ModelAndTokenizer, prepare_inputs
import json
from datasets import load_from_disk
from tqdm import tqdm
from PIL import Image


def load_vlm(device):
    """Load the DeepSeek-VL2 vision-language model."""
    model_path = "deepseek-ai/deepseek-vl2-tiny"
    vl_chat_processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    mt = ModelAndTokenizer(
        "deepseek-ai/deepseek-vl2-7b-tiny",
        model=vl_gpt,
        tokenizer=tokenizer,
        processor=vl_chat_processor,
        low_cpu_mem_usage=False,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    mt.model = mt.model.to(mt.device)
    return mt


def load_cate_mapping(mapping_path):
    """Load the category mapping from JSON file."""
    with open(mapping_path, "r") as f:
        category_map = json.load(f)
    return category_map


def generate_response(mt, inp):
    """
    Generate a response from the model without any patching.
    This is the baseline behavior - just ask the model directly.
    """
    with torch.no_grad():
        output_ids = mt.model.generate(
            inputs_embeds=mt.model.prepare_inputs_embeds(**inp),
            attention_mask=inp["attention_mask"],
            pad_token_id=mt.tokenizer.pad_token_id,
            eos_token_id=mt.tokenizer.eos_token_id,
            bos_token_id=mt.tokenizer.bos_token_id,
            max_new_tokens=20,
            do_sample=False
        )

    # Decode the output tokens to text
    output = mt.processor.tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    
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
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Results path
    results_path = "/home/sc305/VLM/Final Experiments/text-patching-ld/results/ds_baseline_results.json"
    
    # Data paths
    sample_trainset = load_from_disk("../data/full_sample")
    category_mapping = load_cate_mapping("../data/instances_category_map.json")
    
    # ==================== Load Model ====================
    print("Loading DeepSeek-VL2 model...")
    mt = load_vlm(device)
    print("Model loaded successfully!")
    
    # Use None as blank image for DeepSeek (as done in ds-exp.py)
    blank_image = None
    
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
            
            # Prepare inputs with blank image (None)
            inp = prepare_inputs(prompt, blank_image, mt.processor, mt.device)
            
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
                        "model": "deepseek-ai/deepseek-vl2-tiny",
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
            "model": "deepseek-ai/deepseek-vl2-tiny",
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

