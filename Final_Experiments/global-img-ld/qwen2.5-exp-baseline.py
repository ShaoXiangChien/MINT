import os
import sys
from pathlib import Path

# Get the absolute path of the current script
current_script_path = Path(__file__).parent.absolute()
# Add the Qwen2-VL directory to Python path
sys.path.append("/home/sc305/VLM/Qwen2-VL")
# Change to Qwen2-VL directory
os.chdir("/home/sc305/VLM/Qwen2-VL")

from PIL import Image
import numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
from general_utils import ModelAndTokenizer, prepare_inputs
from qwen_vl_utils import process_vision_info
import math
import json
from datasets import load_from_disk
from tqdm import tqdm
import re


def load_vlm(model_path, device):
    """
    Load the Qwen2-VL vision-language model.
    This function is identical to the original experiment for consistency.
    """
    # Load model with automatic precision and device mapping
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)

    mt = ModelAndTokenizer(
        model_path,
        model=model,
        low_cpu_mem_usage=False,
        device=device
    )

    mt.processor = processor
    mt.model.eval()
    return mt


def generate_qwen_baseline_response(mt, prompt, image, max_gen_len=20, temperature=None, verbose=False):
    """
    Generate a baseline response from the Qwen2-VL model WITHOUT any patching.
    
    This is the key difference from the original experiment:
    - No layer-wise patching or inspection
    - Direct generation from the model using the given image and prompt
    - Uses Qwen2-VL's native inference pipeline
    
    Args:
        mt: ModelAndTokenizer object containing the Qwen2-VL model, tokenizer, and processor
        prompt: Text prompt to send to the model
        image: PIL Image to process
        max_gen_len: Maximum number of tokens to generate
        temperature: Sampling temperature (None for greedy decoding)
        verbose: Whether to print debug information
        
    Returns:
        Generated text response from the model
    """
    # Prepare inputs using the same method as the original experiment
    # This ensures consistency in input preprocessing with Qwen2-VL format
    inputs = prepare_inputs(prompt, image.resize((384, 384)), mt.processor, mt.device)
    
    # Generate response without any hooks or patching
    # This is pure Qwen2-VL model inference - what the model naturally produces
    with torch.no_grad():
        # Use Qwen2-VL's generate method directly
        output_ids = mt.model.generate(
            **inputs,
            max_new_tokens=max_gen_len,
        )
    
    
    output_text = mt.processor.batch_decode(
            [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], output_ids)],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
    
    if verbose:
        print(f"Generated Qwen baseline output: {output_text}")
    
    return output_text


def load_cate_mapping(mapping_path):
    """
    Load category mapping from JSON file.
    This function is identical to the original experiment.
    """
    with open(mapping_path, "r") as f:
        category_map = json.load(f)
    return category_map


if __name__ == "__main__":
    # Set device
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    model_path = "Qwen/Qwen2.5-VL-7B-Instruct"

    # Output directory for baseline results
    results_path = "/home/sc305/VLM/Final Experiments/global-img-ld/lg_results/qwen2.5_baseline_results.json"

    # Load dataset and category mapping (same as original)
    sample_trainset = load_from_disk("../data/full_sample")
    category_mapping = load_cate_mapping("../data/instances_category_map.json")

    # Load vision-language model
    mt = load_vlm(model_path, device)
    
    # Load target image (this seems to be unused in baseline, but kept for consistency)
    target_image = Image.open("../DeepSeek-VL2/images/apple.png")

    exp_results = []
    pbar = tqdm(sample_trainset)
    count = 0

    print("Starting Qwen2.5-VL baseline experiment...")
    print("This experiment generates responses without any layer patching or inspection.")
    print("Each sample will be processed once, representing the Qwen2-VL model's natural behavior.")
    
    for sample in pbar:
        # Get category for this sample
        category = category_mapping[str(sample["annotations"]["category_id"][0])]
        prompt = f"What is in the image? {category} or apple"

        try:
            # BASELINE APPROACH: Single response generation without patching
            # Unlike the original experiment which tests different layer combinations (0-28, step 3),
            # the baseline simply asks the Qwen2-VL model what it sees in the image
            result = generate_qwen_baseline_response(mt, prompt, sample["image"])
            
            # Determine if the model correctly identified the category
            # This uses the same evaluation logic as the original experiment
            prediction = 1 if category in result.lower() else 0
            
            # Store results in a format that's comparable to the original experiment
            # Note: For baseline, we don't have layer-wise results, so we store a single prediction
            exp_results.append({
                "sample_id": count,
                "category": category,
                "prompt": prompt,
                "model_response": result,
                "prediction": prediction,
                "baseline": True,  # Flag to indicate this is baseline data
                "model": "Qwen2-VL-7B-Instruct"  # Model identifier
            })
            
        except Exception as e:
            print(f"Error processing sample {count}: {e}")
            continue
        
        # Save results after each sample (same approach as original)
        with open(results_path, "w") as f:
            json.dump(exp_results, f, indent=2)

        count += 1
        pbar.set_description(f"Processed: {count}, Correct predictions: {sum(r['prediction'] for r in exp_results)}")

    # Final results summary
    total_samples = len(exp_results)
    correct_predictions = sum(r['prediction'] for r in exp_results)
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0

    print(f"\n=== QWEN2.5-VL BASELINE EXPERIMENT RESULTS ===")
    print(f"Model: {model_path}")
    print(f"Total samples processed: {total_samples}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Baseline accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")

    # Save final results
    with open(results_path, "w") as f:
        json.dump(exp_results, f, indent=2)
    
    print(f"Results saved to: {results_path}")
    
    # Additional comparison notes for the instructor
    print(f"\n=== COMPARISON WITH PATCHED EXPERIMENT ===")
    print(f"Original experiment: Tests {len(range(0, 28, 3))}x{len(range(0, 28, 3))} = {len(range(0, 28, 3))**2} layer combinations")
    print(f"Baseline experiment: Single inference per sample (natural model behavior)")
    print(f"Use this baseline to understand if patching improves performance over natural responses")
