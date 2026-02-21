"""
VPatchscope for Incorrect Samples Only (NegBench Dataset)
=========================================================

This script performs activation patching on samples where the baseline model
answered incorrectly (hallucinated). The goal is to see if patching can fix
the model's hallucinations.

Dataset: NegBench - Questions about objects NOT present in images
Expected Answer: "No" (the object is NOT in the image)
Model Error: Said "Yes" when object was not present (hallucination)
"""

import os
import sys
import json
from pathlib import Path

# Get the absolute path of the current script
current_script_path = Path(__file__).parent.absolute()
# Add the Qwen2-VL directory to Python path
sys.path.append("/home/sc305/VLM/Qwen2-VL")
# Change to Qwen2-VL directory for relative imports
os.chdir("/home/sc305/VLM/Qwen2-VL")

from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from general_utils import ModelAndTokenizer
from patchscopes_utils import inspect_vision_in_lm
from tqdm import tqdm


def load_vlm(model_path: str, device: str) -> ModelAndTokenizer:
    """
    Load the Qwen2.5-VL model and processor.
    
    Args:
        model_path: HuggingFace model ID or local path
        device: Device to run on (e.g., 'cuda:0')
    
    Returns:
        ModelAndTokenizer object with model and processor attached
    """
    print(f"Loading model: {model_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
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
    print("✅ Model loaded successfully")
    return mt


def load_incorrect_samples(baseline_results_path: str) -> list:
    """
    Load samples where the model answered incorrectly from baseline results.
    
    In NegBench, the expected answer is "No" (object not present).
    Incorrect samples are where the model said "Yes" (hallucinated).
    
    Args:
        baseline_results_path: Path to the JSONL file with baseline results
    
    Returns:
        List of dictionaries containing incorrect sample information
    """
    incorrect_samples = []
    
    print(f"Loading baseline results from: {baseline_results_path}")
    
    with open(baseline_results_path, "r") as f:
        for line_num, line in enumerate(f):
            try:
                sample = json.loads(line.strip())
                # Filter for incorrect samples (model hallucinated)
                # correct=0 means model's prediction was wrong
                # expected_label=0 means the correct answer is "No"
                # prediction=1 means model said "Yes" (hallucination)
                if sample.get("correct") == 0:
                    incorrect_samples.append(sample)
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line {line_num}: {e}")
                continue
    
    print(f"✅ Found {len(incorrect_samples)} incorrect samples (hallucinations)")
    return incorrect_samples


def evaluate_response(response: str, expected_answer: str = "no") -> dict:
    """
    Evaluate if the model's response matches the expected answer.
    
    For NegBench, the expected answer is always "No" since we're asking
    about objects that are NOT present in the image.
    
    Args:
        response: Model's response text
        expected_answer: Expected answer (default "no" for NegBench)
    
    Returns:
        Dictionary with prediction details
    """
    response_lower = response.lower().strip()
    
    # Check for explicit yes/no answers
    if "yes" in response_lower and "no" not in response_lower:
        prediction = 1  # Model said "Yes" (still hallucinating)
        is_correct = (expected_answer.lower() == "yes")
    elif "no" in response_lower and "yes" not in response_lower:
        prediction = 0  # Model said "No" (correct for NegBench)
        is_correct = (expected_answer.lower() == "no")
    else:
        # Ambiguous response - default to incorrect
        prediction = -1
        is_correct = False
    
    return {
        "prediction": prediction,
        "is_correct": is_correct,
        "response": response
    }


def run_patching_experiment(
    mt: ModelAndTokenizer,
    sample: dict,
    target_image: Image.Image,
    layer_range: range,
    max_gen_len: int = 20
) -> dict:
    """
    Run activation patching experiment on a single sample.
    
    The experiment patches activations from the source image (the actual image)
    into the target image (a blank image) to see if the model can correctly
    answer about the source image's content.
    
    Args:
        mt: ModelAndTokenizer object
        sample: Dictionary with sample info (image_path, category, prompt)
        target_image: Blank target image for patching
        layer_range: Range of layers to sweep
        max_gen_len: Maximum generation length
    
    Returns:
        Dictionary with patching results for all layer combinations
    """
    # Load the source image (the actual image from the dataset)
    source_image = Image.open(sample["image_path"]).resize((384, 384))
    
    # Get the prompt (asking about the absent object)
    prompt = sample.get("prompt", f"Is there a {sample['category']} in the image? Answer yes or no.")
    
    # Run layer sweep
    layer_results = []
    
    for layer_source in layer_range:
        source_layer_results = []
        for layer_target in layer_range:
            try:
                # Perform activation patching
                result = inspect_vision_in_lm(
                    mt,
                    prompt_target=prompt,
                    source_image=source_image,
                    target_image=target_image,
                    layer_source=layer_source,
                    layer_target=layer_target,
                    max_gen_len=max_gen_len
                )
                
                # Evaluate the response
                # For NegBench, the correct answer is "No"
                eval_result = evaluate_response(result, expected_answer="no")
                prediction = eval_result["prediction"]
                
                # Store 1 if correct (said "No"), 0 if still hallucinating (said "Yes")
                source_layer_results.append(prediction)
                
            except Exception as e:
                print(f"Error at layer_source={layer_source}, layer_target={layer_target}: {e}")
                source_layer_results.append(-1)  # Error marker
        
        layer_results.append(source_layer_results)
    
    return {
        "sample_id": sample["sample_id"],
        "filename": sample["filename"],
        "image_path": sample["image_path"],
        "category": sample["category"],
        "prompt": prompt,
        "original_prediction": sample["prediction"],  # Original hallucination (1 = Yes)
        "expected_label": sample["expected_label"],   # Always 0 (No) for NegBench
        "results": layer_results  # Results matrix for layer sweep
    }


def main():
    """Main function to run the patching experiment on incorrect samples."""
    
    # ===========================================
    # Configuration
    # ===========================================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    
    # Input: Baseline results with correct/incorrect labels
    baseline_results_path = "/home/sc305/VLM/Qwen2-VL/negbench_eval_results_qwen2_5/baseline_results.jsonl"
    
    # Output: Patching experiment results
    output_dir = str(current_script_path)
    results_path = os.path.join(output_dir, "patching_results_incorrect_only.json")
    
    # Blank target image for patching
    # The blank image is located in the surf-exp directory
    blank_image_path = "/home/sc305/VLM/Final Experiments/negation/v-patchscope-demo-code/surf-exp/test_images_split/test_images/blank_384x384.png"
    
    # Layer sweep configuration
    # Qwen2.5-VL-7B has 28 layers in the language model
    layer_step = 3  # Step between layers to sweep
    layer_range = range(0, 28, layer_step)
    max_gen_len = 20
    
    # ===========================================
    # Load Data
    # ===========================================
    print("="*80)
    print("ACTIVATION PATCHING EXPERIMENT FOR INCORRECT SAMPLES")
    print("="*80)
    print(f"\nDataset: NegBench (asking about objects NOT in images)")
    print(f"Expected Answer: No")
    print(f"Model Error: Said Yes (hallucination)")
    print(f"\nGoal: Check if activation patching can fix hallucinations")
    print("="*80)
    
    # Load incorrect samples from baseline results
    incorrect_samples = load_incorrect_samples(baseline_results_path)
    
    if not incorrect_samples:
        print("❌ No incorrect samples found!")
        return
    
    # ===========================================
    # Load Model
    # ===========================================
    mt = load_vlm(model_path, device)
    
    # Load blank target image
    if os.path.exists(blank_image_path):
        target_image = Image.open(blank_image_path).resize((384, 384))
        print(f"✅ Loaded blank target image from: {blank_image_path}")
    else:
        # Create a blank white image if the file doesn't exist
        print(f"⚠️ Blank image not found at {blank_image_path}, creating one...")
        target_image = Image.new("RGB", (384, 384), color=(255, 255, 255))
    
    # ===========================================
    # Load Existing Results (Resume Support)
    # ===========================================
    exp_results = []
    processed_sample_ids = set()
    
    if os.path.exists(results_path):
        try:
            with open(results_path, "r") as f:
                exp_results = json.load(f)
            processed_sample_ids = {r["sample_id"] for r in exp_results}
            print(f"✅ Loaded {len(exp_results)} existing results from {results_path}")
        except Exception as e:
            print(f"⚠️ Could not load existing results: {e}")
            exp_results = []
    
    # Filter out already processed samples
    remaining_samples = [s for s in incorrect_samples if s["sample_id"] not in processed_sample_ids]
    print(f"\nTotal incorrect samples: {len(incorrect_samples)}")
    print(f"Already processed: {len(processed_sample_ids)}")
    print(f"Remaining to process: {len(remaining_samples)}")
    
    # ===========================================
    # Run Experiment
    # ===========================================
    print("\n" + "="*80)
    print("Running patching experiment...")
    print(f"Layer range: {list(layer_range)}")
    print("="*80 + "\n")
    
    for sample in tqdm(remaining_samples, desc="Processing incorrect samples"):
        try:
            # Verify image exists
            if not os.path.exists(sample["image_path"]):
                print(f"\n⚠️ Image not found: {sample['image_path']}")
                continue
            
            # Run patching experiment
            result = run_patching_experiment(
                mt=mt,
                sample=sample,
                target_image=target_image,
                layer_range=layer_range,
                max_gen_len=max_gen_len
            )
            
            # Debug output for first sample
            if len(exp_results) == 0:
                print(f"\n{'='*60}")
                print(f"DEBUG - First sample")
                print(f"Category: {result['category']}")
                print(f"Prompt: {result['prompt']}")
                print(f"Original prediction: {result['original_prediction']} (1=Yes, 0=No)")
                print(f"Expected label: {result['expected_label']} (should be 0=No)")
                print(f"Patching results matrix shape: {len(result['results'])}x{len(result['results'][0])}")
                print(f"{'='*60}\n")
            
            # Save result
            exp_results.append(result)
            
            # Save after each sample (for resuming)
            with open(results_path, "w") as f:
                json.dump(exp_results, f, indent=2)
                
        except Exception as e:
            print(f"\n⚠️ Error processing sample {sample['sample_id']}: {e}")
            continue
    
    # ===========================================
    # Final Summary
    # ===========================================
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"Total samples processed: {len(exp_results)}")
    print(f"Results saved to: {results_path}")
    
    # Calculate summary statistics
    if exp_results:
        print("\n" + "-"*60)
        print("RESULTS INTERPRETATION:")
        print("-"*60)
        print("In the results matrix:")
        print("  - 0 = Model said 'No' = CORRECT (hallucination fixed)")
        print("  - 1 = Model said 'Yes' = INCORRECT (still hallucinating)")
        print("  - -1 = Error during inference")
        print("-"*60)
        
        # Count how many samples were "fixed" by patching at any layer combination
        fixed_at_any_layer = 0
        total_layer_combinations = 0
        total_fixed_combinations = 0
        
        layer_list = list(layer_range)
        num_layers = len(layer_list)
        
        # Accumulate per-layer-combination statistics
        layer_fix_counts = [[0 for _ in range(num_layers)] for _ in range(num_layers)]
        
        for result in exp_results:
            results_matrix = result["results"]
            sample_fixed = False
            
            for i, row in enumerate(results_matrix):
                for j, pred in enumerate(row):
                    if pred != -1:
                        total_layer_combinations += 1
                        if pred == 0:  # Fixed (said "No" correctly)
                            total_fixed_combinations += 1
                            layer_fix_counts[i][j] += 1
                            sample_fixed = True
            
            if sample_fixed:
                fixed_at_any_layer += 1
        
        print(f"\nSamples fixed by patching (at any layer): {fixed_at_any_layer}/{len(exp_results)} ({100*fixed_at_any_layer/len(exp_results):.1f}%)")
        print(f"Total layer combinations evaluated: {total_layer_combinations}")
        print(f"Total combinations that fixed hallucination: {total_fixed_combinations} ({100*total_fixed_combinations/total_layer_combinations:.1f}%)")
        
        # Find best layer combination
        best_fix_rate = 0
        best_layer_pair = (0, 0)
        for i in range(num_layers):
            for j in range(num_layers):
                fix_rate = layer_fix_counts[i][j] / len(exp_results) if exp_results else 0
                if fix_rate > best_fix_rate:
                    best_fix_rate = fix_rate
                    best_layer_pair = (layer_list[i], layer_list[j])
        
        print(f"\nBest layer combination: source={best_layer_pair[0]}, target={best_layer_pair[1]}")
        print(f"Fix rate at best layer: {100*best_fix_rate:.1f}%")
    
    print("="*80)


if __name__ == "__main__":
    main()

