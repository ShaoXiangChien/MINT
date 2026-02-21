"""
Verify Negation Understanding
==============================

Tests if the fine-tuned models truly understand negation, or if they just
learned to always say "No". 

We test on a balanced dataset with both Yes and No answers.
"""

import json
import os
from pathlib import Path
import sys

sys.path.append("/home/sc305/VLM/Qwen2-VL")
os.chdir("/home/sc305/VLM/Qwen2-VL")

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from patchscopes_utils import prepare_inputs
from PIL import Image
import torch
from tqdm import tqdm


def load_model_with_adapter(base_model_path, adapter_path, device):
    """Load base model with LoRA adapter."""
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype="auto",
        device_map=device
    )
    
    if adapter_path and os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(model, adapter_path)
    
    processor = AutoProcessor.from_pretrained(base_model_path)
    model.eval()
    return model, processor


def evaluate_response(response_text: str) -> int:
    """Convert response to binary prediction."""
    response_lower = response_text.lower().strip()
    if "yes" in response_lower and "no" not in response_lower:
        return 1
    if "no" in response_lower and "yes" not in response_lower:
        return 0
    return 1  # Default to yes if unclear


def run_inference(model, processor, image_path, prompt, device):
    """Run inference on a single example."""
    image = Image.open(image_path).resize((384, 384))
    inputs = prepare_inputs(prompt, image, processor, device)
    
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=20)
    
    decoded = processor.batch_decode(
        [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], output_ids)],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    
    return decoded


def create_balanced_test_set(baseline_results_path, negbench_csv_path, output_path, num_samples_per_label=20):
    """
    Create a balanced test set with both Yes and No answers.
    
    For "No" samples: Use misclassified from baseline (should say No but said Yes)
    For "Yes" samples: Find images from COCO that actually contain the object
    """
    print(f"Loading baseline results from {baseline_results_path}...")
    
    # Load all baseline results
    all_results = []
    with open(baseline_results_path, 'r') as f:
        for line in f:
            if line.strip():
                all_results.append(json.loads(line))
    
    print(f"   Loaded {len(all_results)} total results")
    
    # For "No" samples: Use misclassified (baseline said Yes but should say No)
    misclassified = [r for r in all_results if r['correct'] == 0]
    print(f"\n   Misclassified (should say No): {len(misclassified)}")
    
    # For "Yes" samples: We need to find images that actually contain objects
    # Load NegBench CSV to find positive_objects
    print(f"\n   Loading NegBench CSV to find positive objects...")
    import pandas as pd
    import ast
    import random
    
    df = pd.read_csv(negbench_csv_path)
    
    # Find samples where baseline was correct (said No correctly)
    # These are images that don't contain the negative object
    # But we can use positive_objects to find images that DO contain objects
    yes_samples = []
    
    # Strategy: Collect all categories from misclassified, then find Yes samples
    categories_to_find = set()
    for mis_item in misclassified[:num_samples_per_label]:
        categories_to_find.add(mis_item.get('category', 'unknown'))
    
    print(f"   Looking for images with positive objects: {categories_to_find}")
    
    # Find Yes samples: images where category is in positive_objects
    used_image_ids = set()
    random.seed(42)
    
    # Shuffle dataframe for random sampling
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    for category in categories_to_find:
        found_count = 0
        for idx, row in df_shuffled.iterrows():
            if found_count >= num_samples_per_label // max(1, len(categories_to_find)):
                break
            
            image_id = row.get('image_id')
            if image_id in used_image_ids:
                continue
            
            # Parse positive_objects
            try:
                pos_objs = row.get('positive_objects', '[]')
                if isinstance(pos_objs, str):
                    pos_objs = ast.literal_eval(pos_objs)
                if not isinstance(pos_objs, list):
                    continue
            except:
                continue
            
            # Check if category is in positive_objects
            if category in pos_objs:
                # Use new_filepath if available, otherwise build from image_id
                img_path = row.get('new_filepath')
                if not img_path or not os.path.exists(img_path):
                    if image_id:
                        img_path = f"/home/sc305/VLM/Qwen2-VL/data/val2017/{image_id:012d}.jpg"
                
                if img_path and os.path.exists(img_path):
                    yes_samples.append({
                        'image_path': img_path,
                        'prompt': f"Is there a {category} in the image? Answer yes or no.",
                        'expected_label': 1,
                        'category': category
                    })
                    used_image_ids.add(image_id)
                    found_count += 1
        
        # If we still need more samples, continue with other categories
        if len(yes_samples) >= num_samples_per_label:
            break
    
    # Sample "No" samples from misclassified
    random.seed(42)
    
    sampled_no = random.sample(misclassified, min(num_samples_per_label, len(misclassified)))
    
    # Convert to test format
    no_samples = []
    for item in sampled_no:
        no_samples.append({
            'image_path': item['image_path'],
            'prompt': item['prompt'],
            'expected_label': 0,
            'category': item.get('category', 'unknown')
        })
    
    balanced_set = no_samples + yes_samples
    random.shuffle(balanced_set)
    
    print(f"\n   Created balanced test set:")
    print(f"   - {len(no_samples)} samples expecting 'No' (from misclassified)")
    print(f"   - {len(yes_samples)} samples expecting 'Yes' (from positive objects)")
    print(f"   - Total: {len(balanced_set)} samples")
    
    # Save as SFT format
    sft_data = []
    for item in balanced_set:
        # Extract prompt from baseline result
        prompt = item.get('prompt', f"Is there a {item.get('category', 'object')} in the image? Answer yes or no.")
        expected_answer = "Yes." if item['expected_label'] == 1 else "No."
        
        sft_example = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": item['image_path']},
                        {"type": "text", "text": prompt}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": expected_answer}]
                }
            ]
        }
        sft_data.append({
            'example': sft_example,
            'expected_label': item['expected_label'],
            'category': item.get('category', 'unknown')
        })
    
    # Save
    with open(output_path, 'w') as f:
        for item in sft_data:
            f.write(json.dumps(item['example']) + '\n')
    
    # Save metadata
    metadata_path = output_path.replace('.jsonl', '_metadata.json')
    metadata = {
        'total_samples': len(balanced_set),
        'num_no': len(no_samples),
        'num_yes': len(yes_samples),
        'samples': [
            {
                'image_path': item['example']['messages'][0]['content'][0]['image'],
                'expected_label': item['expected_label'],
                'category': item['category']
            }
            for item in sft_data
        ]
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n   ✅ Saved balanced test set to {output_path}")
    print(f"   ✅ Saved metadata to {metadata_path}")
    
    return balanced_set


def evaluate_on_balanced_set(model, processor, balanced_set, device, model_name):
    """Evaluate model on balanced test set."""
    print(f"\n{'='*80}")
    print(f"📊 Evaluating: {model_name}")
    print(f"{'='*80}")
    
    results = {
        'correct_no': 0,
        'total_no': 0,
        'correct_yes': 0,
        'total_yes': 0,
        'all_correct': 0,
        'all_total': 0
    }
    
    for item in tqdm(balanced_set, desc=f"Evaluating {model_name}"):
        img_path = item['image_path']
        prompt = item['prompt']
        expected = item['expected_label']
        
        try:
            response = run_inference(model, processor, img_path, prompt, device)
            prediction = evaluate_response(response)
            
            is_correct = (prediction == expected)
            
            if expected == 0:
                results['total_no'] += 1
                if is_correct:
                    results['correct_no'] += 1
            else:
                results['total_yes'] += 1
                if is_correct:
                    results['correct_yes'] += 1
            
            if is_correct:
                results['all_correct'] += 1
            results['all_total'] += 1
            
        except Exception as e:
            print(f"\n   ⚠️  Error: {e}")
            continue
    
    # Calculate accuracies
    acc_no = (results['correct_no'] / results['total_no'] * 100) if results['total_no'] > 0 else 0
    acc_yes = (results['correct_yes'] / results['total_yes'] * 100) if results['total_yes'] > 0 else 0
    acc_all = (results['all_correct'] / results['all_total'] * 100) if results['all_total'] > 0 else 0
    
    print(f"\n✅ Evaluation complete!")
    print(f"   Overall accuracy: {acc_all:.2f}% ({results['all_correct']}/{results['all_total']})")
    print(f"   'No' accuracy: {acc_no:.2f}% ({results['correct_no']}/{results['total_no']})")
    print(f"   'Yes' accuracy: {acc_yes:.2f}% ({results['correct_yes']}/{results['total_yes']})")
    
    return {
        'model_name': model_name,
        'overall_accuracy': acc_all,
        'no_accuracy': acc_no,
        'yes_accuracy': acc_yes,
        'results': results
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify negation understanding")
    parser.add_argument(
        "--baseline_results",
        type=str,
        default="/home/sc305/VLM/Qwen2-VL/baseline_results_5000.jsonl",
        help="Path to baseline results JSONL"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2-VL-7B-Instruct",
        help="Base model path"
    )
    parser.add_argument(
        "--deep_adapter",
        type=str,
        default="./lora_adapters/deep_layers_12-18",
        help="Deep layers adapter path"
    )
    parser.add_argument(
        "--shallow_adapter",
        type=str,
        default="./lora_adapters/shallow_layers_1-7",
        help="Shallow layers adapter path"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:2",
        help="Device"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="Number of samples per label (Yes/No)"
    )
    parser.add_argument(
        "--negbench_csv",
        type=str,
        default="/home/sc305/VLM/Qwen2-VL/data/negbench_with_coco.csv",
        help="Path to NegBench CSV with positive_objects"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("🔍 VERIFYING NEGATION UNDERSTANDING")
    print("="*80)
    print("\nThis test checks if models truly understand negation, or just")
    print("learned to always say 'No'. We test on a balanced dataset with")
    print("both Yes and No answers.\n")
    
    # Create balanced test set
    balanced_jsonl = "/home/sc305/VLM/Qwen2-VL/data/balanced_verification_test.jsonl"
    balanced_set = create_balanced_test_set(
        args.baseline_results,
        args.negbench_csv,
        balanced_jsonl,
        num_samples_per_label=args.num_samples
    )
    
    # Load balanced set metadata
    metadata_path = balanced_jsonl.replace('.jsonl', '_metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Convert to format for evaluation
    test_items = []
    for sample in metadata['samples']:
        # Find corresponding item in balanced_set
        for item in balanced_set:
            if item['image_path'] == sample['image_path']:
                test_items.append({
                    'image_path': item['image_path'],
                    'prompt': item['prompt'],
                    'expected_label': sample['expected_label']
                })
                break
    
    # Evaluate baseline
    print("\n" + "="*80)
    print("1️⃣  EVALUATING BASELINE")
    print("="*80)
    base_model, processor = load_model_with_adapter(args.base_model, None, args.device)
    baseline_metrics = evaluate_on_balanced_set(
        base_model, processor, test_items, args.device, "Baseline (No LoRA)"
    )
    del base_model
    torch.cuda.empty_cache()
    
    # Evaluate deep adapter
    print("\n" + "="*80)
    print("2️⃣  EVALUATING DEEP LAYERS ADAPTER")
    print("="*80)
    deep_model, _ = load_model_with_adapter(args.base_model, args.deep_adapter, args.device)
    deep_metrics = evaluate_on_balanced_set(
        deep_model, processor, test_items, args.device, "Deep Layers (12-18)"
    )
    del deep_model
    torch.cuda.empty_cache()
    
    # Evaluate shallow adapter
    print("\n" + "="*80)
    print("3️⃣  EVALUATING SHALLOW LAYERS ADAPTER")
    print("="*80)
    shallow_model, _ = load_model_with_adapter(args.base_model, args.shallow_adapter, args.device)
    shallow_metrics = evaluate_on_balanced_set(
        shallow_model, processor, test_items, args.device, "Shallow Layers (1-7)"
    )
    del shallow_model
    torch.cuda.empty_cache()
    
    # Compare results
    print("\n" + "="*80)
    print("📊 COMPARISON: NEGATION UNDERSTANDING")
    print("="*80)
    
    print(f"\n{'Model':<30} {'Overall':<12} {'No Acc':<12} {'Yes Acc':<12} {'Verdict':<20}")
    print("-" * 90)
    
    for metrics, name in [
        (baseline_metrics, "Baseline"),
        (deep_metrics, "Deep Layers"),
        (shallow_metrics, "Shallow Layers")
    ]:
        overall = metrics['overall_accuracy']
        no_acc = metrics['no_accuracy']
        yes_acc = metrics['yes_accuracy']
        
        # Verdict: if Yes accuracy is very low, model might just be saying No always
        if yes_acc < 50:
            verdict = "⚠️  Always says No?"
        elif overall > 80:
            verdict = "✅ Understands negation"
        else:
            verdict = "❓ Mixed performance"
        
        print(f"{name:<30} {overall:>6.1f}%      {no_acc:>6.1f}%      {yes_acc:>6.1f}%      {verdict:<20}")
    
    print("\n" + "="*80)
    print("💡 Interpretation:")
    print("   - If 'Yes Acc' is very low (<50%), model might just be saying 'No' always")
    print("   - If both 'No Acc' and 'Yes Acc' are high (>80%), model truly understands")
    print("   - Baseline should have good 'Yes Acc' (it wasn't fine-tuned)")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

