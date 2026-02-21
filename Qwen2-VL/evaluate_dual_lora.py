"""
Evaluation Script for Dual LoRA Adapters
==========================================

This script evaluates and compares the performance of two LoRA adapters
(deep layers vs shallow layers) on the negation task.

It loads each adapter, runs inference on a test set, and reports metrics.

Author: Generated for educational purposes
"""

import os
import sys
import glob
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

# Add paths for utilities
current_script_path = Path(__file__).parent.absolute()
sys.path.append("/home/sc305/VLM/Qwen2-VL")
os.chdir("/home/sc305/VLM/Qwen2-VL")

from patchscopes_utils import prepare_inputs  # noqa: E402


def load_base_model(model_path: str, device: str):
    """
    Load the base Qwen2-VL model and processor.
    
    Args:
        model_path: Path to base model (e.g., "Qwen/Qwen2-VL-7B-Instruct")
        device: Device to load model on
        
    Returns:
        Tuple of (model, processor)
    """
    print(f"\n📦 Loading base model from {model_path}...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map=device
    )
    processor = AutoProcessor.from_pretrained(model_path)
    model.eval()
    print("   ✓ Base model loaded")
    return model, processor


def load_lora_adapter(base_model, adapter_path: str, adapter_name: str):
    """
    Load a LoRA adapter onto the base model.
    
    Args:
        base_model: Base Qwen2-VL model
        adapter_path: Path to saved LoRA adapter
        adapter_name: Name for logging
        
    Returns:
        PEFT model with adapter loaded
    """
    print(f"\n🔧 Loading {adapter_name} adapter from {adapter_path}...")
    
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Adapter not found at {adapter_path}")
    
    # Load adapter
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)
    peft_model.eval()
    
    # Load metadata if available
    metadata_path = Path(adapter_path) / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        print(f"   Adapter info:")
        print(f"   - Layers: {metadata.get('layer_range', 'N/A')}")
        print(f"   - Training loss: {metadata.get('average_loss', 'N/A'):.4f}")
        print(f"   - Training steps: {metadata.get('total_steps', 'N/A')}")
    
    print(f"   ✓ {adapter_name} adapter loaded")
    return peft_model


def load_category_mapping(mapping_path: str) -> Dict[str, str]:
    """Load category ID to name mapping."""
    with open(mapping_path, "r") as f:
        return json.load(f)


def get_category_from_filename(filename: str) -> str:
    """Extract category ID from filename like '1000_000000508602.jpg'"""
    base = os.path.splitext(filename)[0]
    category_id = base.split("_")[0] if "_" in base else "0"
    return category_id


def evaluate_response(response_text: str, category: str) -> int:
    """
    Convert a free-form model response into a binary prediction.
    Returns 1 if presence is detected (Yes), otherwise 0 (No).
    """
    response_lower = response_text.lower().strip()
    
    # Clear yes/no detection
    if "yes" in response_lower and "no" not in response_lower:
        return 1
    if "no" in response_lower and "yes" not in response_lower:
        return 0
    
    # Fallback: check if category name is mentioned
    return 1 if category.lower() in response_lower else 0


def run_inference(model, processor, image_path: str, prompt: str, device: str, max_new_tokens: int = 20) -> str:
    """
    Run inference on a single image with the given prompt.
    
    Args:
        model: Model to use (can be base or PEFT model)
        processor: Qwen2-VL processor
        image_path: Path to image file
        prompt: Text prompt
        device: Device string
        max_new_tokens: Max tokens to generate
        
    Returns:
        Generated text response
    """
    # Load and prepare image
    image = Image.open(image_path).resize((384, 384))
    inputs = prepare_inputs(prompt, image, processor, device)
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    
    # Decode (only the generated part)
    decoded = processor.batch_decode(
        [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], output_ids)],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    
    return decoded


def load_test_set_from_sft_jsonl(jsonl_path: str) -> List[Dict]:
    """
    Load test set directly from SFT-formatted JSONL.
    
    Args:
        jsonl_path: Path to SFT JSONL file
        
    Returns:
        List of dicts with: image_path, prompt, category
    """
    test_examples = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                example = json.loads(line)
                user_msg = example['messages'][0]
                
                # Extract image path and prompt
                image_path = None
                prompt = None
                
                for content_item in user_msg['content']:
                    if content_item.get('type') == 'image':
                        image_path = content_item['image']
                    elif content_item.get('type') == 'text':
                        prompt = content_item['text']
                
                if image_path and prompt:
                    # Extract category from prompt (e.g., "Is there a {category} in the image?")
                    category = "unknown"
                    if "Is there a " in prompt and " in the image?" in prompt:
                        start = prompt.find("Is there a ") + len("Is there a ")
                        end = prompt.find(" in the image?")
                        category = prompt[start:end]
                    
                    test_examples.append({
                        'image_path': image_path,
                        'prompt': prompt,
                        'category': category
                    })
    
    return test_examples


def evaluate_model(
    model,
    processor,
    test_images: List[str] = None,
    category_mapping: Dict[str, str] = None,
    expected_label: int = 0,
    device: str = "cuda:0",
    model_name: str = "Model",
    save_results_path: str = None,
    test_examples_from_jsonl: List[Dict] = None
) -> Dict:
    """
    Evaluate a model (base or with adapter) on test images.
    
    Args:
        model: Model to evaluate
        processor: Processor for inputs
        test_images: List of image paths
        category_mapping: Category ID to name mapping
        expected_label: Expected label (0 or 1)
        device: Device string
        model_name: Name for logging
        save_results_path: Optional path to save detailed results
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n{'='*80}")
    print(f"📊 Evaluating: {model_name}")
    print(f"{'='*80}")
    
    # Determine test data source
    if test_examples_from_jsonl:
        # Use data from SFT JSONL
        test_data = test_examples_from_jsonl
        print(f"   Test samples: {len(test_data)} (from SFT JSONL)")
    else:
        # Use old method: convert image paths to test data format
        test_data = []
        if test_images:
            for img_path in test_images:
                fname = os.path.basename(img_path)
                category_id = get_category_from_filename(fname)
                category = category_mapping.get(category_id, "unknown") if category_mapping else "unknown"
                prompt = f"Is there a {category} in the image? Answer yes or no."
                test_data.append({
                    'image_path': img_path,
                    'prompt': prompt,
                    'category': category
                })
        print(f"   Test samples: {len(test_data)} (from directory)")
    
    print(f"   Expected label: {expected_label} ({'No' if expected_label == 0 else 'Yes'})")
    
    results = []
    correct = 0
    total = 0
    
    # Track per-category performance
    category_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    
    for test_item in tqdm(test_data, desc=f"Evaluating {model_name}"):
        img_path = test_item['image_path']
        prompt = test_item['prompt']
        category = test_item['category']
        fname = os.path.basename(img_path)
        
        try:
            # Run inference
            response = run_inference(model, processor, img_path, prompt, device)
            prediction = evaluate_response(response, category)
            
            # Check correctness
            is_correct = (prediction == expected_label)
            if is_correct:
                correct += 1
            total += 1
            
            # Update category stats
            category_stats[category]["total"] += 1
            if is_correct:
                category_stats[category]["correct"] += 1
            
            # Store result
            # Try to extract category_id from filename (for backward compatibility)
            # If filename format is {category_id}_{image_id}.jpg, extract it
            category_id = None
            try:
                if "_" in fname:
                    category_id = fname.split("_")[0]
            except:
                category_id = None
            
            results.append({
                "filename": fname,
                "image_path": img_path,
                "category_id": category_id,
                "category": category,
                "prompt": prompt,
                "response": response,
                "prediction": prediction,
                "expected": expected_label,
                "correct": is_correct
            })
            
        except Exception as e:
            print(f"\n   ⚠ Error processing {fname}: {e}")
            continue
    
    # Calculate metrics
    accuracy = (correct / total * 100) if total > 0 else 0.0
    
    print(f"\n✅ Evaluation complete!")
    print(f"   Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    # Per-category breakdown
    print(f"\n📋 Per-category accuracy:")
    category_accuracies = {}
    for category, stats in sorted(category_stats.items()):
        cat_acc = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0.0
        category_accuracies[category] = cat_acc
        print(f"   {category:20s}: {cat_acc:6.2f}% ({stats['correct']}/{stats['total']})")
    
    # Save detailed results if requested
    if save_results_path:
        os.makedirs(os.path.dirname(save_results_path), exist_ok=True)
        with open(save_results_path, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")
        print(f"\n💾 Detailed results saved to {save_results_path}")
    
    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "category_accuracies": category_accuracies,
        "results": results
    }


def compare_results(baseline_metrics: Dict, deep_metrics: Dict, shallow_metrics: Dict):
    """
    Print a comparison table of all three models.
    
    Args:
        baseline_metrics: Metrics from base model
        deep_metrics: Metrics from deep layers adapter
        shallow_metrics: Metrics from shallow layers adapter
    """
    print("\n" + "="*80)
    print("📊 FINAL COMPARISON")
    print("="*80)
    
    print(f"\n{'Model':<30} {'Accuracy':<15} {'Correct/Total':<20} {'Improvement':<15}")
    print("-" * 80)
    
    baseline_acc = baseline_metrics["accuracy"]
    
    # Baseline
    print(f"{'Baseline (No LoRA)':<30} {baseline_acc:>6.2f}%{'':<7} "
          f"{baseline_metrics['correct']:>4}/{baseline_metrics['total']:<4}       {'---':<15}")
    
    # Deep layers
    deep_acc = deep_metrics["accuracy"]
    deep_improvement = deep_acc - baseline_acc
    deep_symbol = "📈" if deep_improvement > 0 else ("📉" if deep_improvement < 0 else "➡️")
    print(f"{'Deep Layers (12-18)':<30} {deep_acc:>6.2f}%{'':<7} "
          f"{deep_metrics['correct']:>4}/{deep_metrics['total']:<4}       "
          f"{deep_symbol} {deep_improvement:+.2f}%")
    
    # Shallow layers
    shallow_acc = shallow_metrics["accuracy"]
    shallow_improvement = shallow_acc - baseline_acc
    shallow_symbol = "📈" if shallow_improvement > 0 else ("📉" if shallow_improvement < 0 else "➡️")
    print(f"{'Shallow Layers (1-7)':<30} {shallow_acc:>6.2f}%{'':<7} "
          f"{shallow_metrics['correct']:>4}/{shallow_metrics['total']:<4}       "
          f"{shallow_symbol} {shallow_improvement:+.2f}%")
    
    print("\n" + "="*80)
    
    # Determine winner
    best_acc = max(baseline_acc, deep_acc, shallow_acc)
    if best_acc == deep_acc and deep_acc > baseline_acc:
        print("🏆 Winner: Deep Layers Adapter")
        print(f"   Improved accuracy by {deep_improvement:.2f}% over baseline")
    elif best_acc == shallow_acc and shallow_acc > baseline_acc:
        print("🏆 Winner: Shallow Layers Adapter")
        print(f"   Improved accuracy by {shallow_improvement:.2f}% over baseline")
    elif baseline_acc >= max(deep_acc, shallow_acc):
        print("🏆 Winner: Baseline (No fine-tuning needed)")
        print("   LoRA adapters did not improve performance")
    
    # Category-level comparison
    print(f"\n📋 Best adapter per category:")
    all_categories = set(baseline_metrics["category_accuracies"].keys())
    
    for category in sorted(all_categories):
        baseline_cat = baseline_metrics["category_accuracies"].get(category, 0)
        deep_cat = deep_metrics["category_accuracies"].get(category, 0)
        shallow_cat = shallow_metrics["category_accuracies"].get(category, 0)
        
        best_cat_acc = max(baseline_cat, deep_cat, shallow_cat)
        
        if best_cat_acc == deep_cat and deep_cat > baseline_cat:
            winner = "Deep"
        elif best_cat_acc == shallow_cat and shallow_cat > baseline_cat:
            winner = "Shallow"
        else:
            winner = "Baseline"
        
        print(f"   {category:20s}: {winner:10s} ({best_cat_acc:.1f}%)")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate dual LoRA adapters")
    
    # Model paths
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2-VL-7B-Instruct",
        help="Path to base model"
    )
    parser.add_argument(
        "--deep_adapter",
        type=str,
        default="./lora_adapters/deep_layers_12-18",
        help="Path to deep layers adapter"
    )
    parser.add_argument(
        "--shallow_adapter",
        type=str,
        default="./lora_adapters/shallow_layers_1-7",
        help="Path to shallow layers adapter"
    )
    
    # Dataset paths
    parser.add_argument(
        "--test_dir",
        type=str,
        default="/home/sc305/VLM/Final Experiments/negation/v-patchscope-demo-code/surf-exp/qwen/results/misclassified_images",
        help="Directory with test images"
    )
    parser.add_argument(
        "--test_jsonl",
        type=str,
        default="qwen_balanced_sft_test.jsonl",
        help="Test set JSONL to filter which images to evaluate (prevents data leakage)"
    )
    parser.add_argument(
        "--category_mapping",
        type=str,
        default="/home/sc305/VLM/Final Experiments/negation/v-patchscope-demo-code/surf-exp/test_images_split/neg_object_map.json",
        help="Path to category mapping JSON"
    )
    
    # Evaluation settings
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:2" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--expected_label",
        type=int,
        default=0,
        choices=[0, 1],
        help="Expected label for test set (0=No, 1=Yes)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of test samples (None = all)"
    )
    
    # Output paths
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save evaluation results"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🎯 DUAL LORA ADAPTER EVALUATION")
    print("="*80)
    print(f"\n📋 Configuration:")
    print(f"   Base model: {args.base_model}")
    print(f"   Deep adapter: {args.deep_adapter}")
    print(f"   Shallow adapter: {args.shallow_adapter}")
    print(f"   Test directory: {args.test_dir}")
    print(f"   Device: {args.device}")
    print(f"   Expected label: {args.expected_label} ({'No' if args.expected_label == 0 else 'Yes'})")
    
    # Check if we should load directly from SFT JSONL
    use_sft_jsonl = False
    test_examples_from_jsonl = None
    
    if args.test_jsonl and os.path.exists(args.test_jsonl):
        print(f"\n📂 Loading test set from SFT JSONL: {args.test_jsonl}...")
        test_examples_from_jsonl = load_test_set_from_sft_jsonl(args.test_jsonl)
        print(f"   ✓ Loaded {len(test_examples_from_jsonl)} test examples from JSONL")
        
        if len(test_examples_from_jsonl) > 0:
            use_sft_jsonl = True
            # Verify images exist
            missing = 0
            for item in test_examples_from_jsonl:
                if not os.path.exists(item['image_path']):
                    missing += 1
            if missing > 0:
                print(f"   ⚠️  Warning: {missing} images not found")
            else:
                print(f"   ✓ All {len(test_examples_from_jsonl)} images found")
    
    # If not using SFT JSONL, use old method
    if not use_sft_jsonl:
        # Load category mapping
        print(f"\n📂 Loading category mapping from {args.category_mapping}...")
        if os.path.exists(args.category_mapping):
            category_mapping = load_category_mapping(args.category_mapping)
            print(f"   ✓ Loaded {len(category_mapping)} categories")
        else:
            print(f"   ⚠️  Warning: Category mapping not found, using defaults")
            category_mapping = {}
        
        # Load test set from JSONL to get the exact images we should evaluate on
        print(f"\n📂 Loading test set from {args.test_jsonl}...")
        test_set_filenames = set()
        if args.test_jsonl and os.path.exists(args.test_jsonl):
            with open(args.test_jsonl, 'r') as f:
                for line in f:
                    if line.strip():
                        example = json.loads(line)
                        user_msg = example['messages'][0]
                        for content_item in user_msg['content']:
                            if content_item.get('type') == 'image':
                                image_path = content_item['image']
                                filename = os.path.basename(image_path)
                                test_set_filenames.add(filename)
                                break
            print(f"   ✓ Test set contains {len(test_set_filenames)} examples")
        else:
            print(f"   ⚠️  Warning: Test JSONL not found at {args.test_jsonl}")
            print(f"   ⚠️  Will evaluate on ALL images in {args.test_dir}")
        
        # Collect test images from directory
        print(f"\n📂 Collecting test images from {args.test_dir}...")
        all_images = glob.glob(os.path.join(args.test_dir, "*.jpg")) + \
                     glob.glob(os.path.join(args.test_dir, "*.png"))
        
        if not all_images:
            print(f"❌ No images found in {args.test_dir}")
            return
        
        # Filter to only test set images if we have a test JSONL
        if test_set_filenames:
            test_images = [img for img in all_images if os.path.basename(img) in test_set_filenames]
            print(f"   ✓ Filtered to {len(test_images)} test set images")
        else:
            test_images = all_images
            print(f"   ⚠️  Using all {len(test_images)} images (no filtering)")
        
        if len(test_images) == 0:
            print(f"❌ No test images remaining after filtering!")
            return
        
        if args.max_samples:
            test_images = test_images[:args.max_samples]
    else:
        # Using SFT JSONL, set dummy values for old code path
        test_images = []
        category_mapping = {}
        
        if args.max_samples:
            test_examples_from_jsonl = test_examples_from_jsonl[:args.max_samples]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load base model
    base_model, processor = load_base_model(args.base_model, args.device)
    
    # === EVALUATE BASELINE ===
    baseline_metrics = evaluate_model(
        model=base_model,
        processor=processor,
        test_images=test_images if not use_sft_jsonl else None,
        category_mapping=category_mapping if not use_sft_jsonl else None,
        expected_label=args.expected_label,
        device=args.device,
        model_name="Baseline (No LoRA)",
        save_results_path=os.path.join(args.output_dir, "baseline_results.jsonl"),
        test_examples_from_jsonl=test_examples_from_jsonl if use_sft_jsonl else None
    )
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    
    # === EVALUATE DEEP LAYERS ADAPTER ===
    deep_model = load_lora_adapter(base_model, args.deep_adapter, "Deep Layers (12-18)")
    deep_metrics = evaluate_model(
        model=deep_model,
        processor=processor,
        test_images=test_images if not use_sft_jsonl else None,
        category_mapping=category_mapping if not use_sft_jsonl else None,
        expected_label=args.expected_label,
        device=args.device,
        model_name="Deep Layers (12-18)",
        save_results_path=os.path.join(args.output_dir, "deep_layers_results.jsonl"),
        test_examples_from_jsonl=test_examples_from_jsonl if use_sft_jsonl else None
    )
    
    # Cleanup deep model
    del deep_model
    torch.cuda.empty_cache()
    
    # === EVALUATE SHALLOW LAYERS ADAPTER ===
    shallow_model = load_lora_adapter(base_model, args.shallow_adapter, "Shallow Layers (1-7)")
    shallow_metrics = evaluate_model(
        model=shallow_model,
        processor=processor,
        test_images=test_images if not use_sft_jsonl else None,
        category_mapping=category_mapping if not use_sft_jsonl else None,
        expected_label=args.expected_label,
        device=args.device,
        model_name="Shallow Layers (1-7)",
        save_results_path=os.path.join(args.output_dir, "shallow_layers_results.jsonl"),
        test_examples_from_jsonl=test_examples_from_jsonl if use_sft_jsonl else None
    )
    
    # Cleanup
    del shallow_model, base_model, processor
    torch.cuda.empty_cache()
    
    # === COMPARE RESULTS ===
    compare_results(baseline_metrics, deep_metrics, shallow_metrics)
    
    # Save summary
    summary = {
        "baseline": {
            "accuracy": baseline_metrics["accuracy"],
            "correct": baseline_metrics["correct"],
            "total": baseline_metrics["total"],
            "category_accuracies": baseline_metrics["category_accuracies"]
        },
        "deep_layers": {
            "accuracy": deep_metrics["accuracy"],
            "correct": deep_metrics["correct"],
            "total": deep_metrics["total"],
            "improvement": deep_metrics["accuracy"] - baseline_metrics["accuracy"],
            "category_accuracies": deep_metrics["category_accuracies"]
        },
        "shallow_layers": {
            "accuracy": shallow_metrics["accuracy"],
            "correct": shallow_metrics["correct"],
            "total": shallow_metrics["total"],
            "improvement": shallow_metrics["accuracy"] - baseline_metrics["accuracy"],
            "category_accuracies": shallow_metrics["category_accuracies"]
        }
    }
    
    summary_path = os.path.join(args.output_dir, "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Evaluation summary saved to {summary_path}")
    print(f"\n✅ Evaluation complete! Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()

