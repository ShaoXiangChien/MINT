"""
Evaluation Script for Dual LoRA Adapters from NegBench CSV
==========================================================

This script evaluates and compares the performance of two LoRA adapters
(deep layers vs shallow layers) on the negation task using NegBench CSV format.

It loads each adapter, runs inference on a CSV-based test set, and reports metrics.

Author: Generated for educational purposes
"""

import os
import sys
import json
import ast
import argparse
import pandas as pd
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


def parse_negative_objects(neg_objs_str):
    """
    Parse negative_objects from CSV (could be string representation of list).
    
    Args:
        neg_objs_str: String or list from CSV column
        
    Returns:
        List of negative object names
    """
    if pd.isna(neg_objs_str):
        return ["unknown"]
    
    # If it's already a list, return as is
    if isinstance(neg_objs_str, list):
        return neg_objs_str if neg_objs_str else ["unknown"]
    
    # Try to parse as Python list literal
    try:
        parsed = ast.literal_eval(str(neg_objs_str))
        if isinstance(parsed, list) and len(parsed) > 0:
            return parsed
    except:
        pass
    
    # Fallback: treat as single string
    return [str(neg_objs_str)] if neg_objs_str else ["unknown"]


def load_dataset_from_csv(csv_path: str, coco_base_dir: str = None):
    """
    Load dataset from NegBench CSV format.
    
    Args:
        csv_path: Path to CSV file
        coco_base_dir: Base directory for COCO images (if filepath needs to be resolved)
        
    Returns:
        List of dicts with: image_path, category, image_id, negative_objects
    """
    print(f"\n📂 Loading dataset from CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"   Loaded {len(df)} rows from CSV")
    
    dataset = []
    for idx, row in df.iterrows():
        # Parse negative objects
        neg_objs = parse_negative_objects(row.get('negative_objects', 'unknown'))
        category = neg_objs[0] if neg_objs else "unknown"
        
        # Get image path
        image_id = row.get('image_id', None)
        filepath = row.get('filepath', None)
        new_filepath = row.get('new_filepath', None)  # Check for new_filepath column
        
        # Resolve image path - try multiple strategies
        img_path = None
        
        # Strategy 0: Use new_filepath if it exists (often has absolute paths)
        if new_filepath and os.path.exists(new_filepath):
            img_path = new_filepath
        
        # Strategy 1: Use filepath if it's absolute and exists
        elif filepath and os.path.isabs(filepath) and os.path.exists(filepath):
            img_path = filepath
        
        # Strategy 2: If filepath is relative, try to resolve it
        elif filepath and not os.path.isabs(filepath):
            # Extract filename from filepath (e.g., "data/coco/images/val2017/000000397133.jpg" -> "000000397133.jpg")
            filename = os.path.basename(filepath)
            
            # Try in coco_base_dir
            if coco_base_dir:
                candidate_path = os.path.join(coco_base_dir, filename)
                if os.path.exists(candidate_path):
                    img_path = candidate_path
            
            # If still not found, try common COCO paths (including Qwen2-VL/data)
            if not img_path:
                common_paths = [
                    '/home/sc305/VLM/Qwen2-VL/data/val2017',  # Your actual data location
                    '/home/sc305/VLM/data/val2017',
                    '/home/sc305/VLM/data/coco/val2017',
                    str(current_script_path / 'data' / 'val2017'),  # Relative to script
                ]
                for base in common_paths:
                    if not base:
                        continue
                    candidate = os.path.join(base, filename)
                    if os.path.exists(candidate):
                        img_path = candidate
                        break
        
        # Strategy 3: Build from image_id if we have it
        if not img_path and image_id:
            # Try different filename formats
            candidates = [
                f"{image_id:012d}.jpg",  # 12-digit zero-padded
                f"{image_id:06d}.jpg",   # 6-digit zero-padded
                f"{image_id}.jpg",       # No padding
            ]
            
            base_dirs = [
                coco_base_dir,
                '/home/sc305/VLM/Qwen2-VL/data/val2017',  # Your actual data location
                '/home/sc305/VLM/data/val2017',
                '/home/sc305/VLM/data/coco/val2017',
                str(current_script_path / 'data' / 'val2017'),  # Relative to script
            ]
            
            for base_dir in base_dirs:
                if not base_dir:
                    continue
                for candidate_filename in candidates:
                    candidate_path = os.path.join(base_dir, candidate_filename)
                    if os.path.exists(candidate_path):
                        img_path = candidate_path
                        break
                if img_path:
                    break
        
        # If still not found, skip this row
        if not img_path:
            if idx < 10:  # Only print first 10 warnings
                print(f"   ⚠️  Warning: Row {idx} - Could not resolve image path (filepath={filepath}, image_id={image_id})")
            continue
        
        # Verify image exists
        if not os.path.exists(img_path):
            if idx < 10:
                print(f"   ⚠️  Warning: Image not found: {img_path}")
            continue
        
        dataset.append({
            'image_path': img_path,
            'category': category,
            'image_id': image_id,
            'negative_objects': neg_objs,
            'row_index': idx
        })
    
    print(f"   ✅ Successfully loaded {len(dataset)} valid entries")
    return dataset


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


def load_processed_set(results_jsonl_path: str) -> set:
    """
    Load the set of already processed image paths.
    This enables resuming evaluation if interrupted.
    
    Args:
        results_jsonl_path: Path to results JSONL file
        
    Returns:
        Set of image paths that have already been processed
    """
    if not os.path.exists(results_jsonl_path):
        return set()
    
    seen = set()
    with open(results_jsonl_path, "r") as f:
        for line in f:
            try:
                record = json.loads(line)
                img_path = record.get("image_path")
                if img_path:
                    seen.add(img_path)
            except Exception:
                continue
    return seen


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    if path and path.strip():
        os.makedirs(path, exist_ok=True)


def evaluate_model(
    model,
    processor,
    dataset: List[Dict],
    expected_label: int = 0,
    device: str = "cuda:0",
    model_name: str = "Model",
    save_results_path: str = None,
    max_new_tokens: int = 20,
    resume: bool = True
) -> Dict:
    """
    Evaluate a model (base or with adapter) on CSV-loaded dataset.
    
    Args:
        model: Model to evaluate
        processor: Processor for inputs
        dataset: List of dicts with image_path, category, etc.
        expected_label: Expected label (0 or 1)
        device: Device string
        model_name: Name for logging
        save_results_path: Optional path to save detailed results
        max_new_tokens: Maximum tokens to generate
        resume: Whether to resume from existing results file
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n{'='*80}")
    print(f"📊 Evaluating: {model_name}")
    print(f"{'='*80}")
    
    print(f"   Test samples: {len(dataset)}")
    print(f"   Expected label: {expected_label} ({'No' if expected_label == 0 else 'Yes'})")
    
    # Resume support: check which images have already been processed
    processed = set()
    if resume and save_results_path:
        processed = load_processed_set(save_results_path)
        print(f"   Already processed: {len(processed)} images")
    
    # Filter to unprocessed images
    to_process = [d for d in dataset if d['image_path'] not in processed]
    print(f"   Remaining to process: {len(to_process)} images")
    
    if len(to_process) == 0:
        print("   ⚠️  All images already processed. Loading existing results...")
        # Load existing results
        results = []
        if save_results_path and os.path.exists(save_results_path):
            with open(save_results_path, "r") as f:
                for line in f:
                    try:
                        results.append(json.loads(line))
                    except Exception:
                        continue
    else:
        # Ensure output directory exists
        if save_results_path:
            ensure_dir(os.path.dirname(save_results_path))
        
        results = []
        correct = 0
        total = 0
        
        # Track per-category performance
        category_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        
        # Process each image
        for item in tqdm(to_process, desc=f"Evaluating {model_name}"):
            img_path = item['image_path']
            category = item['category']
            fname = os.path.basename(img_path)
            
            try:
                # Build prompt
                prompt = f"Is there a {category} in the image? Answer yes or no."
                
                # Run inference
                response = run_inference(model, processor, img_path, prompt, device, max_new_tokens)
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
                result = {
                    "filename": fname,
                    "image_path": img_path,
                    "image_id": item.get('image_id'),
                    "category": category,
                    "negative_objects": item.get('negative_objects', []),
                    "prompt": prompt,
                    "response": response,
                    "prediction": prediction,
                    "expected": expected_label,
                    "correct": is_correct
                }
                results.append(result)
                
                # Append to file immediately (for resume support)
                if save_results_path:
                    with open(save_results_path, "a") as f:
                        f.write(json.dumps(result) + "\n")
                
            except Exception as e:
                print(f"\n   ⚠ Error processing {fname}: {e}")
                continue
        
        # Load all existing results if resuming
        if resume and save_results_path and os.path.exists(save_results_path):
            existing_results = []
            with open(save_results_path, "r") as f:
                for line in f:
                    try:
                        existing_results.append(json.loads(line))
                    except Exception:
                        continue
            # Combine existing and new results
            existing_paths = {r.get("image_path") for r in existing_results}
            new_results = [r for r in results if r.get("image_path") not in existing_paths]
            results = existing_results + new_results
        
        # Recalculate metrics from all results
        correct = sum(1 for r in results if r.get("correct", False))
        total = len(results)
        category_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        for r in results:
            cat = r.get("category", "unknown")
            category_stats[cat]["total"] += 1
            if r.get("correct", False):
                category_stats[cat]["correct"] += 1
    
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
    
    if save_results_path:
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
    parser = argparse.ArgumentParser(description="Evaluate dual LoRA adapters from NegBench CSV")
    
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
        "--csv_file",
        type=str,
        required=True,
        help="Path to NegBench CSV file"
    )
    parser.add_argument(
        "--coco_base_dir",
        type=str,
        default="/home/sc305/VLM/data/val2017",
        help="Base directory for COCO images"
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
        "--max_new_tokens",
        type=int,
        default=20,
        help="Maximum number of tokens to generate per response"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of test samples (None = all)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume evaluation from existing results files"
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
    print("🎯 DUAL LORA ADAPTER EVALUATION FROM CSV")
    print("="*80)
    print(f"\n📋 Configuration:")
    print(f"   Base model: {args.base_model}")
    print(f"   Deep adapter: {args.deep_adapter}")
    print(f"   Shallow adapter: {args.shallow_adapter}")
    print(f"   CSV file: {args.csv_file}")
    print(f"   COCO base dir: {args.coco_base_dir}")
    print(f"   Device: {args.device}")
    print(f"   Expected label: {args.expected_label} ({'No' if args.expected_label == 0 else 'Yes'})")
    print(f"   Resume: {args.resume}")
    
    # Load dataset from CSV
    dataset = load_dataset_from_csv(args.csv_file, args.coco_base_dir)
    
    if not dataset:
        print("❌ ERROR: No valid images found in CSV!")
        return
    
    # Limit dataset size if requested
    if args.max_samples:
        dataset = dataset[:args.max_samples]
        print(f"\n📊 Limited to {len(dataset)} samples")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load base model
    base_model, processor = load_base_model(args.base_model, args.device)
    
    # === EVALUATE BASELINE ===
    baseline_metrics = evaluate_model(
        model=base_model,
        processor=processor,
        dataset=dataset,
        expected_label=args.expected_label,
        device=args.device,
        model_name="Baseline (No LoRA)",
        save_results_path=os.path.join(args.output_dir, "baseline_results.jsonl"),
        max_new_tokens=args.max_new_tokens,
        resume=args.resume
    )
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    
    # === EVALUATE DEEP LAYERS ADAPTER ===
    deep_model = load_lora_adapter(base_model, args.deep_adapter, "Deep Layers (12-18)")
    deep_metrics = evaluate_model(
        model=deep_model,
        processor=processor,
        dataset=dataset,
        expected_label=args.expected_label,
        device=args.device,
        model_name="Deep Layers (12-18)",
        save_results_path=os.path.join(args.output_dir, "deep_layers_results.jsonl"),
        max_new_tokens=args.max_new_tokens,
        resume=args.resume
    )
    
    # Cleanup deep model
    del deep_model
    torch.cuda.empty_cache()
    
    # === EVALUATE SHALLOW LAYERS ADAPTER ===
    shallow_model = load_lora_adapter(base_model, args.shallow_adapter, "Shallow Layers (1-7)")
    shallow_metrics = evaluate_model(
        model=shallow_model,
        processor=processor,
        dataset=dataset,
        expected_label=args.expected_label,
        device=args.device,
        model_name="Shallow Layers (1-7)",
        save_results_path=os.path.join(args.output_dir, "shallow_layers_results.jsonl"),
        max_new_tokens=args.max_new_tokens,
        resume=args.resume
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


