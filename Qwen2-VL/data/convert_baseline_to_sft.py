"""
Convert Baseline Results to SFT Format
======================================

Extracts misclassified samples from baseline_results_5000.jsonl,
converts them to SFT format, and splits into train/test sets.
"""

import json
import random
from pathlib import Path
from collections import defaultdict


def load_baseline_results(jsonl_path):
    """Load baseline results from JSONL file."""
    results = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def extract_misclassified(results):
    """Extract only misclassified samples (correct == 0)."""
    misclassified = [r for r in results if r.get('correct', 1) == 0]
    return misclassified


def convert_to_sft_format(misclassified):
    """Convert misclassified results to SFT format."""
    sft_data = []
    
    for item in misclassified:
        sft_example = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": item['image_path']},
                        {"type": "text", "text": item['prompt']}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "No."}  # Correct answer (expected_label is 0)
                    ]
                }
            ]
        }
        sft_data.append(sft_example)
    
    return sft_data


def stratified_split(sft_data, misclassified, test_ratio=0.25, seed=42):
    """
    Split SFT data into train/test sets using stratified sampling by category.
    
    Args:
        sft_data: List of SFT examples
        misclassified: Original misclassified results (for category info)
        test_ratio: Proportion for test set
        seed: Random seed
        
    Returns:
        Tuple of (train_examples, test_examples)
    """
    random.seed(seed)
    
    # Group by category
    category_examples = defaultdict(list)
    for sft_ex, orig in zip(sft_data, misclassified):
        category = orig.get('category', 'unknown')
        category_examples[category].append((sft_ex, orig))
    
    train_examples = []
    test_examples = []
    
    print(f"\n📊 Stratified split by category:")
    print(f"{'Category':<20} {'Total':<8} {'Train':<8} {'Test':<8}")
    print("-" * 50)
    
    for category, cat_examples in sorted(category_examples.items()):
        random.shuffle(cat_examples)
        
        # Calculate split
        n_test = max(1, int(len(cat_examples) * test_ratio))
        n_train = len(cat_examples) - n_test
        
        # Split
        train_examples.extend([ex[0] for ex in cat_examples[:n_train]])
        test_examples.extend([ex[0] for ex in cat_examples[n_train:]])
        
        print(f"{category:<20} {len(cat_examples):<8} {n_train:<8} {n_test:<8}")
    
    # Shuffle final splits
    random.shuffle(train_examples)
    random.shuffle(test_examples)
    
    print("-" * 50)
    print(f"{'TOTAL':<20} {len(sft_data):<8} {len(train_examples):<8} {len(test_examples):<8}")
    
    return train_examples, test_examples


def save_jsonl(examples, file_path):
    """Save examples to JSONL file."""
    with open(file_path, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')


def main():
    # Paths
    baseline_results_path = '/home/sc305/VLM/Qwen2-VL/baseline_results_5000.jsonl'
    output_dir = Path('/home/sc305/VLM/Qwen2-VL')
    
    # Output files (will replace existing)
    train_output = output_dir / 'qwen_misclassified_sft_train.jsonl'
    test_output = output_dir / 'qwen_misclassified_sft_test.jsonl'
    full_output = output_dir / 'qwen_misclassified_sft.jsonl'
    
    print("="*80)
    print("🔄 CONVERTING BASELINE RESULTS TO SFT FORMAT")
    print("="*80)
    
    # Step 1: Load baseline results
    print(f"\n📂 Loading baseline results from {baseline_results_path}...")
    all_results = load_baseline_results(baseline_results_path)
    print(f"   ✓ Loaded {len(all_results)} total results")
    
    # Step 2: Extract misclassified
    print(f"\n🔍 Extracting misclassified samples...")
    misclassified = extract_misclassified(all_results)
    print(f"   ✓ Found {len(misclassified)} misclassified samples")
    
    if len(misclassified) == 0:
        print("❌ No misclassified samples found! Cannot create training dataset.")
        return
    
    # Step 3: Convert to SFT format
    print(f"\n📝 Converting to SFT format...")
    sft_data = convert_to_sft_format(misclassified)
    print(f"   ✓ Converted {len(sft_data)} examples to SFT format")
    
    # Step 4: Split into train/test
    print(f"\n✂️  Splitting into train/test sets...")
    train_examples, test_examples = stratified_split(sft_data, misclassified, test_ratio=0.25, seed=42)
    
    # Step 5: Save files
    print(f"\n💾 Saving files...")
    
    # Save full dataset
    save_jsonl(sft_data, full_output)
    print(f"   ✓ Saved full dataset ({len(sft_data)} examples) to {full_output}")
    
    # Save train set
    save_jsonl(train_examples, train_output)
    print(f"   ✓ Saved training set ({len(train_examples)} examples) to {train_output}")
    
    # Save test set
    save_jsonl(test_examples, test_output)
    print(f"   ✓ Saved test set ({len(test_examples)} examples) to {test_output}")
    
    # Summary
    print("\n" + "="*80)
    print("✅ CONVERSION COMPLETE!")
    print("="*80)
    print(f"\n📊 Summary:")
    print(f"   Total misclassified: {len(misclassified)}")
    print(f"   Training set: {len(train_examples)} ({len(train_examples)/len(misclassified):.1%})")
    print(f"   Test set: {len(test_examples)} ({len(test_examples)/len(misclassified):.1%})")
    print(f"\n📁 Files created:")
    print(f"   - {full_output}")
    print(f"   - {train_output}")
    print(f"   - {test_output}")
    print("\n💡 Next steps:")
    print(f"   1. Train your models using: {train_output}")
    print(f"   2. Evaluate on the test set to measure improvement")
    print(f"   3. Use test set for fair comparison between adapters\n")


if __name__ == "__main__":
    main()

