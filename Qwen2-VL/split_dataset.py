"""
Dataset Splitting Script
========================

Splits the misclassified dataset into train and test sets.
This ensures proper evaluation by preventing data leakage.

Author: Generated for educational purposes
"""

import json
import random
import argparse
from pathlib import Path
from collections import defaultdict


def load_jsonl(file_path):
    """Load JSONL file into a list of examples."""
    examples = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def save_jsonl(examples, file_path):
    """Save examples to JSONL file."""
    with open(file_path, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')


def get_category_from_example(example):
    """Extract category from the prompt in an example."""
    # Example prompt: "Is there a {category} in the image? Answer yes or no."
    user_msg = example['messages'][0]
    for content_item in user_msg['content']:
        if content_item.get('type') == 'text':
            text = content_item['text']
            # Extract category between "Is there a " and " in the image?"
            if "Is there a " in text and " in the image?" in text:
                start = text.find("Is there a ") + len("Is there a ")
                end = text.find(" in the image?")
                category = text[start:end]
                return category
    return "unknown"


def stratified_split(examples, test_ratio=0.2, seed=42):
    """
    Split examples into train and test sets using stratified sampling.
    This ensures each category is represented proportionally in both sets.
    
    Args:
        examples: List of examples to split
        test_ratio: Proportion of data for test set (0.0 to 1.0)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_examples, test_examples)
    """
    random.seed(seed)
    
    # Group examples by category
    category_examples = defaultdict(list)
    for example in examples:
        category = get_category_from_example(example)
        category_examples[category].append(example)
    
    train_examples = []
    test_examples = []
    
    print(f"\n📊 Stratified split by category:")
    print(f"{'Category':<20} {'Total':<8} {'Train':<8} {'Test':<8}")
    print("-" * 50)
    
    for category, cat_examples in sorted(category_examples.items()):
        # Shuffle examples within category
        random.shuffle(cat_examples)
        
        # Calculate split point
        n_test = max(1, int(len(cat_examples) * test_ratio))  # At least 1 test sample per category
        n_train = len(cat_examples) - n_test
        
        # Split
        train_examples.extend(cat_examples[:n_train])
        test_examples.extend(cat_examples[n_train:])
        
        print(f"{category:<20} {len(cat_examples):<8} {n_train:<8} {n_test:<8}")
    
    # Shuffle the final splits
    random.shuffle(train_examples)
    random.shuffle(test_examples)
    
    print("-" * 50)
    print(f"{'TOTAL':<20} {len(examples):<8} {len(train_examples):<8} {len(test_examples):<8}")
    
    return train_examples, test_examples


def main():
    parser = argparse.ArgumentParser(description="Split dataset into train/test sets")
    
    parser.add_argument(
        "--input",
        type=str,
        default="qwen_misclassified_sft.jsonl",
        help="Input JSONL file with all examples"
    )
    parser.add_argument(
        "--train_output",
        type=str,
        default="qwen_misclassified_sft_train.jsonl",
        help="Output file for training set"
    )
    parser.add_argument(
        "--test_output",
        type=str,
        default="qwen_misclassified_sft_test.jsonl",
        help="Output file for test set"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.25,
        help="Proportion of data for test set (default: 0.25 = 25%%)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("📊 DATASET SPLITTING")
    print("="*80)
    print(f"\n⚙️  Configuration:")
    print(f"   Input file: {args.input}")
    print(f"   Test ratio: {args.test_ratio:.1%}")
    print(f"   Random seed: {args.seed}")
    print(f"   Train output: {args.train_output}")
    print(f"   Test output: {args.test_output}")
    
    # Load full dataset
    print(f"\n📂 Loading dataset from {args.input}...")
    examples = load_jsonl(args.input)
    print(f"   ✓ Loaded {len(examples)} examples")
    
    if len(examples) == 0:
        print("❌ Error: No examples found in input file!")
        return
    
    # Perform stratified split
    train_examples, test_examples = stratified_split(
        examples, 
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    # Save splits
    print(f"\n💾 Saving splits...")
    save_jsonl(train_examples, args.train_output)
    print(f"   ✓ Saved {len(train_examples)} training examples to {args.train_output}")
    
    save_jsonl(test_examples, args.test_output)
    print(f"   ✓ Saved {len(test_examples)} test examples to {args.test_output}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("✅ SPLIT COMPLETE!")
    print("="*80)
    print(f"\n📊 Summary:")
    print(f"   Total examples: {len(examples)}")
    print(f"   Training set: {len(train_examples)} ({len(train_examples)/len(examples):.1%})")
    print(f"   Test set: {len(test_examples)} ({len(test_examples)/len(examples):.1%})")
    
    print(f"\n💡 Next steps:")
    print(f"   1. Train your models using: {args.train_output}")
    print(f"   2. Evaluate on the held-out test set")
    print(f"   3. Use test set images for fair comparison")
    
    print("\n⚠️  Important: DO NOT use test set examples during training!")
    print("   This would cause data leakage and invalid results.\n")


if __name__ == "__main__":
    main()

