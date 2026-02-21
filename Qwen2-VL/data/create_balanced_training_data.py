"""
Create Balanced Training Data
=============================

Adds Yes samples to the training data to prevent models from learning
to always say "No". Creates a balanced dataset with both Yes and No answers.
"""

import json
import os
import pandas as pd
import ast
import random
from collections import Counter


def load_existing_sft_data(jsonl_path):
    """Load existing SFT-formatted data."""
    examples = []
    if os.path.exists(jsonl_path):
        with open(jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
    return examples


def create_yes_samples_from_negbench(negbench_csv_path, categories_needed, num_samples_per_category=50):
    """
    Create Yes samples from NegBench CSV where objects are in positive_objects.
    
    Args:
        negbench_csv_path: Path to negbench_with_coco.csv
        categories_needed: Set of categories we need Yes samples for
        num_samples_per_category: Target number of samples per category
    
    Returns:
        List of SFT-formatted examples with "Yes." answers
    """
    print(f"\n📂 Loading NegBench CSV from {negbench_csv_path}...")
    df = pd.read_csv(negbench_csv_path)
    print(f"   Loaded {len(df)} rows")
    
    yes_samples = []
    used_image_ids = set()
    
    # Shuffle for random sampling
    random.seed(42)
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    category_counts = Counter()
    
    print(f"\n🔍 Looking for Yes samples for categories: {categories_needed}")
    
    for idx, row in df_shuffled.iterrows():
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
        
        # Check if any needed category is in positive_objects
        for category in categories_needed:
            if category_counts[category] >= num_samples_per_category:
                continue
            
            if category in pos_objs:
                # Use new_filepath if available, otherwise build from image_id
                img_path = row.get('new_filepath')
                if not img_path or not os.path.exists(img_path):
                    if image_id:
                        img_path = f"/home/sc305/VLM/Qwen2-VL/data/val2017/{image_id:012d}.jpg"
                
                if img_path and os.path.exists(img_path):
                    sft_example = {
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": img_path},
                                    {"type": "text", "text": f"Is there a {category} in the image? Answer yes or no."}
                                ]
                            },
                            {
                                "role": "assistant",
                                "content": [{"type": "text", "text": "Yes."}]
                            }
                        ]
                    }
                    yes_samples.append(sft_example)
                    used_image_ids.add(image_id)
                    category_counts[category] += 1
                    break  # Only use this image once
        
        # Check if we have enough samples
        if all(category_counts[c] >= num_samples_per_category for c in categories_needed):
            break
    
    print(f"\n✅ Created {len(yes_samples)} Yes samples:")
    for category, count in category_counts.items():
        print(f"   - {category}: {count} samples")
    
    return yes_samples


def extract_categories_from_no_samples(no_samples):
    """Extract unique categories from No samples."""
    categories = set()
    for example in no_samples:
        prompt = example['messages'][0]['content'][1]['text']
        # Extract category from prompt like "Is there a {category} in the image?"
        if 'Is there a' in prompt and 'in the image' in prompt:
            try:
                category = prompt.split('Is there a ')[1].split(' in the image')[0].strip()
                categories.add(category)
            except:
                pass
    return categories


def create_balanced_dataset(no_samples, yes_samples, train_ratio=0.8):
    """
    Combine No and Yes samples and split into train/test sets.
    
    Args:
        no_samples: List of SFT examples with "No." answers
        yes_samples: List of SFT examples with "Yes." answers
        train_ratio: Ratio of data to use for training
    
    Returns:
        train_data, test_data (both lists of SFT examples)
    """
    print(f"\n📊 Creating balanced dataset...")
    print(f"   No samples: {len(no_samples)}")
    print(f"   Yes samples: {len(yes_samples)}")
    
    # Balance the dataset (use min to ensure equal distribution)
    min_count = min(len(no_samples), len(yes_samples))
    print(f"\n   Balancing to {min_count} samples per label...")
    
    random.seed(42)
    balanced_no = random.sample(no_samples, min_count)
    balanced_yes = random.sample(yes_samples, min_count)
    
    # Combine and shuffle
    all_samples = balanced_no + balanced_yes
    random.shuffle(all_samples)
    
    print(f"   Total balanced samples: {len(all_samples)} ({min_count} No + {min_count} Yes)")
    
    # Split into train/test
    split_idx = int(len(all_samples) * train_ratio)
    train_data = all_samples[:split_idx]
    test_data = all_samples[split_idx:]
    
    # Count labels in each split
    train_no = sum(1 for ex in train_data if ex['messages'][1]['content'][0]['text'].strip().lower().startswith('no'))
    train_yes = len(train_data) - train_no
    test_no = sum(1 for ex in test_data if ex['messages'][1]['content'][0]['text'].strip().lower().startswith('no'))
    test_yes = len(test_data) - test_no
    
    print(f"\n   Train split: {len(train_data)} samples ({train_no} No, {train_yes} Yes)")
    print(f"   Test split: {len(test_data)} samples ({test_no} No, {test_yes} Yes)")
    
    return train_data, test_data


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Create balanced training data with Yes and No samples")
    parser.add_argument(
        "--no_samples_jsonl",
        type=str,
        default="/home/sc305/VLM/Qwen2-VL/qwen_misclassified_sft.jsonl",
        help="Path to existing No samples (SFT format)"
    )
    parser.add_argument(
        "--negbench_csv",
        type=str,
        default="/home/sc305/VLM/Qwen2-VL/data/negbench_with_coco.csv",
        help="Path to NegBench CSV with positive_objects"
    )
    parser.add_argument(
        "--output_train",
        type=str,
        default="/home/sc305/VLM/Qwen2-VL/qwen_balanced_sft_train.jsonl",
        help="Output path for balanced training set"
    )
    parser.add_argument(
        "--output_test",
        type=str,
        default="/home/sc305/VLM/Qwen2-VL/qwen_balanced_sft_test.jsonl",
        help="Output path for balanced test set"
    )
    parser.add_argument(
        "--num_yes_per_category",
        type=int,
        default=50,
        help="Number of Yes samples to create per category"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of data for training (rest for testing)"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("🔄 CREATING BALANCED TRAINING DATA")
    print("="*80)
    print("\nThis script will:")
    print("  1. Load existing No samples from misclassified data")
    print("  2. Create Yes samples from NegBench CSV (positive_objects)")
    print("  3. Balance the dataset (equal No and Yes)")
    print("  4. Split into train/test sets")
    print("="*80)
    
    # Step 1: Load existing No samples
    print(f"\n1️⃣  Loading No samples from {args.no_samples_jsonl}...")
    no_samples = load_existing_sft_data(args.no_samples_jsonl)
    print(f"   ✅ Loaded {len(no_samples)} No samples")
    
    # Step 2: Extract categories from No samples
    categories = extract_categories_from_no_samples(no_samples)
    print(f"\n   📋 Found {len(categories)} unique categories: {sorted(categories)}")
    
    # Step 3: Create Yes samples
    print(f"\n2️⃣  Creating Yes samples from NegBench CSV...")
    yes_samples = create_yes_samples_from_negbench(
        args.negbench_csv,
        categories,
        num_samples_per_category=args.num_yes_per_category
    )
    
    # Step 4: Create balanced dataset and split
    print(f"\n3️⃣  Creating balanced dataset and splitting...")
    train_data, test_data = create_balanced_dataset(
        no_samples,
        yes_samples,
        train_ratio=args.train_ratio
    )
    
    # Step 5: Save outputs
    print(f"\n4️⃣  Saving balanced datasets...")
    
    # Save training set
    with open(args.output_train, 'w') as f:
        for example in train_data:
            f.write(json.dumps(example) + '\n')
    print(f"   ✅ Saved training set: {args.output_train} ({len(train_data)} samples)")
    
    # Save test set
    with open(args.output_test, 'w') as f:
        for example in test_data:
            f.write(json.dumps(example) + '\n')
    print(f"   ✅ Saved test set: {args.output_test} ({len(test_data)} samples)")
    
    print("\n" + "="*80)
    print("✅ BALANCED DATASET CREATION COMPLETE!")
    print("="*80)
    print(f"\n📊 Summary:")
    print(f"   Training set: {args.output_train}")
    print(f"   Test set: {args.output_test}")
    print(f"   Total samples: {len(train_data) + len(test_data)}")
    print(f"   Balanced: {min(len(no_samples), len(yes_samples))} No + {min(len(no_samples), len(yes_samples))} Yes")
    print("\n💡 Next steps:")
    print(f"   1. Update train_dual_lora.py to use: {args.output_train}")
    print(f"   2. Update evaluate_dual_lora.py to use: {args.output_test}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

