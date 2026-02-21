"""
Extract Test Image Paths
=========================

Extracts the image paths from the test set JSONL for evaluation purposes.
Creates a text file with one image path per line.

Author: Generated for educational purposes
"""

import json
import argparse
from pathlib import Path


def extract_image_paths(jsonl_path, output_txt_path=None):
    """
    Extract image paths from JSONL test set.
    
    Args:
        jsonl_path: Path to test set JSONL file
        output_txt_path: Optional path to save image paths as text file
        
    Returns:
        List of image paths
    """
    image_paths = []
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                example = json.loads(line)
                # Extract image path from first message (user message)
                user_msg = example['messages'][0]
                for content_item in user_msg['content']:
                    if content_item.get('type') == 'image':
                        image_path = content_item['image']
                        image_paths.append(image_path)
                        break
    
    # Remove duplicates while preserving order
    seen = set()
    unique_paths = []
    for path in image_paths:
        if path not in seen:
            seen.add(path)
            unique_paths.append(path)
    
    # Optionally save to text file
    if output_txt_path:
        with open(output_txt_path, 'w') as f:
            for path in unique_paths:
                f.write(path + '\n')
    
    return unique_paths


def main():
    parser = argparse.ArgumentParser(description="Extract test image paths from JSONL")
    
    parser.add_argument(
        "--test_jsonl",
        type=str,
        default="qwen_misclassified_sft_test.jsonl",
        help="Test set JSONL file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="test_images.txt",
        help="Output text file with image paths"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("📂 EXTRACTING TEST IMAGE PATHS")
    print("="*80)
    
    print(f"\n📂 Reading test set from {args.test_jsonl}...")
    image_paths = extract_image_paths(args.test_jsonl, args.output)
    
    print(f"   ✓ Found {len(image_paths)} unique test images")
    print(f"   ✓ Saved to {args.output}")
    
    print("\n📋 Test image paths:")
    for i, path in enumerate(image_paths, 1):
        filename = Path(path).name
        print(f"   {i:2d}. {filename}")
    
    print("\n" + "="*80)
    print("✅ EXTRACTION COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()

