"""MINDCUBE Dataset Preparation for MINT Pathology Diagnosis.

The MINDCUBE dataset contains spatial mental modeling tasks.
This script guides the user to download the dataset and processes the JSONL files
into a flat JSON list containing:
    {"image_file": ..., "target_question": ..., "expected_answer": ...}

Usage::

    python data/prepare/prepare_mindcube.py \
        --data_dir /path/to/mindcube/data \
        --output_dir data/mindcube
"""

import argparse
import json
import os
from pathlib import Path

def prepare_mindcube(data_dir: str, output_dir: str, max_samples: int = None) -> None:
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # We will look for the raw JSONL files
    raw_dir = data_path / "raw"
    if not raw_dir.exists():
        print(f"Error: Could not find 'raw' directory in {data_dir}")
        print("Please download the MindCube dataset first:")
        print("  huggingface-cli download Inevitablevalor/MindCube data.zip --repo-type dataset")
        print("  unzip data.zip")
        return
        
    # Process the tinybench first for quick testing, or train if requested
    jsonl_file = raw_dir / "MindCube_tinybench.jsonl"
    if not jsonl_file.exists():
        jsonl_file = raw_dir / "MindCube.jsonl"
        
    if not jsonl_file.exists():
        print(f"Error: Could not find MindCube JSONL files in {raw_dir}")
        return
        
    print(f"Loading MINDCUBE data from {jsonl_file}...")
    
    all_samples = []
    skipped = 0
    
    with open(jsonl_file, "r") as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        if max_samples and i >= max_samples:
            break
            
        if not line.strip():
            continue
            
        item = json.loads(line)
        
        # In MindCube, images are in other_all_image directory
        # The image path in JSON might be relative to the data root
        # Extract the question and answer
        
        # Since we don't have the exact JSON schema, we handle common formats
        # We assume fields like 'image_path', 'question', 'answer' or similar
        
        # Try to find image path
        img_rel_path = item.get("image_path") or item.get("image") or item.get("img_path")
        if not img_rel_path:
            skipped += 1
            continue
            
        # The path in jsonl might not include 'other_all_image'
        if "other_all_image" not in img_rel_path:
            img_full_path = data_path / "other_all_image" / img_rel_path
        else:
            img_full_path = data_path / img_rel_path
            
        if not img_full_path.exists():
            # Try to search for it if the path structure is different
            img_name = Path(img_rel_path).name
            found = False
            for ext in ['.png', '.jpg', '.jpeg']:
                for root, _, files in os.walk(data_path / "other_all_image"):
                    for file in files:
                        if file == img_name or file == Path(img_name).stem + ext:
                            img_full_path = Path(root) / file
                            found = True
                            break
                    if found: break
                if found: break
                
            if not found:
                skipped += 1
                continue
                
        # Try to find question
        question = item.get("question") or item.get("text") or item.get("prompt")
        if not question:
            # Maybe it's in a nested structure
            conversations = item.get("conversations", [])
            for conv in conversations:
                if conv.get("from") == "human":
                    question = conv.get("value")
                    break
                    
        # Try to find answer
        answer = item.get("answer") or item.get("label") or item.get("ground_truth")
        if not answer:
            conversations = item.get("conversations", [])
            for conv in conversations:
                if conv.get("from") == "gpt" or conv.get("from") == "assistant":
                    answer = conv.get("value")
                    break
                    
        if not question or not answer:
            skipped += 1
            continue
            
        all_samples.append({
            "image_file": str(img_full_path.absolute()),
            "target_question": question,
            "expected_answer": answer,
            "meta": {
                "task_type": item.get("task_type", "unknown"),
                "id": item.get("id", str(i))
            }
        })
        
        if i % 100 == 0:
            print(f"Processed {i} samples...")

    output_file = out_path / "mindcube_pathology.json"
    with open(output_file, "w") as f:
        json.dump(all_samples, f, indent=2)

    print(f"\nTotal samples written: {len(all_samples)} ({skipped} skipped)")
    print(f"Output: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Prepare MINDCUBE dataset for MINT")
    parser.add_argument(
        "--data_dir", required=True,
        help="Path to the extracted MindCube data directory (containing 'raw' and 'other_all_image')",
    )
    parser.add_argument(
        "--output_dir", default="data/mindcube",
        help="Directory to save the prepared dataset (default: data/mindcube)",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Maximum number of original dataset rows to process",
    )
    args = parser.parse_args()

    print("=== Preparing MINDCUBE dataset ===")
    prepare_mindcube(args.data_dir, args.output_dir, args.max_samples)


if __name__ == "__main__":
    main()
