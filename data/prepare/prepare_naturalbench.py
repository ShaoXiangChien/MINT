"""NaturalBench Dataset Preparation for MINT Pathology Diagnosis.

Downloads and formats the NaturalBench dataset from HuggingFace.
We focus on the adversarial pairs where language priors contradict the visual evidence.
The script formats the dataset into a flat JSON list containing:
    {"image_file": ..., "target_question": ..., "expected_answer": ...}

Usage::

    python data/prepare/prepare_naturalbench.py \
        --output_dir data/naturalbench
"""

import argparse
import json
import os
from pathlib import Path
from datasets import load_dataset
from PIL import Image

def prepare_naturalbench(output_dir: str, max_samples: int = None) -> None:
    out_path = Path(output_dir)
    img_dir = out_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading NaturalBench dataset from HuggingFace...")
    dataset = load_dataset("BaiqiL/NaturalBench", split="train")
    
    all_samples = []
    
    # NaturalBench has pairs of images and questions.
    # Image_0 + Question_0 -> Image_0_Question_0 (Answer)
    # Image_1 + Question_0 -> Image_1_Question_0 (Answer)
    # We will flatten this structure.
    
    for i, item in enumerate(dataset):
        if max_samples and i >= max_samples:
            break
            
        # We only use yes/no questions for simpler evaluation in patching
        if item.get("Question Type", "").lower() != "yes or no":
            continue
            
        idx = item["Index"]
        
        # Save images
        img0_path = img_dir / f"{idx}_0.jpg"
        img1_path = img_dir / f"{idx}_1.jpg"
        
        if not img0_path.exists():
            item["Image_0"].convert("RGB").save(img0_path)
        if not img1_path.exists():
            item["Image_1"].convert("RGB").save(img1_path)
            
        # Question 0 with Image 0
        all_samples.append({
            "image_file": str(img0_path.absolute()),
            "target_question": item["Question_0"],
            "expected_answer": item["Image_0_Question_0"],
            "meta": {"index": idx, "image_idx": 0, "question_idx": 0}
        })
        
        # Question 0 with Image 1 (Adversarial pair)
        all_samples.append({
            "image_file": str(img1_path.absolute()),
            "target_question": item["Question_0"],
            "expected_answer": item["Image_1_Question_0"],
            "meta": {"index": idx, "image_idx": 1, "question_idx": 0}
        })
        
        # Question 1 with Image 0
        all_samples.append({
            "image_file": str(img0_path.absolute()),
            "target_question": item["Question_1"],
            "expected_answer": item["Image_0_Question_1"],
            "meta": {"index": idx, "image_idx": 0, "question_idx": 1}
        })
        
        # Question 1 with Image 1
        all_samples.append({
            "image_file": str(img1_path.absolute()),
            "target_question": item["Question_1"],
            "expected_answer": item["Image_1_Question_1"],
            "meta": {"index": idx, "image_idx": 1, "question_idx": 1}
        })
        
        if i % 100 == 0:
            print(f"Processed {i} samples...")

    output_file = out_path / "naturalbench_pathology.json"
    with open(output_file, "w") as f:
        json.dump(all_samples, f, indent=2)

    print(f"\nTotal samples written: {len(all_samples)}")
    print(f"Output: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Prepare NaturalBench dataset for MINT")
    parser.add_argument(
        "--output_dir", default="data/naturalbench",
        help="Directory to save the prepared dataset and images (default: data/naturalbench)",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Maximum number of original dataset rows to process",
    )
    args = parser.parse_args()

    print("=== Preparing NaturalBench dataset ===")
    prepare_naturalbench(args.output_dir, args.max_samples)


if __name__ == "__main__":
    main()
