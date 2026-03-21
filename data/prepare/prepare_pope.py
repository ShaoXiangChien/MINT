"""POPE Dataset Preparation for MINT Baseline Fusion Experiment.

Downloads the POPE adversarial/popular/random splits from the official
RUCAIBox/POPE repository and reformats them into MINT's unified minimal-pair
format.  Only the "random" split is used by default, as it provides the
least-biased binary object-existence questions.

Output format (JSON Lines, one object per line):
    {
        "image_file": "COCO_val2014_000000016631.jpg",
        "question":   "Is there a person in the image?",
        "label":      "yes",   # or "no"
        "object":     "person",
        "split":      "random"
    }

Usage::

    python data/prepare/prepare_pope.py \
        --coco_image_dir /path/to/coco/val2014 \
        --output_dir     data/pope \
        --split          random
"""

import argparse
import json
import os
import re
import urllib.request
from pathlib import Path


# Official POPE data files hosted on GitHub (RUCAIBox/POPE)
POPE_URLS = {
    "random":     "https://raw.githubusercontent.com/AoiDragon/POPE/main/output/coco/coco_pope_random.json",
    "popular":    "https://raw.githubusercontent.com/AoiDragon/POPE/main/output/coco/coco_pope_popular.json",
    "adversarial":"https://raw.githubusercontent.com/AoiDragon/POPE/main/output/coco/coco_pope_adversarial.json",
}


def download_pope_split(split: str, output_dir: Path) -> Path:
    """Download a POPE split JSON file if not already present."""
    url = POPE_URLS[split]
    dest = output_dir / f"coco_pope_{split}.json"
    if dest.exists():
        print(f"  [{split}] already downloaded: {dest}")
        return dest
    print(f"  [{split}] downloading from {url} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"  [{split}] saved to {dest}")
    return dest


def extract_object_from_question(question: str) -> str:
    """Extract the object name from a POPE question string.

    POPE questions follow the template "Is there a/an [object] in the image?"
    """
    match = re.search(r"Is there an? (.+?) in the image", question, re.IGNORECASE)
    return match.group(1) if match else "unknown"


def prepare_pope(coco_image_dir: str, output_dir: str, splits: list) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    coco_dir = Path(coco_image_dir)

    all_samples = []
    for split in splits:
        raw_path = download_pope_split(split, out_path)

        with open(raw_path) as f:
            # POPE files are JSON Lines (one JSON object per line)
            lines = [l.strip() for l in f if l.strip()]

        skipped = 0
        for line in lines:
            item = json.loads(line)
            image_file = item["image"]
            full_path = coco_dir / image_file

            # Verify the image exists so we don't create broken entries
            if not full_path.exists():
                skipped += 1
                continue

            all_samples.append({
                "image_file":  str(full_path),
                "question":    item["text"],
                "label":       item["label"],   # "yes" or "no"
                "object":      extract_object_from_question(item["text"]),
                "split":       split,
            })

        print(f"  [{split}] loaded {len(lines) - skipped} samples "
              f"({skipped} skipped – image not found)")

    output_file = out_path / "pope_minimal_pairs.json"
    with open(output_file, "w") as f:
        json.dump(all_samples, f, indent=2)

    print(f"\nTotal samples written: {len(all_samples)}")
    print(f"Output: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Prepare POPE dataset for MINT")
    parser.add_argument(
        "--coco_image_dir", required=True,
        help="Path to the COCO val2014 image directory",
    )
    parser.add_argument(
        "--output_dir", default="data/pope",
        help="Directory to save the prepared dataset (default: data/pope)",
    )
    parser.add_argument(
        "--split", nargs="+",
        default=["random"],
        choices=["random", "popular", "adversarial"],
        help="POPE split(s) to include (default: random)",
    )
    args = parser.parse_args()

    print("=== Preparing POPE dataset ===")
    prepare_pope(args.coco_image_dir, args.output_dir, args.split)


if __name__ == "__main__":
    main()
