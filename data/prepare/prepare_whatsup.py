"""What's Up Dataset Preparation for MINT Baseline Fusion Experiment.

Reads the ``controlled_images_dataset.json`` from the What's Up dataset
(Kamath et al., EMNLP 2023) and reformats it into MINT's unified minimal-pair
format.

Expected directory layout after downloading from Google Drive:
    <root_dir>/
        controlled_images_dataset.json   ← JSON index (86 KB)
        controlled_images/               ← extracted from controlled_images.tar.gz
            left_of_xxx.jpg
            right_of_xxx.jpg
            ...

The JSON contains items with the following structure:
    {
        "image_path": "controlled_images/left_of_xxx.jpg",   ← relative path
        "caption_options": ["correct caption", "wrong caption 1", ...]
    }
The FIRST caption_option is always the correct one (per the dataset README).

We convert each item into a MINT minimal pair:
    - positive: question from caption_options[0] → label: "yes"
    - negative: question from caption_options[1] → label: "no"

Output format (JSON):
    [
      {
        "image_file": "/absolute/path/to/controlled_images/left_of_xxx.jpg",
        "image_id":   0,
        "relation":   "left_of",
        "positive": {
            "question": "Is a cup to the left of a plate?",
            "label":    "yes"
        },
        "negative": {
            "question": "Is a plate to the left of a cup?",
            "label":    "no"
        }
      },
      ...
    ]

Usage::

    python data/prepare/prepare_whatsup.py \\
        --root_dir   /path/to/whatsup_data \\
        --output_dir data/whatsup

Where ``root_dir`` is the directory that contains both
``controlled_images_dataset.json`` and the ``controlled_images/`` folder.
"""

import argparse
import json
import re
from pathlib import Path


def caption_to_question(caption: str) -> str:
    """Convert a spatial caption into a yes/no question.

    Example:
        "A cup to the left of a plate" → "Is a cup to the left of a plate?"
    """
    caption = caption.strip().rstrip(".")
    if not caption:
        return ""
    # Lower-case the first letter so "Is A cup..." doesn't happen
    first_lower = caption[0].lower() + caption[1:]
    return f"Is {first_lower}?"


def infer_relation(image_path: str) -> str:
    """Infer the spatial relation from the image filename."""
    fname = Path(image_path).name
    if "left_of" in fname:
        return "left_of"
    elif "right_of" in fname:
        return "right_of"
    elif "_on_" in fname:
        return "on"
    elif "under" in fname:
        return "under"
    return "unknown"


def prepare_whatsup(root_dir: str, output_dir: str) -> None:
    root = Path(root_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    annotation_file = root / "controlled_images_dataset.json"
    if not annotation_file.exists():
        raise FileNotFoundError(
            f"Could not find {annotation_file}.\n"
            "Please download 'controlled_images_dataset.json' from the What's Up "
            "Google Drive and place it in the root_dir."
        )

    with open(annotation_file) as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} items from {annotation_file}")

    samples = []
    skipped = 0

    for idx, item in enumerate(dataset):
        # Resolve absolute image path
        rel_path = item.get("image_path", "")
        abs_path = root / rel_path

        caption_options = item.get("caption_options", [])
        if len(caption_options) < 2:
            skipped += 1
            continue

        if not abs_path.exists():
            # Store path anyway; user can verify after extraction
            print(f"  WARNING: image not found: {abs_path}")

        correct_caption   = caption_options[0]   # always correct per README
        incorrect_caption = caption_options[1]   # first distractor

        samples.append({
            "image_file": str(abs_path),
            "image_id":   idx,
            "relation":   infer_relation(rel_path),
            "positive": {
                "question": caption_to_question(correct_caption),
                "label":    "yes",
            },
            "negative": {
                "question": caption_to_question(incorrect_caption),
                "label":    "no",
            },
        })

    print(f"  Converted {len(samples)} minimal pairs ({skipped} skipped).")

    # Print relation distribution
    from collections import Counter
    rel_counts = Counter(s["relation"] for s in samples)
    print("  Relation distribution:")
    for rel, count in sorted(rel_counts.items()):
        print(f"    {rel}: {count}")

    output_file = out_path / "whatsup_spatial_minimal_pairs.json"
    with open(output_file, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"\nOutput: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare What's Up Controlled_Images dataset for MINT")
    parser.add_argument(
        "--root_dir", required=True,
        help=(
            "Directory containing 'controlled_images_dataset.json' and the "
            "'controlled_images/' folder. Download both from the What's Up "
            "Google Drive: https://drive.google.com/drive/folders/164q6X9hrvP-QYpi3ioSnfMuyHpG5oRkZ"
        ),
    )
    parser.add_argument(
        "--output_dir", default="data/whatsup",
        help="Directory to save the prepared dataset (default: data/whatsup)",
    )
    args = parser.parse_args()

    print("=== Preparing What's Up Controlled_Images dataset ===")
    prepare_whatsup(args.root_dir, args.output_dir)


if __name__ == "__main__":
    main()
