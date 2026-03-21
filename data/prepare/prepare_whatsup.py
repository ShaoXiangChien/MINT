"""What's Up Dataset Preparation for MINT Baseline Fusion Experiment.

Downloads the "Controlled_Images" subset of the What's Up dataset
(Kamath et al., EMNLP 2023) and reformats it into MINT's unified
minimal-pair format.

The Controlled_Images subset contains real photographs with controlled
spatial relationships (e.g. "A cup to the left of a plate").  Each item
provides two caption options:
  - caption_options[0]: the CORRECT spatial description
  - caption_options[1]: the INCORRECT (flipped) spatial description

We convert each item into two MINT samples:
  - Positive: question from correct caption → label: "yes"
  - Negative: question from incorrect caption → label: "no"

Output format (JSON):
    [
      {
        "image_file": "/path/to/image.jpg",
        "image_id":   42,
        "subset":     "A",
        "positive": {
            "question": "Is the cup to the left of the plate?",
            "label":    "yes"
        },
        "negative": {
            "question": "Is the plate to the left of the cup?",
            "label":    "no"
        }
      },
      ...
    ]

Usage::

    python data/prepare/prepare_whatsup.py \
        --data_dir   /path/to/whatsup_vlms/data \
        --output_dir data/whatsup \
        --subset     A
"""

import argparse
import json
import urllib.request
from pathlib import Path


# Google Drive direct download links for the What's Up Controlled_Images data.
# These are the JSON index files; images are downloaded separately.
WHATSUP_JSON_URLS = {
    "A": "https://raw.githubusercontent.com/amitakamath/whatsup_vlms/main/dataset_zoo/controlled_images_A.json",
    "B": "https://raw.githubusercontent.com/amitakamath/whatsup_vlms/main/dataset_zoo/controlled_images_B.json",
}


def caption_to_question(caption: str) -> str:
    """Convert a spatial caption into a yes/no question.

    Example:
        "A cup to the left of a plate" → "Is a cup to the left of a plate?"
    """
    caption = caption.strip().rstrip(".")
    # Capitalise first letter
    if caption:
        caption = caption[0].upper() + caption[1:]
    return f"Is {caption[0].lower()}{caption[1:]}?"


def prepare_whatsup(data_dir: str, output_dir: str, subsets: list) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    data_path = Path(data_dir)

    all_samples = []

    for subset in subsets:
        # Try to load from local data_dir first, then fall back to GitHub
        local_json = data_path / f"controlled_images_{subset}.json"
        if not local_json.exists():
            url = WHATSUP_JSON_URLS.get(subset)
            if url is None:
                print(f"  [Subset {subset}] No URL available, skipping.")
                continue
            print(f"  [Subset {subset}] Downloading index JSON from {url} ...")
            try:
                urllib.request.urlretrieve(url, local_json)
            except Exception as e:
                print(f"  [Subset {subset}] Download failed: {e}")
                print(f"  Please download the What's Up dataset manually from:")
                print(f"  https://github.com/amitakamath/whatsup_vlms")
                print(f"  and place the JSON files in: {data_path}")
                continue

        with open(local_json) as f:
            items = json.load(f)

        skipped = 0
        for idx, item in enumerate(items):
            # The JSON stores image_id (int) and caption_options (list of 2 strings)
            # Image path may be stored as image_id or as a filename
            image_id = item.get("image_id", idx)
            caption_options = item.get("caption_options", [])

            if len(caption_options) < 2:
                skipped += 1
                continue

            # Resolve image path
            # What's Up images are typically stored as:
            #   data/controlled_images/subset_{A|B}/{image_id}.jpg
            candidate_paths = [
                data_path / f"controlled_images/subset_{subset}" / f"{image_id}.jpg",
                data_path / f"controlled_images" / f"{image_id}.jpg",
                data_path / f"{image_id}.jpg",
            ]
            image_file = None
            for p in candidate_paths:
                if p.exists():
                    image_file = str(p)
                    break

            if image_file is None:
                # Store the expected path anyway; user can verify after download
                image_file = str(candidate_paths[0])

            correct_caption   = caption_options[0]
            incorrect_caption = caption_options[1]

            all_samples.append({
                "image_file": image_file,
                "image_id":   image_id,
                "subset":     subset,
                "positive": {
                    "question": caption_to_question(correct_caption),
                    "label":    "yes",
                },
                "negative": {
                    "question": caption_to_question(incorrect_caption),
                    "label":    "no",
                },
            })

        print(f"  [Subset {subset}] {len(items) - skipped} samples loaded "
              f"({skipped} skipped).")

    output_file = out_path / "whatsup_spatial_minimal_pairs.json"
    with open(output_file, "w") as f:
        json.dump(all_samples, f, indent=2)

    print(f"\nTotal samples written: {len(all_samples)}")
    print(f"Output: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare What's Up dataset for MINT")
    parser.add_argument(
        "--data_dir", required=True,
        help="Path to the whatsup_vlms/data directory (or any dir containing "
             "controlled_images_A.json / controlled_images_B.json)",
    )
    parser.add_argument(
        "--output_dir", default="data/whatsup",
        help="Directory to save the prepared dataset (default: data/whatsup)",
    )
    parser.add_argument(
        "--subset", nargs="+", default=["A"],
        choices=["A", "B"],
        help="Which Controlled_Images subset(s) to include (default: A)",
    )
    args = parser.parse_args()

    print("=== Preparing What's Up dataset ===")
    prepare_whatsup(args.data_dir, args.output_dir, args.subset)


if __name__ == "__main__":
    main()
