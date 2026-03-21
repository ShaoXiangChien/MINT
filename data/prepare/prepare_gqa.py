"""GQA Attribute Minimal-Pair Preparation for MINT Baseline Fusion Experiment.

Parses the GQA validation scene graphs and question files to extract a clean
set of binary attribute questions (colour and material) that form perfect
minimal pairs: two questions about the same object in the same image, where
one answer is correct and the other is a plausible distractor.

For each object with a known attribute (e.g. "red apple"), we generate:
  - Positive: "Is the apple red?"   → label: "yes"
  - Negative: "Is the apple green?" → label: "no"

The negative attribute is sampled from the same attribute category (colour or
material) so the distractor is visually plausible.

Output format (JSON):
    [
      {
        "image_file":  "/path/to/gqa/images/1234.jpg",
        "image_id":    "1234",
        "object_name": "apple",
        "attribute":   "red",
        "distractor":  "green",
        "positive": {
            "question": "Is the apple red?",
            "label":    "yes"
        },
        "negative": {
            "question": "Is the apple green?",
            "label":    "no"
        }
      },
      ...
    ]

Usage::

    python data/prepare/prepare_gqa.py \
        --scene_graphs  /path/to/gqa/val_sceneGraphs.json \
        --gqa_image_dir /path/to/gqa/images \
        --output_dir    data/gqa \
        --max_samples   500
"""

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict


# Colour and material attribute pools for distractor sampling.
# These are the most common GQA attribute values in each category.
COLOUR_POOL = [
    "red", "green", "blue", "yellow", "orange", "purple",
    "pink", "brown", "black", "white", "gray", "silver",
]
MATERIAL_POOL = [
    "wooden", "metal", "plastic", "glass", "stone",
    "fabric", "rubber", "ceramic", "paper", "leather",
]

# Attribute categories we care about
ATTRIBUTE_CATEGORIES = {
    "colour":   set(COLOUR_POOL),
    "material": set(MATERIAL_POOL),
}


def classify_attribute(attr: str):
    """Return (category, normalised_value) or None if not a target attribute."""
    attr_lower = attr.lower().strip()
    for category, pool in ATTRIBUTE_CATEGORIES.items():
        if attr_lower in pool:
            return category, attr_lower
    return None


def sample_distractor(category: str, true_attr: str) -> str:
    """Sample a plausible distractor from the same attribute category."""
    if category == "colour":
        pool = [c for c in COLOUR_POOL if c != true_attr]
    else:
        pool = [m for m in MATERIAL_POOL if m != true_attr]
    return random.choice(pool)


def prepare_gqa(scene_graphs_path: str, gqa_image_dir: str,
                output_dir: str, max_samples: int, seed: int) -> None:
    random.seed(seed)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    img_dir = Path(gqa_image_dir)

    print(f"Loading scene graphs from {scene_graphs_path} ...")
    with open(scene_graphs_path) as f:
        scene_graphs = json.load(f)
    print(f"  Loaded {len(scene_graphs)} images.")

    samples = []
    seen_pairs = set()  # deduplicate (image_id, object_name, attribute)

    for image_id, sg in scene_graphs.items():
        image_file = img_dir / f"{image_id}.jpg"
        if not image_file.exists():
            continue

        for obj_id, obj in sg.get("objects", {}).items():
            obj_name = obj.get("name", "").strip()
            if not obj_name:
                continue

            for attr in obj.get("attributes", []):
                result = classify_attribute(attr)
                if result is None:
                    continue
                category, true_attr = result

                key = (image_id, obj_name, true_attr)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)

                distractor = sample_distractor(category, true_attr)
                article = "an" if true_attr[0] in "aeiou" else "a"
                article_d = "an" if distractor[0] in "aeiou" else "a"
                article_obj = "an" if obj_name[0] in "aeiou" else "a"

                samples.append({
                    "image_file":  str(image_file),
                    "image_id":    image_id,
                    "object_name": obj_name,
                    "attribute":   true_attr,
                    "distractor":  distractor,
                    "category":    category,
                    "positive": {
                        "question": f"Is the {obj_name} {true_attr}?",
                        "label":    "yes",
                    },
                    "negative": {
                        "question": f"Is the {obj_name} {distractor}?",
                        "label":    "no",
                    },
                })

    print(f"  Found {len(samples)} candidate minimal pairs before sampling.")

    # Shuffle and cap
    random.shuffle(samples)
    samples = samples[:max_samples]

    output_file = out_path / "gqa_attribute_minimal_pairs.json"
    with open(output_file, "w") as f:
        json.dump(samples, f, indent=2)

    # Print attribute distribution
    cat_counts = defaultdict(int)
    for s in samples:
        cat_counts[s["category"]] += 1
    print(f"\nAttribute distribution in final {len(samples)} samples:")
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count}")
    print(f"\nOutput: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare GQA attribute minimal pairs for MINT")
    parser.add_argument(
        "--scene_graphs", required=True,
        help="Path to GQA val_sceneGraphs.json",
    )
    parser.add_argument(
        "--gqa_image_dir", required=True,
        help="Directory containing GQA images (*.jpg)",
    )
    parser.add_argument(
        "--output_dir", default="data/gqa",
        help="Directory to save the prepared dataset (default: data/gqa)",
    )
    parser.add_argument(
        "--max_samples", type=int, default=500,
        help="Maximum number of minimal pairs to include (default: 500)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    print("=== Preparing GQA attribute minimal pairs ===")
    prepare_gqa(
        args.scene_graphs, args.gqa_image_dir,
        args.output_dir, args.max_samples, args.seed,
    )


if __name__ == "__main__":
    main()
