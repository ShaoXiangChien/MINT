"""Baseline Fusion Patching Experiment (Section 5.1 – Healthy Baseline).

This experiment establishes the existence of the Fusion Band as an
architectural constant by running the same causal patching sweep across
three fundamentally different visual dimensions:

  - Object Existence  (POPE random split)
  - Attribute         (GQA colour/material minimal pairs)
  - Spatial           (What's Up Controlled_Images)

For each minimal pair (positive + negative question on the same image),
we perform the standard source-to-target decoder patching sweep and
record the Override Accuracy (OA): the fraction of target-layer positions
at which injecting the source hidden states causes the model to answer
correctly.

If the Fusion Band is an architectural constant, the OA heatmaps across
all three dimensions should show the same critical layer range.

Usage::

    python -m experiments.07_baseline_fusion.run_experiment \\
        --model          qwen \\
        --device         cuda:0 \\
        --pope_data      data/pope/pope_minimal_pairs.json \\
        --gqa_data       data/gqa/gqa_attribute_minimal_pairs.json \\
        --whatsup_data   data/whatsup/whatsup_spatial_minimal_pairs.json \\
        --output         results/baseline_fusion_qwen.json \\
        --max_samples    200

Prepare the datasets first::

    python data/prepare/prepare_pope.py    --coco_image_dir /path/to/coco/val2014
    python data/prepare/prepare_gqa.py     --scene_graphs   /path/to/gqa/val_sceneGraphs.json \\
                                           --gqa_image_dir  /path/to/gqa/images
    python data/prepare/prepare_whatsup.py --data_dir       /path/to/whatsup_vlms/data
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from src.models import get_adapter
from src.patching.decoder_patching import (
    capture_decoder_hs,
    patch_decoder_and_generate,
    make_image_token_patch_fn,
)


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_pope(path: str, max_samples: int):
    """Load POPE minimal pairs.

    prepare_pope.py outputs a flat list of items with keys:
        image_file, question, label ("yes"/"no"), object, split

    POPE is structured so that for each image there is one "yes" question
    ("Is there a <object>?") and one "no" question ("Is there a <other_object>?").
    We group by image_file and pair the yes/no items together.
    """
    with open(path) as f:
        raw = json.load(f)

    # Group by image_file, then pick one yes and one no per image
    from collections import defaultdict
    groups = defaultdict(lambda: {"yes": [], "no": []})
    for item in raw:
        groups[item["image_file"]][item["label"]].append(item)

    pairs = []
    for image_file, labels in groups.items():
        if labels["yes"] and labels["no"]:
            pos = labels["yes"][0]
            neg = labels["no"][0]
            pairs.append({
                "dimension":  "object",
                "image_file": image_file,
                "positive":   {"question": pos["question"], "label": "yes"},
                "negative":   {"question": neg["question"],  "label": "no"},
                "meta":       {"object": pos["object"]},
            })

    return pairs[:max_samples]


def load_gqa(path: str, max_samples: int):
    """Load GQA attribute minimal pairs."""
    with open(path) as f:
        raw = json.load(f)

    pairs = []
    for item in raw:
        pairs.append({
            "dimension":  "attribute",
            "image_file": item["image_file"],
            "positive":   item["positive"],
            "negative":   item["negative"],
            "meta": {
                "object":     item["object_name"],
                "attribute":  item["attribute"],
                "distractor": item["distractor"],
                "category":   item["category"],
            },
        })

    return pairs[:max_samples]


def load_whatsup(path: str, max_samples: int):
    """Load What's Up spatial minimal pairs."""
    with open(path) as f:
        raw = json.load(f)

    pairs = []
    for item in raw:
        pairs.append({
            "dimension":  "spatial",
            "image_file": item["image_file"],
            "positive":   item["positive"],
            "negative":   item["negative"],
            # prepare_whatsup.py outputs image_id and relation (no subset key)
            "meta":       {"image_id": item["image_id"], "relation": item.get("relation", "")},
        })

    return pairs[:max_samples]


# ---------------------------------------------------------------------------
# Core patching logic
# ---------------------------------------------------------------------------

def run_patching_sweep(adapter, mt, source_image, question, max_layers, layer_step):
    """Run the full (source_layer × target_layer) patching sweep for one sample.

    Source pass: real image + question
    Target pass: blank image + question

    Returns a 2-D list of shape [num_source_layers][num_target_layers],
    where each cell is 1 if the model answers "yes" after patching, else 0.
    """
    blank_image = Image.new("RGB", source_image.size, (0, 0, 0))

    inp_source = adapter.prepare_inputs(question, source_image, mt)
    inp_target = adapter.prepare_inputs(question, blank_image, mt)

    img_start, img_end = adapter.find_image_token_range(mt, inp_source)
    patch_fn = make_image_token_patch_fn(img_start, img_end)

    results = []
    for ls in range(0, max_layers, layer_step):
        cached_hs = capture_decoder_hs(adapter, mt, inp_source, ls)
        row = []
        for lt in range(0, max_layers, layer_step):
            output = patch_decoder_and_generate(
                adapter, mt, inp_target, lt, cached_hs,
                patch_fn, max_new_tokens=5,
            )
            row.append(1 if "yes" in output.lower() else 0)
        results.append(row)

    return results


def process_pair(adapter, mt, pair, max_layers, layer_step):
    """Process one minimal pair and return the result dict."""
    image_file = pair["image_file"]
    source_image = Image.open(image_file).convert("RGB")

    # Positive pass: the correct question (expected answer: "yes")
    pos_results = run_patching_sweep(
        adapter, mt, source_image,
        pair["positive"]["question"],
        max_layers, layer_step,
    )

    # Negative pass: the incorrect question (expected answer: "no")
    # Here we measure whether patching *correctly keeps* the answer as "no"
    # by checking that the model does NOT say "yes" after patching.
    neg_results = run_patching_sweep(
        adapter, mt, source_image,
        pair["negative"]["question"],
        max_layers, layer_step,
    )

    return {
        "dimension":      pair["dimension"],
        "image_file":     image_file,
        "meta":           pair.get("meta", {}),
        "positive_sweep": pos_results,   # 1 = model says "yes" (correct)
        "negative_sweep": neg_results,   # 1 = model says "yes" (incorrect)
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Baseline fusion patching experiment (Section 5.1)")
    parser.add_argument("--model", required=True,
                        choices=["llava", "deepseek", "qwen", "qwen25", "internvl", "llava_onevision", "internvl25"])
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")

    # Dataset paths
    parser.add_argument("--pope_data",    type=str, default=None,
                        help="Path to pope_minimal_pairs.json")
    parser.add_argument("--gqa_data",     type=str, default=None,
                        help="Path to gqa_attribute_minimal_pairs.json")
    parser.add_argument("--whatsup_data", type=str, default=None,
                        help="Path to whatsup_spatial_minimal_pairs.json")

    parser.add_argument("--output",       type=str, required=True)
    parser.add_argument("--max_samples",  type=int, default=200,
                        help="Max samples per dimension (default: 200)")
    parser.add_argument("--layer_step",   type=int, default=2)
    parser.add_argument("--max_layers",   type=int, default=28)
    parser.add_argument("--test", action="store_true",
                        help="Test mode: run only 1 pair per dimension with "
                             "layer_step=4 to verify the pipeline quickly")
    args = parser.parse_args()

    if args.test:
        args.max_samples = 1
        args.layer_step  = 4
        print("[TEST MODE] Running 1 pair per dimension with layer_step=4")

    defaults = {
        "llava":           "liuhaotian/llava-v1.5-7b",
        "deepseek":        "deepseek-ai/deepseek-vl2-tiny",
        "qwen":            "Qwen/Qwen2-VL-7B-Instruct",
        "internvl":        "OpenGVLab/InternVL2-8B",
        "llava_onevision": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        "qwen25":           "Qwen/Qwen2.5-VL-7B-Instruct",
        "internvl25":          "OpenGVLab/InternVL2_5-8B",
    }
    model_path = args.model_path or defaults[args.model]

    adapter = get_adapter(args.model)
    mt = adapter.load_model(model_path, args.device)

    # Load datasets
    all_pairs = []
    if args.pope_data:
        pope_pairs = load_pope(args.pope_data, args.max_samples)
        print(f"POPE:    {len(pope_pairs)} pairs")
        all_pairs.extend(pope_pairs)
    if args.gqa_data:
        gqa_pairs = load_gqa(args.gqa_data, args.max_samples)
        print(f"GQA:     {len(gqa_pairs)} pairs")
        all_pairs.extend(gqa_pairs)
    if args.whatsup_data:
        wu_pairs = load_whatsup(args.whatsup_data, args.max_samples)
        print(f"What'sUp:{len(wu_pairs)} pairs")
        all_pairs.extend(wu_pairs)

    if not all_pairs:
        print("ERROR: No dataset paths provided. Use --pope_data, --gqa_data, "
              "and/or --whatsup_data.")
        return

    # Resume from checkpoint
    if Path(args.output).exists():
        with open(args.output) as f:
            exp_results = json.load(f)
        completed = len(exp_results)
        print(f"Resuming from checkpoint: {completed} pairs already done")
    else:
        exp_results = []
        completed = 0

    for idx, pair in enumerate(tqdm(all_pairs)):
        if idx < completed:
            continue

        try:
            result = process_pair(adapter, mt, pair, args.max_layers, args.layer_step)
            result["pair_id"] = idx
            exp_results.append(result)
        except Exception as e:
            print(f"Error at pair {idx} ({pair['dimension']}): {e}")
            continue

        if idx % 50 == 0:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(exp_results, f, indent=2)

    print(f"\nCompleted: {len(exp_results)} pairs")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(exp_results, f, indent=2)


if __name__ == "__main__":
    main()
