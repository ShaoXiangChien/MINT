"""Attention Weight Corroboration Experiment (Cross-Metric Validation).

For each minimal pair from the three baseline datasets, this script extracts
the mean text-to-image attention weight at every decoder layer.  The results
are saved alongside the causal patching data so that both signals can be
plotted on the same graph to demonstrate that the Fusion Band identified by
MINT aligns with the model's internal attention hotspot.

This provides the "Cross-Metric Corroboration" evidence described in the
ARR May revision plan.

Usage::

    python -m experiments.07_baseline_fusion.run_attention_corroboration \\
        --model          qwen \\
        --device         cuda:0 \\
        --pope_data      data/pope/pope_minimal_pairs.json \\
        --gqa_data       data/gqa/gqa_attribute_minimal_pairs.json \\
        --whatsup_data   data/whatsup/whatsup_spatial_minimal_pairs.json \\
        --output         results/attention_corroboration_qwen.json \\
        --max_samples    100
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from src.models import get_adapter
from src.patching.attention_extraction import extract_text_to_image_attention
from experiments.07_baseline_fusion.run_experiment import (
    load_pope, load_gqa, load_whatsup,
)


def main():
    parser = argparse.ArgumentParser(
        description="Attention weight corroboration for baseline fusion")
    parser.add_argument("--model", required=True,
                        choices=["llava", "deepseek", "qwen", "llava_onevision"])
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--pope_data",    type=str, default=None)
    parser.add_argument("--gqa_data",     type=str, default=None)
    parser.add_argument("--whatsup_data", type=str, default=None)

    parser.add_argument("--output",      type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=100)
    args = parser.parse_args()

    defaults = {
        "llava":           "liuhaotian/llava-v1.5-7b",
        "deepseek":        "deepseek-ai/deepseek-vl2-tiny",
        "qwen":            "Qwen/Qwen2-VL-7B-Instruct",
        "llava_onevision": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    }
    model_path = args.model_path or defaults[args.model]

    adapter = get_adapter(args.model)
    mt = adapter.load_model(model_path, args.device)

    all_pairs = []
    if args.pope_data:
        all_pairs.extend(load_pope(args.pope_data, args.max_samples))
    if args.gqa_data:
        all_pairs.extend(load_gqa(args.gqa_data, args.max_samples))
    if args.whatsup_data:
        all_pairs.extend(load_whatsup(args.whatsup_data, args.max_samples))

    if not all_pairs:
        print("ERROR: No dataset paths provided.")
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
            image = Image.open(pair["image_file"]).convert("RGB")
            # Use the positive question for attention extraction
            inputs = adapter.prepare_inputs(
                pair["positive"]["question"], image, mt)
            img_start, img_end = adapter.find_image_token_range(mt, inputs)

            attn_weights = extract_text_to_image_attention(
                adapter, mt, inputs, img_start, img_end)

            exp_results.append({
                "pair_id":        idx,
                "dimension":      pair["dimension"],
                "image_file":     pair["image_file"],
                "meta":           pair.get("meta", {}),
                "attn_by_layer":  attn_weights,
                "img_token_count": img_end - img_start,
            })
        except Exception as e:
            print(f"Error at pair {idx}: {e}")
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
