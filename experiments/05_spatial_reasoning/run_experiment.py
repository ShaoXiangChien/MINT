"""Spatial Reasoning Patching Experiment (Decoder Level).

Patches decoder hidden states for image tokens from a source image
(with a known spatial layout) into a target image (blank), then asks
the model about spatial relationships.  Reveals at which decoder depth
spatial information becomes available.

Usage::

    python -m experiments.05_spatial_reasoning.run_experiment \
        --model qwen --device cuda:0 --output results/qwen_spatial.json
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


def main():
    parser = argparse.ArgumentParser(description="Spatial reasoning patching")
    parser.add_argument("--model", required=True, choices=["llava", "deepseek", "qwen", "llava_onevision"])
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to controlled_images_dataset.json")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory containing the controlled spatial images")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--layer_step", type=int, default=3)
    parser.add_argument("--max_layers", type=int, default=28)
    args = parser.parse_args()

    defaults = {"llava": "liuhaotian/llava-v1.5-7b",
                "deepseek": "deepseek-ai/deepseek-vl2-tiny",
                "qwen": "Qwen/Qwen2-VL-7B-Instruct",
                "llava_onevision": "llava-hf/llava-onevision-qwen2-7b-ov-hf"}
    model_path = args.model_path or defaults[args.model]

    adapter = get_adapter(args.model)
    mt = adapter.load_model(model_path, args.device)

    with open(args.dataset) as f:
        data = json.load(f)

    # Resume from checkpoint if output file already exists
    # Note: exp_results for exp05 uses image_id (not sample_id) as the key
    if Path(args.output).exists():
        with open(args.output) as f:
            exp_results = json.load(f)
        completed_ids = {r["image_id"] for r in exp_results}
        print(f"Resuming from checkpoint: {len(completed_ids)} images already done")
    else:
        exp_results = []
        completed_ids = set()

    for count, item in enumerate(tqdm(data)):
        if count in completed_ids:
            continue
        source_image = Image.open(Path(args.image_dir) / item["image_path"]).convert("RGB")
        target_image = Image.new("RGB", source_image.size, (0, 0, 0))

        for caption_idx, caption in enumerate(item["caption_options"]):
            prompt = f"Is {caption}? only answer YES or NO"

            inp_source = adapter.prepare_inputs(prompt, source_image, mt)
            inp_target = adapter.prepare_inputs(prompt, target_image, mt)

            try:
                img_start, img_end = adapter.find_image_token_range(mt, inp_source)
            except ValueError:
                continue

            patch_fn = make_image_token_patch_fn(img_start, img_end)
            results = []

            try:
                for ls in range(0, args.max_layers, args.layer_step):
                    layer_results = []
                    cached_hs = capture_decoder_hs(adapter, mt, inp_source, ls)
                    for lt in range(0, args.max_layers, args.layer_step):
                        output = patch_decoder_and_generate(
                            adapter, mt, inp_target, lt, cached_hs,
                            patch_fn, max_new_tokens=5,
                        )
                        prediction = 1 if "yes" in output.lower() else 0
                        correct = (prediction == 1) if caption_idx == 0 else (prediction == 0)
                        layer_results.append({
                            "layer_source": ls,
                            "layer_target": lt,
                            "result": output,
                            "correct": correct,
                            "prediction": prediction,
                        })
                    results.append(layer_results)
            except Exception as e:
                print(f"Error at sample {count}, caption {caption_idx}: {e}")
                continue

            exp_results.append({
                "image_id": count,
                "label": "yes" if caption_idx == 0 else "no",
                "results": results,
            })

        if count % 50 == 0:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(exp_results, f, indent=2)

    print(f"Completed: {len(exp_results)} entries")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(exp_results, f, indent=2)


if __name__ == "__main__":
    main()
