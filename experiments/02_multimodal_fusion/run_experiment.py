"""Multimodal Fusion Patching Experiment (Decoder Level).

Patches all image-token hidden states from a source forward pass (real
image + negation text) into a target forward pass (blank image + question)
at the decoder level.  Sweeps ``(source_layer, target_layer)`` to locate
the *fusion band* where visual evidence overrides linguistic priors.

Usage::

    python -m experiments.02_multimodal_fusion.run_experiment \
        --model qwen --device cuda:0 --output results/qwen_mm_fusion.json
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from datasets import load_from_disk
from tqdm import tqdm

from src.models import get_adapter
from src.patching.decoder_patching import (
    capture_decoder_hs,
    patch_decoder_and_generate,
    make_image_token_patch_fn,
)


def main():
    parser = argparse.ArgumentParser(description="Multimodal fusion patching")
    parser.add_argument("--model", required=True, choices=["llava", "deepseek", "qwen"])
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--layer_step", type=int, default=2)
    parser.add_argument("--max_layers", type=int, default=28)
    args = parser.parse_args()

    defaults = {"llava": "liuhaotian/llava-v1.5-7b",
                "deepseek": "deepseek-ai/deepseek-vl2-tiny",
                "qwen": "Qwen/Qwen2-VL-7B-Instruct"}
    model_path = args.model_path or defaults[args.model]

    adapter = get_adapter(args.model)
    mt = adapter.load_model(model_path, args.device)

    data_dir = Path(args.data_dir)
    sample_trainset = load_from_disk(str(data_dir / "full_sample"))
    with open(data_dir / "instances_category_map.json") as f:
        category_mapping = json.load(f)

    exp_results = []

    for count, sample in enumerate(tqdm(sample_trainset)):
        category = category_mapping[str(sample["annotations"]["category_id"][0])]
        source_prompt = f"The main object is not a {category}"
        target_prompt = f"Is the main object in the image a {category}? only answer with yes or no"

        if isinstance(sample["image"], Image.Image):
            sample["image"] = sample["image"].convert("RGB")
        blank_image = Image.new("RGB", sample["image"].size, (0, 0, 0))

        inp_source = adapter.prepare_inputs(source_prompt, sample["image"], mt)
        inp_target = adapter.prepare_inputs(target_prompt, blank_image, mt)

        # Find image token range for patching
        try:
            img_start, img_end = adapter.find_image_token_range(mt, inp_source)
        except ValueError:
            print(f"Skipping sample {count}: no image tokens")
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
                        patch_fn, max_new_tokens=20,
                    )
                    prediction = 1 if "yes" in output.lower() else 0
                    layer_results.append(prediction)
                results.append(layer_results)
        except Exception as e:
            print(f"Error processing sample {count}: {e}")
            continue

        if results:
            exp_results.append({
                "sample_id": count,
                "category": category,
                "results": results,
            })

        if count % 100 == 0:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(exp_results, f, indent=2)

    print(f"Completed: {len(exp_results)} samples")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(exp_results, f, indent=2)


if __name__ == "__main__":
    main()
