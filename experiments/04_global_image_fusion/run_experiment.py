"""Global Image Fusion Patching Experiment (Decoder Level).

Patches all image-token hidden states from a source image into a target
image (distractor) at the decoder level.  The model is asked to identify the
object: if patching succeeds, it reports the *source* object instead of the
distractor.  The layer grid reveals the depth at which visual override occurs.

Usage::

    python -m experiments.04_global_image_fusion.run_experiment \
        --model qwen --device cuda:0 --output results/qwen_global_img.json
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
    parser = argparse.ArgumentParser(description="Global image fusion patching")
    parser.add_argument("--model", required=True, choices=["llava", "deepseek", "qwen", "internvl", "llava_onevision"])
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--distractor_image", type=str, default=None,
                        help="Path to distractor image (defaults to solid-colour image)")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--layer_step", type=int, default=3)
    parser.add_argument("--max_layers", type=int, default=28)
    args = parser.parse_args()

    defaults = {"llava": "liuhaotian/llava-v1.5-7b",
                "deepseek": "deepseek-ai/deepseek-vl2-tiny",
                "qwen": "Qwen/Qwen2-VL-7B-Instruct",
                "internvl": "OpenGVLab/InternVL3.5-8B",
                "llava_onevision": "llava-hf/llava-onevision-qwen2-7b-ov-hf"}
    model_path = args.model_path or defaults[args.model]

    adapter = get_adapter(args.model)
    mt = adapter.load_model(model_path, args.device)

    data_dir = Path(args.data_dir)
    sample_trainset = load_from_disk(str(data_dir / "full_sample"))
    with open(data_dir / "instances_category_map.json") as f:
        category_mapping = json.load(f)

    if args.distractor_image:
        distractor = Image.open(args.distractor_image).convert("RGB")
    else:
        distractor = Image.new("RGB", (224, 224), (128, 128, 128))

    # Resume from checkpoint if output file already exists
    if Path(args.output).exists():
        with open(args.output) as f:
            exp_results = json.load(f)
        completed_ids = {r["sample_id"] for r in exp_results}
        print(f"Resuming from checkpoint: {len(exp_results)} samples already done")
    else:
        exp_results = []
        completed_ids = set()

    for count, sample in enumerate(tqdm(sample_trainset)):
        if count in completed_ids:
            continue
        category = category_mapping[str(sample["annotations"]["category_id"][0])]
        prompt = f"What is in the image? {category} or apple"

        if isinstance(sample["image"], Image.Image):
            sample["image"] = sample["image"].convert("RGB")

        inp_source = adapter.prepare_inputs(prompt, sample["image"], mt)
        inp_target = adapter.prepare_inputs(prompt, distractor, mt)

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
                        patch_fn, max_new_tokens=20,
                    )
                    prediction = 1 if category in output.lower() else 0
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
