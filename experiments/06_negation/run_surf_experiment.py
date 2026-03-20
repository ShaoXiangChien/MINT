"""SURF Negation Patching Experiment (Decoder Level).

Patches image-token hidden states to test whether visual evidence can
override a model's tendency to mishandle negation.  Uses the SURF
(negation benchmark) dataset of images where models commonly fail to
understand negated object presence.

Usage::

    python -m experiments.06_negation.run_surf_experiment \
        --model qwen --device cuda:0 \
        --test_images_dir data/surf/test_images \
        --output results/qwen_negation.json
"""

import argparse
import glob
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
    parser = argparse.ArgumentParser(description="SURF negation patching")
    parser.add_argument("--model", required=True, choices=["llava", "deepseek", "qwen", "llava_onevision"])
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--test_images_dir", type=str, required=True,
                        help="Directory with test images (one per object category)")
    parser.add_argument("--neg_object_map", type=str, default=None,
                        help="Path to neg_object_map.json mapping image -> object name")
    parser.add_argument("--blank_image", type=str, default=None,
                        help="Path to blank reference image")
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

    # Load test images and object mapping
    image_paths = sorted(glob.glob(str(Path(args.test_images_dir) / "*.jpg")) +
                         glob.glob(str(Path(args.test_images_dir) / "*.png")))

    if args.neg_object_map:
        with open(args.neg_object_map) as f:
            object_map = json.load(f)
    else:
        object_map = {}

    if args.blank_image:
        blank = Image.open(args.blank_image).convert("RGB")
    else:
        blank = Image.new("RGB", (384, 384), (0, 0, 0))

    exp_results = []

    for idx, img_path in enumerate(tqdm(image_paths)):
        img_name = Path(img_path).stem
        source_image = Image.open(img_path).convert("RGB")
        obj_name = object_map.get(img_name, img_name.split("_")[0])

        prompt = f"Is there a {obj_name} in the image? Answer yes or no."

        inp_source = adapter.prepare_inputs(prompt, source_image, mt)
        inp_target = adapter.prepare_inputs(prompt, blank.resize(source_image.size), mt)

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
                    layer_results.append(prediction)
                results.append(layer_results)
        except Exception as e:
            print(f"Error at image {idx}: {e}")
            continue

        if results:
            exp_results.append({
                "image_id": idx,
                "image_name": img_name,
                "object": obj_name,
                "results": results,
            })

    print(f"Completed: {len(exp_results)} images")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(exp_results, f, indent=2)


if __name__ == "__main__":
    main()
