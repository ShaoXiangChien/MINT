"""Object Patching Experiment (Vision Encoder Level).

Patches object-specific embeddings from the vision encoder of a source image
into a target image to test whether the model can be causally steered to
detect the object.  Sweeps a ``(source_layer, target_layer)`` grid over
the vision encoder blocks.

Usage::

    python -m experiments.01_object_patching.run_experiment \
        --model qwen --device cuda:0 --output results/qwen_obj_patching.json
"""

import argparse
import json
import math
from pathlib import Path

import torch
from PIL import Image
from datasets import load_from_disk
from tqdm import tqdm

from src.models import get_adapter
from src.patching.vision_patching import capture_vision_emb, patch_vision_and_generate


def determine_object_patches(bbox, orig_w, orig_h, grid_h, grid_w, patch_size=14):
    """Map a COCO bounding box to vision-encoder patch-token indices.

    Args:
        bbox: COCO-format ``[x, y, w, h]`` in original image pixels.
        orig_w, orig_h: Original image dimensions.
        grid_h, grid_w: Patch grid dimensions from the vision encoder.
        patch_size: Spatial size of each patch (typically 14).

    Returns:
        List of 1-based patch-token indices that overlap the bounding box.
    """
    bx, by, bw, bh = bbox
    resize_w, resize_h = grid_w * patch_size, grid_h * patch_size
    scale_x, scale_y = resize_w / orig_w, resize_h / orig_h

    x1, y1 = bx * scale_x, by * scale_y
    x2, y2 = (bx + bw) * scale_x, (by + bh) * scale_y

    x_start = max(0, min(grid_w - 1, int(math.floor(x1 / patch_size))))
    x_end = max(0, min(grid_w - 1, int(math.floor((x2 - 1) / patch_size))))
    y_start = max(0, min(grid_h - 1, int(math.floor(y1 / patch_size))))
    y_end = max(0, min(grid_h - 1, int(math.floor((y2 - 1) / patch_size))))

    return [
        1 + y * grid_w + x
        for y in range(y_start, y_end + 1)
        for x in range(x_start, x_end + 1)
    ]


def main():
    parser = argparse.ArgumentParser(description="Object patching experiment")
    parser.add_argument("--model", required=True, choices=["llava", "deepseek", "qwen", "llava_onevision"])
    parser.add_argument("--model_path", type=str, default=None,
                        help="HuggingFace model path (uses default per model if omitted)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data_dir", type=str, default="data/",
                        help="Directory containing full_sample dataset and category mapping")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save results JSON")
    parser.add_argument("--layer_step", type=int, default=3,
                        help="Step size for vision layer sweep")
    parser.add_argument("--max_layers", type=int, default=32,
                        help="Maximum vision layer index")
    args = parser.parse_args()

    defaults = {"llava": "liuhaotian/llava-v1.5-7b",
                "deepseek": "deepseek-ai/deepseek-vl2-tiny",
                "qwen": "Qwen/Qwen2-VL-7B-Instruct",
                "llava_onevision": "llava-hf/llava-onevision-qwen2-7b-ov-hf"}
    model_path = args.model_path or defaults[args.model]

    adapter = get_adapter(args.model)
    mt = adapter.load_model(model_path, args.device)

    data_dir = Path(args.data_dir)
    sample_trainset = load_from_disk(str(data_dir / "full_sample"))
    with open(data_dir / "instances_category_map.json") as f:
        category_mapping = json.load(f)

    target_image = Image.new("RGB", (224, 224), (128, 128, 128))
    exp_results = []

    for count, sample in enumerate(tqdm(sample_trainset)):
        category = category_mapping[str(sample["annotations"]["category_id"][0])]
        prompt = f"Is there a {category} in the image? Only answer with yes or no."

        if isinstance(sample["image"], Image.Image):
            sample["image"] = sample["image"].convert("RGB")

        inp_source = adapter.prepare_inputs("", sample["image"], mt)
        inp_target = adapter.prepare_inputs(prompt, target_image.resize(sample["image"].size), mt)

        # Compute object patch indices (Qwen-specific via image_grid_thw)
        if "image_grid_thw" in inp_source:
            _, grid_h, grid_w = inp_source["image_grid_thw"][0]
        else:
            grid_h, grid_w = 16, 16  # fallback

        patch_indices = determine_object_patches(
            sample["annotations"]["bbox"][0],
            sample["image"].size[0], sample["image"].size[1],
            int(grid_h), int(grid_w),
        )

        results = []
        try:
            for ls in range(0, args.max_layers, args.layer_step):
                layer_results = []
                cached_emb = capture_vision_emb(adapter, mt, inp_source, ls)
                for lt in range(0, args.max_layers, args.layer_step):
                    output = patch_vision_and_generate(
                        adapter, mt, inp_target, lt, cached_emb,
                        patch_indices=patch_indices, max_new_tokens=5,
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
