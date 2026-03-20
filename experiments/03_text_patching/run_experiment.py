"""Text-Only Patching Experiment (Decoder Level).

Patches only the text-token hidden states (excluding image tokens) from a
source forward pass into a target forward pass.  This isolates the
contribution of linguistic representations to the model's output, revealing
at which decoder depth text-only evidence can override visual evidence.

Usage::

    python -m experiments.03_text_patching.run_experiment \
        --model qwen --device cuda:0 --output results/qwen_text_patch.json
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from datasets import load_from_disk
from tqdm import tqdm

from src.models import get_adapter
from src.patching.decoder_patching import capture_decoder_hs, patch_decoder_and_generate


def _make_text_only_patch_fn(text_start: int, text_len: int):
    """Return a patch_fn that replaces only the last *text_len* positions."""
    def _patch(target_out, source_hs):
        target_out[0, -text_len:] = source_hs[0, -text_len:]
        return target_out
    return _patch


def main():
    parser = argparse.ArgumentParser(description="Text-only patching experiment")
    parser.add_argument("--model", required=True, choices=["llava", "deepseek", "qwen", "llava_onevision"])
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--layer_step", type=int, default=3)
    parser.add_argument("--max_layers", type=int, default=28)
    parser.add_argument("--text_token_count", type=int, default=19,
                        help="Number of trailing text tokens to patch")
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

    blank_image = Image.new("RGB", (1, 1), (0, 0, 0))
    exp_results = []

    for count, sample in enumerate(tqdm(sample_trainset)):
        category = category_mapping[str(sample["annotations"]["category_id"][0])]
        prompt = f"Is there a {category} in the image? only answer with yes or no"

        if isinstance(sample["image"], Image.Image):
            sample["image"] = sample["image"].convert("RGB")

        inp_source = adapter.prepare_inputs(prompt, sample["image"], mt)
        inp_target = adapter.prepare_inputs(prompt, blank_image, mt)

        patch_fn = _make_text_only_patch_fn(0, args.text_token_count)

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
