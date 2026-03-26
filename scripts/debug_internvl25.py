#!/usr/bin/env python3
"""Debug script for InternVL2.5 adapter pipeline.

This script runs through the full patching pipeline step by step,
printing detailed diagnostics at each stage. It is designed to be
run on the server to identify exactly where 'NoneType' errors occur.

Usage:
    cd ~/MINT
    CUDA_VISIBLE_DEVICES=5 python scripts/debug_internvl25.py \
        --pope_data data/pope/pope_minimal_pairs.json \
        --num_pairs 3
"""

import argparse
import json
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

# Ensure MINT root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import get_adapter
from src.patching.decoder_patching import (
    capture_decoder_hs,
    patch_decoder_and_generate,
    make_image_token_patch_fn,
)


def load_pope_pairs(path, n):
    """Load n POPE minimal pairs."""
    with open(path) as f:
        raw = json.load(f)
    groups = defaultdict(lambda: {"yes": [], "no": []})
    for item in raw:
        groups[item["image_file"]][item["label"]].append(item)
    pairs = []
    for image_file, labels in groups.items():
        if labels["yes"] and labels["no"]:
            pos = labels["yes"][0]
            neg = labels["no"][0]
            pairs.append({
                "dimension": "object",
                "image_file": image_file,
                "positive": {"question": pos["question"], "label": "yes"},
                "negative": {"question": neg["question"], "label": "no"},
            })
        if len(pairs) >= n:
            break
    return pairs


def test_step(name, fn):
    """Run fn(), print result or full traceback on failure."""
    print(f"\n  [{name}]")
    try:
        result = fn()
        print(f"  -> OK: {result}")
        return result
    except Exception:
        print(f"  -> FAILED:")
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pope_data", required=True)
    parser.add_argument("--num_pairs", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model_path", default="OpenGVLab/InternVL2_5-8B")
    args = parser.parse_args()

    print("=" * 70)
    print("InternVL2.5 Debug Script")
    print("=" * 70)

    # ---- Step 0: Environment info ----
    print("\n--- Step 0: Environment ---")
    import transformers
    print(f"  transformers version: {transformers.__version__}")
    print(f"  torch version: {torch.__version__}")
    print(f"  device: {args.device}")

    # ---- Step 1: Load model ----
    print("\n--- Step 1: Load model ---")
    adapter = get_adapter("internvl25")
    mt = adapter.load_model(args.model_path, args.device)
    print(f"  model type: {type(mt.model).__name__}")
    print(f"  language_model type: {type(mt.model.language_model).__name__}")
    print(f"  img_context_token_id: {mt.model.img_context_token_id}")
    print(f"  num_decoder_layers: {adapter.num_decoder_layers(mt)}")

    # Check GenerationMixin
    from transformers import GenerationMixin
    lm = mt.model.language_model
    has_gen = hasattr(lm, "generate")
    is_mixin = isinstance(lm, GenerationMixin)
    has_config = getattr(lm, "generation_config", None) is not None
    print(f"  language_model has .generate: {has_gen}")
    print(f"  language_model is GenerationMixin: {is_mixin}")
    print(f"  language_model has generation_config: {has_config}")

    # ---- Step 2: Load pairs ----
    print("\n--- Step 2: Load data ---")
    pairs = load_pope_pairs(args.pope_data, args.num_pairs)
    print(f"  Loaded {len(pairs)} pairs")

    # ---- Step 3: Test each pair ----
    for i, pair in enumerate(pairs):
        print(f"\n{'='*70}")
        print(f"PAIR {i}: {pair['image_file']}")
        print(f"  positive Q: {pair['positive']['question']}")
        print(f"  negative Q: {pair['negative']['question']}")
        print(f"{'='*70}")

        # 3a: Load image
        image = test_step("Load image", lambda: Image.open(pair["image_file"]).convert("RGB"))
        if image is None:
            continue

        # 3b: Create blank image
        blank = Image.new("RGB", image.size, (0, 0, 0))

        q = pair["positive"]["question"]

        # 3c: prepare_inputs for source (real image)
        inp_src = test_step("prepare_inputs (source)", lambda: adapter.prepare_inputs(q, image, mt))
        if inp_src is None:
            continue

        print(f"  -> input_ids shape: {inp_src['input_ids'].shape}")
        print(f"  -> pixel_values shape: {inp_src.get('pixel_values', 'MISSING')}")
        if "pixel_values" in inp_src:
            print(f"     pixel_values.shape: {inp_src['pixel_values'].shape}")
        print(f"  -> image_flags: {inp_src.get('image_flags', 'MISSING')}")
        print(f"  -> keys: {list(inp_src.keys())}")

        # 3d: prepare_inputs for target (blank image)
        inp_tgt = test_step("prepare_inputs (target)", lambda: adapter.prepare_inputs(q, blank, mt))
        if inp_tgt is None:
            continue

        print(f"  -> input_ids shape: {inp_tgt['input_ids'].shape}")

        # 3e: find_image_token_range
        img_range = test_step("find_image_token_range", lambda: adapter.find_image_token_range(mt, inp_src))
        if img_range is None:
            continue
        img_start, img_end = img_range

        # 3f: get_forward_inputs (what capture_decoder_hs will use)
        fwd_inputs = adapter.get_forward_inputs(inp_src)
        print(f"\n  [get_forward_inputs]")
        print(f"  -> keys: {list(fwd_inputs.keys())}")
        for k, v in fwd_inputs.items():
            if isinstance(v, torch.Tensor):
                print(f"     {k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")
            else:
                print(f"     {k}: {type(v).__name__} = {v}")

        # 3g: Direct mt.model(**fwd_inputs) forward pass
        def _test_forward():
            with torch.no_grad():
                out = mt.model(**fwd_inputs)
            return f"type={type(out).__name__}"
        test_step("mt.model(**fwd_inputs) direct forward", _test_forward)

        # 3h: capture_decoder_hs at layer 0
        cached = test_step("capture_decoder_hs (layer=0)", lambda: capture_decoder_hs(adapter, mt, inp_src, 0))
        if cached is None:
            continue
        print(f"  -> cached shape: {cached.shape}, dtype: {cached.dtype}")

        # 3i: Test generate WITHOUT any patch hook (bare generate)
        def _test_bare_generate():
            return adapter.generate(mt, inp_tgt, max_new_tokens=5)
        bare_out = test_step("adapter.generate (no patch hook)", _test_bare_generate)

        # 3j: Test generate WITH patch hook (the full pipeline)
        patch_fn = make_image_token_patch_fn(img_start, img_end)
        def _test_patched_generate():
            return patch_decoder_and_generate(
                adapter, mt, inp_tgt, 0, cached, patch_fn, max_new_tokens=5
            )
        patched_out = test_step("patch_decoder_and_generate (ls=0, lt=0)", _test_patched_generate)

        # 3k: Test a full mini sweep (2x2 grid)
        print(f"\n  [Mini sweep: 2x2 grid, layers 0 and 16]")
        for ls in [0, 16]:
            cached_ls = test_step(
                f"capture_decoder_hs (layer={ls})",
                lambda ls=ls: capture_decoder_hs(adapter, mt, inp_src, ls)
            )
            if cached_ls is None:
                continue
            for lt in [0, 16]:
                def _test_cell(lt=lt, cached_ls=cached_ls):
                    return patch_decoder_and_generate(
                        adapter, mt, inp_tgt, lt, cached_ls, patch_fn, max_new_tokens=5
                    )
                test_step(f"patch (ls={ls}, lt={lt})", _test_cell)

    print(f"\n{'='*70}")
    print("DEBUG COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
