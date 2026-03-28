"""Pathology Diagnosis Patching Experiment (Section 5.2).

This experiment investigates the root causes of VLM failures on hard tasks
(like adversarial queries or 3D spatial reasoning) by diagnosing bottlenecks
in their Fusion Band.

Experimental Logic:
- Target Run (The Failure): [Image] + [Tricky/Adversarial Question].
  The model relies on language priors or fails to process 3D spatial relations.
- Source Run (The Cure): [Image] + "" (Empty Text Prompt).
  The model extracts pure, unpoisoned visual features.
- Intervention: Patch the hidden states from the Source Run into the Target Run,
  layer by layer, and record the "Flip Rate" (when the incorrect answer flips to correct).

Usage::

    python -m experiments.08_pathology_diagnosis.run_experiment \
        --model          qwen25 \
        --device         cuda:0 \
        --dataset_path   data/naturalbench/naturalbench_pathology.json \
        --output         results/pathology_qwen25_naturalbench.json \
        --max_samples    200
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


def load_dataset(path: str, max_samples: int):
    """Load the flat JSON dataset prepared by data/prepare/prepare_*.py"""
    with open(path) as f:
        raw = json.load(f)
    return raw[:max_samples]


def run_pathology_sweep(adapter, mt, image, target_question, expected_answer, max_layers, layer_step):
    """Run the causal patching sweep for one sample.

    Source pass: real image + "" (Empty Text Prompt)
    Target pass: real image + target_question (Tricky/Adversarial)

    Returns a 2-D list of shape [num_source_layers][num_target_layers],
    where each cell is 1 if the model's answer flips to the correct one, else 0.
    Also returns the baseline target answer (without patching).
    """
    # Source: empty prompt to get pure visual features
    source_prompt = ""
    inp_source = adapter.prepare_inputs(source_prompt, image, mt)
    
    # Target: the tricky question
    inp_target = adapter.prepare_inputs(target_question, image, mt)

    try:
        img_start, img_end = adapter.find_image_token_range(mt, inp_source)
        # We assume the image token range is the same in source and target
        # since the image is the same, only the text prompt changes.
        # But we should verify target range just in case.
        target_img_start, target_img_end = adapter.find_image_token_range(mt, inp_target)
        
        if (img_end - img_start) != (target_img_end - target_img_start):
            # If lengths differ, we can't do simple positional patching.
            # We fallback to the source image token range length.
            length = min(img_end - img_start, target_img_end - target_img_start)
            img_end = img_start + length
            target_img_end = target_img_start + length
            
    except ValueError as e:
        print(f"Warning: Could not find image tokens: {e}")
        return [], "ERROR"

    # The patch function replaces image tokens in the target with image tokens from the source
    def patch_fn(target_out: torch.Tensor, source_hs: torch.Tensor) -> torch.Tensor:
        target_out[0, target_img_start:target_img_end] = source_hs[0, img_start:img_end]
        return target_out

    # First, get the baseline target answer without patching
    baseline_answer = adapter.generate(mt, inp_target, max_new_tokens=10).strip()
    
    # Check if the baseline answer is already correct.
    # We do simple substring matching. For yes/no, we check lowercased.
    expected_lower = expected_answer.lower()
    baseline_lower = baseline_answer.lower()
    
    # If expected answer is "yes" or "no", do exact match checking
    if expected_lower in ["yes", "no"]:
        baseline_is_correct = expected_lower in baseline_lower
    else:
        # For multiple choice or other, check if expected is in baseline
        baseline_is_correct = expected_lower in baseline_lower

    results = []
    for ls in range(0, max_layers, layer_step):
        cached_hs = capture_decoder_hs(adapter, mt, inp_source, ls)
        row = []
        for lt in range(0, max_layers, layer_step):
            output = patch_decoder_and_generate(
                adapter, mt, inp_target, lt, cached_hs,
                patch_fn, max_new_tokens=10,
            )
            output_lower = output.lower()
            
            if expected_lower in ["yes", "no"]:
                is_correct = expected_lower in output_lower
            else:
                is_correct = expected_lower in output_lower
                
            # We record 1 if it's correct (Flip to correct), 0 otherwise
            row.append(1 if is_correct else 0)
        results.append(row)

    return results, baseline_answer, baseline_is_correct


def main():
    parser = argparse.ArgumentParser(
        description="Pathology diagnosis patching experiment (Section 5.2)")
    parser.add_argument("--model", required=True,
                        choices=["llava", "deepseek", "qwen", "qwen25", "internvl", "llava_onevision", "internvl25"])
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the prepared JSON dataset (e.g., naturalbench_pathology.json)")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save the experiment results")
    parser.add_argument("--max_samples", type=int, default=200,
                        help="Max samples to process (default: 200)")
    parser.add_argument("--layer_step", type=int, default=2)
    parser.add_argument("--max_layers", type=int, default=28)
    parser.add_argument("--test", action="store_true",
                        help="Test mode: run only 2 samples with layer_step=4")
    args = parser.parse_args()

    if args.test:
        args.max_samples = 2
        args.layer_step = 4
        print("[TEST MODE] Running 2 samples with layer_step=4")

    defaults = {
        "llava":           "liuhaotian/llava-v1.5-7b",
        "deepseek":        "deepseek-ai/deepseek-vl2-tiny",
        "qwen":            "Qwen/Qwen2-VL-7B-Instruct",
        "internvl":        "OpenGVLab/InternVL2-8B",
        "llava_onevision": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        "qwen25":          "Qwen/Qwen2.5-VL-7B-Instruct",
        "internvl25":      "OpenGVLab/InternVL2_5-8B",
    }
    model_path = args.model_path or defaults[args.model]

    adapter = get_adapter(args.model)
    mt = adapter.load_model(model_path, args.device)

    # Load dataset
    samples = load_dataset(args.dataset_path, args.max_samples)
    print(f"Loaded {len(samples)} samples from {args.dataset_path}")

    if not samples:
        print("ERROR: Dataset is empty.")
        return

    # Resume from checkpoint
    output_path = Path(args.output)
    if output_path.exists():
        with open(output_path) as f:
            exp_results = json.load(f)
        completed = len(exp_results)
        print(f"Resuming from checkpoint: {completed} samples already done")
    else:
        exp_results = []
        completed = 0

    for idx, sample in enumerate(tqdm(samples)):
        if idx < completed:
            continue

        try:
            image_file = sample["image_file"]
            target_question = sample["target_question"]
            expected_answer = sample["expected_answer"]
            
            # For boolean questions, make sure the prompt forces a yes/no answer
            # to match the evaluation logic
            if str(expected_answer).lower() in ["yes", "no"]:
                prompt = f"{target_question} Please answer yes or no."
            else:
                prompt = target_question

            image = Image.open(image_file).convert("RGB")
            
            sweep_results, baseline_ans, baseline_correct = run_pathology_sweep(
                adapter, mt, image, prompt, expected_answer, args.max_layers, args.layer_step
            )
            
            result = {
                "sample_id": idx,
                "image_file": image_file,
                "target_question": target_question,
                "expected_answer": expected_answer,
                "baseline_answer": baseline_ans,
                "baseline_correct": baseline_correct,
                "meta": sample.get("meta", {}),
                "patching_sweep": sweep_results,
            }
            exp_results.append(result)
            
        except Exception as e:
            import traceback
            print(f"Error at sample {idx}: {e}")
            traceback.print_exc()
            continue

        if idx % 10 == 0:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(exp_results, f, indent=2)

    print(f"\nCompleted: {len(exp_results)} samples")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(exp_results, f, indent=2)


if __name__ == "__main__":
    main()
