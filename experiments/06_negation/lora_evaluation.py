"""Dual LoRA Evaluation for Negation Understanding.

Evaluates the performance of baseline (no LoRA), deep-layer LoRA, and
shallow-layer LoRA adapters on negation benchmarks.  Compares accuracy
to understand which decoder depths are most important for negation reasoning.

Usage::

    python -m experiments.06_negation.lora_evaluation \
        --model_path Qwen/Qwen2-VL-7B-Instruct \
        --deep_adapter configs/lora_adapters/deep_layers \
        --shallow_adapter configs/lora_adapters/shallow_layers \
        --test_data data/qwen_balanced_sft_test.jsonl \
        --output results/negation_lora_eval.json
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from tqdm import tqdm


def load_test_set(test_path):
    """Load test set from JSONL format."""
    samples = []
    with open(test_path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def evaluate_model(model, processor, test_samples, device, max_new_tokens=5):
    """Run evaluation and return accuracy."""
    correct, total = 0, 0
    for sample in tqdm(test_samples, desc="Evaluating"):
        image_path = sample.get("image", "")
        question = sample.get("question", "")
        expected = sample.get("answer", "").strip().lower()

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            continue

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }]
        from qwen_vl_utils import process_vision_info
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        output = processor.batch_decode(
            [o[len(i):] for i, o in zip(inputs["input_ids"], output_ids)],
            skip_special_tokens=True,
        )[0].strip().lower()

        if expected in output or output in expected:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="LoRA negation evaluation")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--deep_adapter", type=str, default="configs/lora_adapters/deep_layers")
    parser.add_argument("--shallow_adapter", type=str, default="configs/lora_adapters/shallow_layers")
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    test_samples = load_test_set(args.test_data)
    print(f"Loaded {len(test_samples)} test samples")

    results = {}

    # Baseline evaluation
    print("\n--- Baseline (no LoRA) ---")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="auto", device_map=args.device,
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    model.eval()
    results["baseline"] = evaluate_model(model, processor, test_samples, args.device)
    print(f"Baseline accuracy: {results['baseline']:.4f}")

    # Deep LoRA evaluation
    print("\n--- Deep Layers LoRA ---")
    deep_model = PeftModel.from_pretrained(model, args.deep_adapter)
    deep_model.eval()
    results["deep_lora"] = evaluate_model(deep_model, processor, test_samples, args.device)
    print(f"Deep LoRA accuracy: {results['deep_lora']:.4f}")

    # Shallow LoRA evaluation
    print("\n--- Shallow Layers LoRA ---")
    del deep_model
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="auto", device_map=args.device,
    )
    shallow_model = PeftModel.from_pretrained(model, args.shallow_adapter)
    shallow_model.eval()
    results["shallow_lora"] = evaluate_model(shallow_model, processor, test_samples, args.device)
    print(f"Shallow LoRA accuracy: {results['shallow_lora']:.4f}")

    print(f"\n{'='*40}")
    print("Summary:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
