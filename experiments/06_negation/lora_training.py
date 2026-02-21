"""Dual LoRA Training for Negation Understanding.

Trains two separate LoRA adapters on different layer ranges to investigate
how fine-tuning at different depths affects negation understanding:
  - **Deep layers** (14-16): High-level semantic reasoning
  - **Shallow layers** (1-7): Low-level feature processing

See the paper Section on LoRA-based intervention for details.

Usage::

    python -m experiments.06_negation.lora_training \
        --model_path Qwen/Qwen2-VL-7B-Instruct \
        --train_data data/qwen_balanced_sft_train.jsonl \
        --output_dir configs/lora_adapters/
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model


def create_lora_config(target_layers, rank=8, alpha=16):
    """Create a LoRA config targeting specific decoder layers."""
    target_modules = []
    for layer_idx in target_layers:
        target_modules.extend([
            f"model.layers.{layer_idx}.self_attn.q_proj",
            f"model.layers.{layer_idx}.self_attn.v_proj",
        ])
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )


def main():
    parser = argparse.ArgumentParser(description="Dual LoRA training")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="configs/lora_adapters/")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    print("Loading base model...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="auto", device_map=args.device,
    )
    processor = AutoProcessor.from_pretrained(args.model_path)

    configs = {
        "deep_layers": {"layers": list(range(14, 17)), "name": "deep_layers"},
        "shallow_layers": {"layers": list(range(1, 8)), "name": "shallow_layers"},
    }

    for config_name, config in configs.items():
        print(f"\n{'='*60}")
        print(f"Training {config_name} adapter (layers {config['layers']})")
        print(f"{'='*60}")

        lora_config = create_lora_config(config["layers"])
        peft_model = get_peft_model(model, lora_config)
        peft_model.print_trainable_parameters()

        # Training loop placeholder -- the full training logic
        # depends on the specific SFT data format and training framework.
        # See the original train_dual_lora.py for the complete implementation.
        print(f"[Placeholder] Would train on {args.train_data} for {args.epochs} epochs")

        output_path = Path(args.output_dir) / config_name
        output_path.mkdir(parents=True, exist_ok=True)
        peft_model.save_pretrained(str(output_path))
        print(f"Saved adapter to {output_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
