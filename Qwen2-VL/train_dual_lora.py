"""
Dual LoRA Fine-tuning Script for Qwen2-VL
==========================================

This script demonstrates layer-specific LoRA fine-tuning on two different
layer ranges to compare learning behavior at different depths:
- Deep layers (12-18): Close to output, handles high-level reasoning
- Shallow layers (1-7): Close to input, handles low-level features

Author: Generated for educational purposes
"""

import torch
import argparse
from PIL import Image
from pathlib import Path
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType
from typing import List, Tuple, Dict, Any
import json
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os


class SFTDataset(Dataset):
    """
    Dataset class for loading SFT-formatted JSONL data.
    Each line contains a messages structure with user (image + text) and assistant (text) messages.
    """
    
    def __init__(self, jsonl_path: str):
        """
        Initialize dataset from JSONL file.
        
        Args:
            jsonl_path: Path to the JSONL file containing training examples
        """
        self.examples = []
        with open(jsonl_path, "r") as f:
            for line in f:
                if line.strip():
                    self.examples.append(json.loads(line))
        print(f"Loaded {len(self.examples)} examples from {jsonl_path}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch):
    """
    Custom collate function for DataLoader.
    Since we're using batch_size=1 by default, this just returns the single item.
    For larger batches, this would need to handle batching properly.
    """
    # For batch_size=1, DataLoader will pass a list with one item
    if len(batch) == 1:
        return batch[0]
    # For larger batches, return as-is (would need proper batching logic)
    return batch


class DualLoRATrainer:
    """
    Manages training of two separate LoRA adapters on different layer ranges.
    
    Attributes:
        device (str): Device to run training on (cuda:X or cpu)
        model: Base Qwen2-VL model
        processor: Tokenizer and image processor
    """
    
    def __init__(self, device: str = "cuda:2"):
        """
        Initialize the trainer with model and processor.
        
        Args:
            device: Device string (e.g., "cuda:2" or "cpu")
        """
        self.device = device
        
        print("=" * 80)
        print("🚀 Initializing Dual LoRA Trainer")
        print("=" * 80)
        
        # Load base model with gradient checkpointing
        print("\n📦 Loading base model...")
        self.model = self._load_base_model()
        
        # Load processor
        print("📦 Loading processor...")
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        
        print(f"\n✅ Initialization complete!")
        print(f"   Device: {self.device}")
        print(f"   Total parameters: {self.model.num_parameters() / 1_000_000:.0f}M")
    
    def _load_base_model(self):
        """
        Load the base Qwen2-VL model with proper settings for fine-tuning.
        
        Returns:
            model: Initialized model with gradient checkpointing enabled
        """
        # Load model
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            torch_dtype="auto",
            device_map=self.device
        )
        
        # Enable gradient checkpointing BEFORE adding LoRA
        # This reduces memory usage during backpropagation
        print("   Enabling gradient checkpointing...")
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            print("   ✓ Gradient checkpointing enabled")
        
        # Enable input gradients (important for embedding layers)
        if hasattr(model, 'enable_input_require_grads'):
            model.enable_input_require_grads()
            print("   ✓ Input gradients enabled")
        
        # Freeze all base model parameters
        # LoRA will add trainable parameters on top
        for param in model.parameters():
            param.requires_grad = False
        
        return model
    
    def create_lora_config(
        self, 
        layer_range: Tuple[int, int],
        modules_to_tune: List[str] = ["q_proj", "v_proj"],
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05
    ) -> LoraConfig:
        """
        Create a LoRA configuration for specific layers.
        
        Args:
            layer_range: Tuple of (start_layer, end_layer) inclusive
            modules_to_tune: List of module names to apply LoRA to
            r: LoRA rank (number of low-rank dimensions)
            lora_alpha: LoRA scaling parameter
            lora_dropout: Dropout probability for LoRA layers
            
        Returns:
            LoraConfig object configured for the specified layers
            
        Note:
            - Higher 'r' = more parameters but better expressiveness
            - lora_alpha/r ratio controls the learning rate scaling
            - Common modules: q_proj, k_proj, v_proj, o_proj (attention)
        """
        start_layer, end_layer = layer_range
        
        # Build list of target module paths
        # Structure: model.language_model.layers.{layer_num}.self_attn.{module}
        target_modules_list = []
        for layer_i in range(start_layer, end_layer + 1):
            for module_name in modules_to_tune:
                target_modules_list.append(
                    f"model.language_model.layers.{layer_i}.self_attn.{module_name}"
                )
        
        print(f"\n📋 LoRA Configuration for Layers {start_layer}-{end_layer}:")
        print(f"   Targeting {len(target_modules_list)} modules")
        print(f"   Modules: {modules_to_tune}")
        print(f"   Rank (r): {r}")
        print(f"   Alpha: {lora_alpha}")
        print(f"   Example target: {target_modules_list[0]}")
        
        # Create LoRA configuration
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules_list,
            lora_dropout=lora_dropout,
            bias="none",  # Don't train bias terms
            task_type=TaskType.CAUSAL_LM,  # Causal language modeling
        )
        
        return lora_config
    
    def train_and_verify(
        self,
        lora_config: LoraConfig,
        layer_range: Tuple[int, int],
        dataset: SFTDataset,
        num_epochs: int = 3,
        batch_size: int = 1,
        learning_rate: float = 2e-4,
        save_path: str = None,
        save_steps: int = None
    ) -> dict:
        """
        Train a LoRA adapter on the provided dataset and verify gradient flow.
        
        Args:
            lora_config: LoRA configuration to use
            layer_range: Tuple of (start_layer, end_layer) for reporting
            dataset: SFTDataset containing training examples
            num_epochs: Number of training epochs
            batch_size: Batch size for training (default 1 for memory efficiency)
            learning_rate: Learning rate for optimizer
            save_path: Optional path to save the trained adapter
            save_steps: Optional number of steps between checkpoints (None = only save at end)
            
        Returns:
            Dictionary containing training metrics and verification results
        """
        start_layer, end_layer = layer_range
        
        print("\n" + "=" * 80)
        print(f"🔬 Training LoRA Adapter: Layers {start_layer}-{end_layer}")
        print("=" * 80)
        print(f"   Dataset size: {len(dataset)} examples")
        print(f"   Epochs: {num_epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {learning_rate}")
        
        try:
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            # Create PEFT model (wraps base model with LoRA)
            print("\n🔧 Creating PEFT model...")
            peft_model = get_peft_model(self.model, lora_config)
            peft_model.print_trainable_parameters()
            
            # Create optimizer (only train LoRA parameters)
            optimizer = torch.optim.AdamW(
                peft_model.parameters(),
                lr=learning_rate
            )
            
            # Create DataLoader with custom collate function
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,  # Set to 0 to avoid multiprocessing issues
                collate_fn=collate_fn
            )
            
            # Training loop
            print("\n⚡ Starting training...")
            peft_model.train()
            
            total_steps = 0
            total_loss = 0.0
            losses = []
            verification_results = None  # Will store gradient verification from first batch
            
            for epoch in range(num_epochs):
                print(f"\n📚 Epoch {epoch + 1}/{num_epochs}")
                epoch_loss = 0.0
                
                for batch_idx, example in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}")):
                    try:
                        # Prepare inputs from the example
                        # collate_fn ensures we get a single example dict
                        inputs, labels = self._prepare_inputs_from_example(example)
                        
                        # Forward pass
                        outputs = peft_model(**inputs, labels=labels)
                        loss = outputs.loss
                        
                        # Backward pass
                        loss.backward()
                        
                        # Verify gradients on the very first batch (before zero_grad clears them!)
                        if epoch == 0 and batch_idx == 0:
                            print("\n🔍 Verifying gradient flow (first batch)...")
                            verification_results = self._verify_gradients(peft_model, layer_range)
                        
                        # Gradient clipping for stability
                        torch.nn.utils.clip_grad_norm_(peft_model.parameters(), max_norm=1.0)
                        
                        # Optimizer step
                        optimizer.step()
                        optimizer.zero_grad()
                        
                        # Track metrics
                        loss_value = loss.item()
                        epoch_loss += loss_value
                        total_loss += loss_value
                        total_steps += 1
                        losses.append(loss_value)
                        
                        # Save checkpoint if requested
                        if save_steps and total_steps % save_steps == 0:
                            checkpoint_path = f"{save_path}_checkpoint_step_{total_steps}"
                            peft_model.save_pretrained(checkpoint_path)
                            print(f"\n   💾 Saved checkpoint at step {total_steps} to {checkpoint_path}")
                        
                        # Cleanup
                        del outputs, loss, inputs, labels
                        
                    except Exception as e:
                        print(f"\n   ⚠ Error processing batch {batch_idx}: {e}")
                        continue
                
                avg_epoch_loss = epoch_loss / len(dataloader) if len(dataloader) > 0 else 0.0
                print(f"   Average loss: {avg_epoch_loss:.4f}")
            
            # Final metrics
            avg_loss = total_loss / total_steps if total_steps > 0 else 0.0
            print(f"\n✅ Training complete!")
            print(f"   Total steps: {total_steps}")
            print(f"   Average loss: {avg_loss:.4f}")
            
            # Add training metrics to verification results
            if verification_results is None:
                # Fallback: if somehow we didn't verify (shouldn't happen with epochs > 0)
                print("\n⚠ Warning: No gradient verification was performed")
                verification_results = {
                    "success": False,
                    "error": "No verification performed - training may have failed early"
                }
            
            verification_results["training_metrics"] = {
                "total_steps": total_steps,
                "average_loss": avg_loss,
                "final_losses": losses[-10:] if len(losses) >= 10 else losses  # Last 10 losses
            }
            
            # Save adapter if requested
            if save_path:
                print(f"\n💾 Saving adapter to {save_path}...")
                os.makedirs(save_path, exist_ok=True)
                peft_model.save_pretrained(save_path)
                
                # Save metadata
                metadata = {
                    "layer_range": layer_range,
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "dataset_size": len(dataset),
                    "total_steps": total_steps,
                    "average_loss": avg_loss,
                    "verification": verification_results
                }
                with open(Path(save_path) / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)
                print("   ✓ Saved successfully")
            
            # Cleanup
            print("\n🧹 Cleaning up...")
            optimizer.zero_grad()
            del peft_model, optimizer, dataloader
            torch.cuda.empty_cache()
            print("   ✓ Done")
            
            return verification_results
            
        except Exception as e:
            print(f"\n❌ Error during training: {e}")
            import traceback
            traceback.print_exc()
            torch.cuda.empty_cache()
            return {"success": False, "error": str(e)}
    
    def _prepare_inputs_from_example(self, example: Dict[str, Any]):
        """
        Prepare inputs for training from a JSONL example.
        Converts image paths to PIL Images and processes the messages format.
        
        Args:
            example: Dictionary containing 'messages' key with user/assistant messages
            
        Returns:
            Tuple of (inputs_dict, labels_tensor)
        """
        # Extract messages from example
        messages = example["messages"]
        
        # Process messages: convert image paths to PIL Images
        processed_messages = []
        for msg in messages:
            processed_msg = {"role": msg["role"], "content": []}
            
            for item in msg["content"]:
                if item["type"] == "image":
                    # Load image from path
                    image_path = item["image"]
                    if isinstance(image_path, str):
                        image = Image.open(image_path).convert("RGB")
                        # Resize to standard size (Qwen2-VL typically uses 448x448)
                        image.thumbnail((448, 448), Image.Resampling.LANCZOS)
                        processed_msg["content"].append({"type": "image", "image": image})
                    else:
                        # Already a PIL Image
                        processed_msg["content"].append(item)
                else:
                    # Text content - keep as is
                    processed_msg["content"].append(item)
            
            processed_messages.append(processed_msg)
        
        # Apply chat template to format messages
        text_prompt = self.processor.apply_chat_template(
            processed_messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        # Extract images from messages
        images = [
            item["image"] 
            for msg in processed_messages if msg["role"] == "user"
            for item in msg["content"] 
            if isinstance(item, dict) and item.get("type") == "image"
        ]
        
        # Process inputs (tokenize text + encode image)
        inputs_list = self.processor(
            text=[text_prompt],
            images=images,
            padding=True,
            return_tensors="pt"
        )
        
        # Move tensors to device
        inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
            for k, v in inputs_list.items()
        }
        
        # Create labels (copy of input_ids for causal LM)
        # For training, we want the model to predict the assistant's response
        labels = inputs["input_ids"].clone()
        
        return inputs, labels
    
    def _verify_gradients(self, peft_model, layer_range: Tuple[int, int]) -> dict:
        """
        Verify that gradients are flowing correctly through LoRA layers.
        
        Args:
            peft_model: PEFT-wrapped model to verify
            layer_range: Tuple of (start_layer, end_layer)
            
        Returns:
            Dictionary with verification results
        """
        start_layer, end_layer = layer_range
        
        print("   Checking target layer gradients...")
        
        # Check a sample target layer (first in range)
        target_layer = peft_model.base_model.model.language_model.layers[start_layer].self_attn.q_proj
        
        # Check Matrix A gradients
        lora_A_grad = target_layer.lora_A.default.weight.grad
        lora_A_norm = torch.sum(torch.abs(lora_A_grad)).item() if lora_A_grad is not None else 0.0
        
        # Check Matrix B gradients
        lora_B_grad = target_layer.lora_B.default.weight.grad
        lora_B_norm = torch.sum(torch.abs(lora_B_grad)).item() if lora_B_grad is not None else 0.0
        
        print(f"      Layer {start_layer} q_proj:")
        print(f"      - Matrix A gradient: {lora_A_norm:.6f}")
        print(f"      - Matrix B gradient: {lora_B_norm:.6f}")
        
        # Count all LoRA parameters with gradients
        lora_count = 0
        total_norm = 0.0
        
        for name, param in peft_model.named_parameters():
            if param.requires_grad and 'lora' in name.lower() and param.grad is not None:
                norm = torch.sum(torch.abs(param.grad)).item()
                if norm > 0:
                    lora_count += 1
                    total_norm += norm
        
        print(f"   ✓ LoRA params with gradients: {lora_count}")
        print(f"   ✓ Total gradient magnitude: {total_norm:.2f}")
        
        # Check non-target layer (should not have LoRA)
        non_target = 0 if start_layer > 5 else end_layer + 1
        if non_target < len(peft_model.base_model.model.language_model.layers):
            non_target_layer = peft_model.base_model.model.language_model.layers[non_target].self_attn.q_proj
            has_lora = hasattr(non_target_layer, 'lora_A')
            
            if not has_lora:
                print(f"   ✓ Layer {non_target} correctly frozen (no LoRA)")
            else:
                print(f"   ⚠ Layer {non_target} unexpectedly has LoRA!")
        
        success = lora_count > 0 and total_norm > 0
        
        if success:
            print(f"\n   ✅ SUCCESS: Gradients flowing correctly!")
        else:
            print(f"\n   ❌ FAILED: No gradient flow detected")
        
        return {
            "success": success,
            "lora_params_with_gradients": lora_count,
            "total_gradient_magnitude": total_norm,
            "sample_layer_gradient": {
                "layer": start_layer,
                "matrix_A": lora_A_norm,
                "matrix_B": lora_B_norm
            }
        }


def main():
    """
    Main training function - orchestrates dual LoRA training on misclassified dataset.
    """
    parser = argparse.ArgumentParser(description="Dual LoRA fine-tuning on misclassified dataset")
    
    # Dataset path
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="qwen_balanced_sft_train.jsonl",
        help="Path to the SFT-formatted JSONL dataset (use TRAIN split only!)"
    )
    
    # Device configuration
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:2" if torch.cuda.is_available() else "cpu",
        help="Device to use for training (e.g., 'cuda:2' or 'cpu')"
    )
    
    # Layer ranges to train
    parser.add_argument(
        "--deep_layers",
        type=int,
        nargs=2,
        default=[14, 16],
        help="Deep layer range (start, end) - default: 12 18"
    )
    parser.add_argument(
        "--shallow_layers",
        type=int,
        nargs=2,
        default=[1, 7],
        help="Shallow layer range (start, end) - default: 1 7"
    )
    
    # LoRA hyperparameters
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank (r)")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument(
        "--modules",
        type=str,
        nargs="+",
        default=["q_proj", "v_proj"],
        help="Modules to apply LoRA to (default: q_proj v_proj)"
    )
    
    # Training hyperparameters
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (default 1 for memory efficiency)")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    
    # Output directories
    parser.add_argument(
        "--deep_output",
        type=str,
        default="./lora_adapters/deep_layers_12-18",
        help="Output path for deep layers adapter"
    )
    parser.add_argument(
        "--shallow_output",
        type=str,
        default="./lora_adapters/shallow_layers_1-7",
        help="Output path for shallow layers adapter"
    )
    
    # Optional checkpointing
    parser.add_argument(
        "--save_steps",
        type=int,
        default=None,
        help="Save checkpoint every N steps (None = only save at end)"
    )
    
    args = parser.parse_args()
    
    # Convert layer ranges to tuples
    DEEP_LAYERS = tuple(args.deep_layers)
    SHALLOW_LAYERS = tuple(args.shallow_layers)
    
    print("\n" + "=" * 80)
    print("🎯 DUAL LORA FINE-TUNING ON MISCLASSIFIED DATASET")
    print("=" * 80)
    print(f"\n📊 Configuration:")
    print(f"   Dataset: {args.dataset_path}")
    print(f"   Deep layers: {DEEP_LAYERS[0]}-{DEEP_LAYERS[1]}")
    print(f"   Shallow layers: {SHALLOW_LAYERS[0]}-{SHALLOW_LAYERS[1]}")
    print(f"   LoRA rank: {args.lora_rank}")
    print(f"   LoRA alpha: {args.lora_alpha}")
    print(f"   LoRA dropout: {args.lora_dropout}")
    print(f"   Target modules: {args.modules}")
    print(f"   Epochs: {args.num_epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Device: {args.device}")
    
    # Load dataset
    print(f"\n📂 Loading dataset from {args.dataset_path}...")
    if not os.path.exists(args.dataset_path):
        print(f"❌ Error: Dataset file not found: {args.dataset_path}")
        return
    
    dataset = SFTDataset(args.dataset_path)
    
    if len(dataset) == 0:
        print("❌ Error: Dataset is empty!")
        return
    
    # Initialize trainer
    trainer = DualLoRATrainer(device=args.device)
    
    # --- Train Deep Layers (12-18) ---
    print("\n\n" + "=" * 80)
    print("🎯 TRAINING SET 1: DEEP LAYERS (14-16)")
    print("=" * 80)
    print("These layers are closer to the output and handle high-level reasoning")
    
    deep_config = trainer.create_lora_config(
        layer_range=DEEP_LAYERS,
        modules_to_tune=args.modules,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    
    deep_results = trainer.train_and_verify(
        lora_config=deep_config,
        layer_range=DEEP_LAYERS,
        dataset=dataset,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_path=args.deep_output,
        save_steps=args.save_steps
    )
    
    # --- Train Shallow Layers (1-7) ---
    print("\n\n" + "=" * 80)
    print("🎯 TRAINING SET 2: SHALLOW LAYERS (1-7)")
    print("=" * 80)
    print("These layers are closer to the input and handle low-level features")
    
    shallow_config = trainer.create_lora_config(
        layer_range=SHALLOW_LAYERS,
        modules_to_tune=args.modules,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    
    shallow_results = trainer.train_and_verify(
        lora_config=shallow_config,
        layer_range=SHALLOW_LAYERS,
        dataset=dataset,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_path=args.shallow_output,
        save_steps=args.save_steps
    )
    
    # --- Final Summary ---
    print("\n\n" + "=" * 80)
    print("📊 FINAL SUMMARY")
    print("=" * 80)
    
    print(f"\n🔵 Deep Layers ({DEEP_LAYERS[0]}-{DEEP_LAYERS[1]}):")
    print(f"   Status: {'✅ Success' if deep_results.get('success', False) else '❌ Failed'}")
    if 'training_metrics' in deep_results:
        print(f"   Total steps: {deep_results['training_metrics'].get('total_steps', 0)}")
        print(f"   Average loss: {deep_results['training_metrics'].get('average_loss', 0):.4f}")
    print(f"   Trainable params: {deep_results.get('lora_params_with_gradients', 0)}")
    print(f"   Gradient magnitude: {deep_results.get('total_gradient_magnitude', 0):.2f}")
    print(f"   Saved to: {args.deep_output}")
    
    print(f"\n🟢 Shallow Layers ({SHALLOW_LAYERS[0]}-{SHALLOW_LAYERS[1]}):")
    print(f"   Status: {'✅ Success' if shallow_results.get('success', False) else '❌ Failed'}")
    if 'training_metrics' in shallow_results:
        print(f"   Total steps: {shallow_results['training_metrics'].get('total_steps', 0)}")
        print(f"   Average loss: {shallow_results['training_metrics'].get('average_loss', 0):.4f}")
    print(f"   Trainable params: {shallow_results.get('lora_params_with_gradients', 0)}")
    print(f"   Gradient magnitude: {shallow_results.get('total_gradient_magnitude', 0):.2f}")
    print(f"   Saved to: {args.shallow_output}")
    
    print("\n" + "=" * 80)
    print("✅ DUAL TRAINING COMPLETE!")
    print("=" * 80)
    print("\n💡 Next Steps:")
    print("   1. Compare the two adapters' performance on test data")
    print("   2. Analyze which layer range learns better for your task")
    print("   3. Evaluate on the original test set to measure improvement")
    print("   4. Consider training on more data or adjusting hyperparameters")


if __name__ == "__main__":
    main()

