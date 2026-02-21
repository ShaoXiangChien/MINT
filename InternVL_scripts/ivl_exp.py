import os
import sys
import torch
import argparse
from PIL import Image
from tqdm import tqdm
import json
from datasets import load_from_disk

# Add current directory to path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from general_utils import ModelAndTokenizer
from patchscopes_utils import inspect_vision_in_lm

def parse_args():
    parser = argparse.ArgumentParser(description="Run InternVL Patchscopes Experiment")
    parser.add_argument("--mode", type=str, default="test", choices=["test", "full_sample"],
                        help="Experiment mode: 'test' for small sample size, 'full_sample' for entire dataset")
    parser.add_argument("--limit", type=int, default=5, 
                        help="Number of samples to process in test mode (default: 5)")
    parser.add_argument("--model_path", type=str, default="OpenGVLab/InternVL3_5-8B",
                        help="Path or HF ID of the model to load")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Starting sample index (for parallel processing)")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="Ending sample index (exclusive, for parallel processing)")
    parser.add_argument("--gpu_id", type=int, default=None,
                        help="GPU ID to use (will set CUDA_VISIBLE_DEVICES internally)")
    return parser.parse_args()

def load_vlm(device, model_path):
    print(f"Loading model from {model_path}...")
    mt = ModelAndTokenizer(
        model_path,
        device=device,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    return mt

def load_cate_mapping(mapping_path):
    with open(mapping_path, "r") as f:
        category_map = json.load(f)
    return category_map

if __name__ == "__main__":
    args = parse_args()
    
    # Set GPU if specified
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        print(f"Set CUDA_VISIBLE_DEVICES={args.gpu_id}")
    
    # Set device
    # InternVL with 38B usually requires multiple GPUs or specific CUDA setup.
    # We will let accelerate/transformers handle device map "auto" which uses all available visible GPUs.
    # To limit GPUs, user should run with CUDA_VISIBLE_DEVICES=0,1 python ivl_exp.py ...
    
    # Check CUDA availability but rely on model loading for placement
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. Model loading might fail or be extremely slow.")
        device = "cpu"
    else:
        # We won't enforce "cuda" string here because model might be split across devices
        # But for utility functions that need a device, we can default to the first one.
        device = "cuda"

    print(f"Using device context: {device} (Model loading will use device_map='auto')")
    print(f"Mode: {args.mode}")
    
    # Determine sample range
    if args.end_idx is not None:
        print(f"Processing samples {args.start_idx} to {args.end_idx-1}")
    else:
        print(f"Processing samples starting from {args.start_idx}")

    # Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "../data")
    
    # Create unique result file based on sample range
    if args.end_idx is not None:
        result_dir = os.path.join(base_dir, f"ivl_results_{args.start_idx}_{args.end_idx}.json")
    else:
        result_dir = os.path.join(base_dir, f"ivl_results_{args.start_idx}.json")
    
    print(f"Loading data from {data_dir}")
    sample_trainset = load_from_disk(os.path.join(data_dir, "full_sample"))
    category_mapping = load_cate_mapping(os.path.join(data_dir, "instances_category_map.json"))

    # Load vision-language model
    mt = load_vlm(device, args.model_path)
    print(f"Model loaded. Number of layers: {mt.num_layers}")
    
    # Load target image (dummy image for patching target)
    # Ensure this image exists or create it
    target_image_path = os.path.join(base_dir, "sample.png") # reusing sample image as dummy/target if needed
    if not os.path.exists(target_image_path):
         Image.new('RGB', (448, 448), color='red').save(target_image_path)
    
    target_image = Image.open(target_image_path).convert('RGB')

    exp_results = []
    
    # Determine which samples to process
    end_idx = args.end_idx if args.end_idx is not None else len(sample_trainset)
    
    # In test mode, limit the range
    if args.mode == "test":
        end_idx = min(args.start_idx + args.limit, end_idx)
    
    # Create subset of dataset
    sample_subset = sample_trainset.select(range(args.start_idx, min(end_idx, len(sample_trainset))))
    pbar = tqdm(sample_subset, desc=f"Processing samples {args.start_idx}-{end_idx}")
    
    # We will scan layers with a stride to avoid too much time on 38B model
    # Assuming ~40-80 layers. Let's do every 4th layer or similar.
    layers_to_scan = list(range(0, mt.num_layers, 4)) 
    if mt.num_layers - 1 not in layers_to_scan:
        layers_to_scan.append(mt.num_layers - 1)

    for local_idx, sample in enumerate(pbar):
        # Calculate global index
        count = args.start_idx + local_idx
            
        results = []
        try:
            category_id = str(sample["annotations"]["category_id"][0])
            if category_id not in category_mapping:
                print(f"Category ID {category_id} not found in mapping. Skipping.")
                continue
                
            category = category_mapping[category_id]
            # Prompt: "What is in the image? {category} or apple"
            # InternVL prompt style might need to be more chat-like, but let's stick to the experiment format
            prompt = f"What is in the image? {category} or apple? no other words"
            
            source_image = sample["image"]
            
            # Experiment Loop
            # We iterate over source layers and target layers
            # For 38B model, this is O(N^2) forward passes. 
            # N=10 -> 100 passes per sample.
            
            sample_results = []
            
            for layer_source in layers_to_scan:
                layer_preds = []
                for layer_target in layers_to_scan:
                    # Skip if source > target (usually we patch forward, but patchscopes can do backward/arbitrary)
                    # But typically we want to see if layer X info is usable at layer Y.
                    
                    result_text = inspect_vision_in_lm(
                        mt, 
                        prompt, 
                        source_image, 
                        target_image, 
                        layer_source, 
                        layer_target,
                        max_gen_len=20 # Short generation
                    )

                    print(f"{layer_source}-{layer_target}: {result_text}")
                    
                    # Check prediction
                    # simple keyword check
                    prediction = 1 if category.lower() in result_text.lower() else 0
                    layer_preds.append({
                        "source": layer_source,
                        "target": layer_target,
                        "prediction": prediction,
                        "text": result_text
                    })
                
                sample_results.append(layer_preds)
            
            exp_results.append({
                "sample_id": count,
                "category": category,
                "results": sample_results
            })
            
            # Save intermediate
            with open(result_dir, "w") as f:
                json.dump(exp_results, f, indent=2)
                
        except Exception as e:
            print(f"Error processing sample {count}: {e}")
            # continue or raise based on preference. For exp, maybe just log.
            import traceback
            traceback.print_exc()

    print(f"Finished processing {len(exp_results)} samples. Results saved to {result_dir}")
