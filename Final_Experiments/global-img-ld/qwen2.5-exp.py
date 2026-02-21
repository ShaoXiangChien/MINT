import os
import sys
from pathlib import Path

# Get the absolute path of the current script
current_script_path = Path(__file__).parent.absolute()
# Add the LLaVA directory to Python path
sys.path.append("/home/sc305/VLM/Qwen2-VL")
# Change to LLaVA directory
os.chdir("/home/sc305/VLM/Qwen2-VL")


from PIL import Image
import numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
from general_utils import ModelAndTokenizer
from patchscopes_utils import inspect_vision_in_lm
import math
import json
from datasets import load_from_disk
from tqdm import tqdm

import re



def load_vlm(model_path, device):
    # Load vision-language model

    # 預設以自動精度和設備映射加載模型
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)


    mt = ModelAndTokenizer(
        model_path,
        model=model,
        low_cpu_mem_usage=False,
        device=device
    )

    mt.processor = processor
    mt.model.eval()
    return mt

def load_cate_mapping(mapping_path):
    with open(mapping_path, "r") as f:
        category_map = json.load(f)

    return category_map



if __name__ == "__main__":
    # Set device
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    model_path = "Qwen/Qwen2.5-VL-7B-Instruct"

    results_path = "/home/sc305/VLM/Final Experiments/global-img-ld/lg_results/qwen2.5_results.json"


    sample_trainset = load_from_disk("../data/full_sample")
    category_mapping = load_cate_mapping("../data/instances_category_map.json")

    # Load vision-language model
    mt = load_vlm(model_path, device)
    
    # Load target image
    target_image = Image.open("../DeepSeek-VL2/images/apple.png")

    exp_results = []

    pbar = tqdm(sample_trainset)
    count = 0

    for sample in pbar:
        
        results = []
        category = category_mapping[str(sample["annotations"]["category_id"][0])]
        prompt = f"What is in the image? {category} or apple"

        try:
            for layer_source in range(0, 28, 3):
                layer_results = []
                for layer_target in range(0, 28, 3):
                    result = inspect_vision_in_lm(mt, prompt, sample["image"], target_image, layer_source, layer_target)
                    prediction = 1 if category in result.lower() else 0
                    layer_results.append(prediction)
                
                results.append(layer_results)
        except Exception as e:
            print(f"Error processing sample {count}: {e}")
            count += 1
            continue
        
        # Append results first (only if results is not empty)
        if len(results) != 0:
            exp_results.append({
                "sample_id": count,
                "category": category,
                "results": results
            })
        
        count += 1
        
        # Save every 100 samples
        if count % 100 == 0:
            print(f"\nSaving checkpoint at {count} samples...")
            with open(results_path, "w") as f:
                json.dump(exp_results, f, indent=2)
            print(f"Saved {len(exp_results)} results so far.")


    print(f"\nFinished processing all samples!")
    print(f"Satisfied samples: {len(exp_results)}")

    # Final save with all results
    print(f"Saving final results to {results_path}...")
    with open(results_path, "w") as f:
        json.dump(exp_results, f, indent=2)
    
    
    