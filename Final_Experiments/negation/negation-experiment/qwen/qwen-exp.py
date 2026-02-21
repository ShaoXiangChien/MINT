import os
import sys
from pathlib import Path


from PIL import Image
import numpy as np
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
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
    model = Qwen2VLForConditionalGeneration.from_pretrained(
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
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_path = "Qwen/Qwen2-VL-7B-Instruct"

    results_path = "./qwen_results.json"


    sample_trainset = load_from_disk("/home/sc305/VLM/data/full_sample")
    category_mapping = load_cate_mapping("/home/sc305/VLM/data/instances_category_map.json")

    # Load vision-language model
    mt = load_vlm(model_path, device)
    
    # Load target image
    target_image = Image.open("./apple.png")

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
            continue
        
        # save the results after each sample
        with open(results_path, "w") as f:
            json.dump(exp_results, f)

        if len(results) != 0:
            exp_results.append({
                "sample_id": count,
                "category": category,
                "results": results
            })

        count += 1


    print(f"Satisfied samples: {len(exp_results)}")

    with open(results_path, "w") as f:
        json.dump(exp_results, f)
    
    
    
