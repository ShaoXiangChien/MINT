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
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch
from general_utils import ModelAndTokenizer, prepare_inputs
import math
import json
from datasets import load_from_disk
from tqdm import tqdm

import re



def load_vlm(model_path, device):
    # Load vision-language model

    device = "cuda:0" if torch.cuda.is_available() else "cpu"


    # 預設以自動精度和設備映射加載模型
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map=device
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

def capture_emb(mt, inp_source, layer_source):

    image_activations = {}

    def extract_image_activations(module, input, output):
        if "layer_output" not in image_activations:
            image_activations["layer_output"] = output[0].clone()

    capture_hook = mt.model.model.layers[layer_source].register_forward_hook(extract_image_activations)


    with torch.no_grad():
        _ = mt.model(**inp_source)

    capture_hook.remove()

    return image_activations["layer_output"]

def patch_text(mt, inp_target, layer_target, image_activations):

    patched = False

    def patch_hs(module, input, output):
        nonlocal patched

        if patched:
            return output
        
        img_start_idx = torch.where(inp_target["input_ids"][0] == 151652)[0]
        sep_idx = torch.where(inp_target["input_ids"][0] == 2587)[0]

        for idx in range(img_start_idx, sep_idx):
            output[0][0][idx] = image_activations[0][idx]

        
        patched = True

        return output

    patch_hook = mt.model.model.layers[layer_target].register_forward_hook(patch_hs)

    with torch.no_grad():
        output_ids = mt.model.generate(**inp_target, max_new_tokens=10)

    patch_hook.remove()

    output = mt.processor.batch_decode(
            [out_ids[len(in_ids):] for in_ids, out_ids in zip(inp_target["input_ids"], output_ids)],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
    return output


if __name__ == "__main__":
    # Set device use CUDA_VISIBLE_DEVICES=4 in command
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_path = "Qwen/Qwen2-VL-7B-Instruct"

    results_path = "/home/sc305/VLM/Final Experiments/mm-patching-ld/lg_results/qwen_results.json"


    sample_trainset = load_from_disk("../data/full_sample")
    category_mapping = load_cate_mapping("../data/instances_category_map.json")

    # Load vision-language model
    mt = load_vlm(model_path, device)
    
    # create an empty image

    exp_results = []

    pbar = tqdm(sample_trainset)
    count = 0

    for sample in pbar:
        results = []
        category = category_mapping[str(sample["annotations"]["category_id"][0])]
        source_prompt = f"The main object is not a {category}"
        target_prompt = f"Is the main object in the image a {category}? only answer with yes or no"

        merged_prompt = f"{source_prompt} ; {target_prompt}"

        target_image = Image.new("RGB", sample["image"].size, (0, 0, 0))

        inp_source = prepare_inputs(source_prompt, sample["image"], mt.processor, mt.device)
        inp_target = prepare_inputs(merged_prompt, target_image, mt.processor, mt.device)

        # silence source prompt segment
        img_end_idx = torch.where(inp_target["input_ids"][0] == 151653)[0]
        sep_idx = torch.where(inp_target["input_ids"][0] == 2587)[0]

        for idx in range(img_end_idx + 1, sep_idx):
            inp_target["input_ids"][0][idx] = 0

        try:
            for layer_source in range(0, 28, 3):
                layer_results = []
                cache_hs = capture_emb(mt, inp_source, layer_source)

                for layer_target in range(0, 28, 3):
                    result = patch_text(mt, inp_target, layer_target, cache_hs)
                    prediction = 1 if "yes" in result.lower() else 0
                    layer_results.append(prediction)
                
                results.append(layer_results)
        except Exception as e:
            print(f"Error processing sample {count}: {e}")
            continue
        
        # if count % 100 == 0:
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
    
    
    