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


def determine_object_patches(
    bbox: list,
    orig_w: int,
    orig_h: int,
    grid_h: int,
    grid_w: int,
    patch_size: int = 14,
) -> list:
    """
    根據已知的 grid_h、grid_w（從 patch_embed 輸出取到）來計算 bbox 佔哪些 patch。

    Args:
        bbox (list): COCO bbox [x, y, w, h]，都是原始圖像 (orig_w×orig_h) 的像素。
        orig_w, orig_h (int): 原始圖寬高。
        grid_h, grid_w (int): patch_embed.proj 輸出時的網格高度與寬度。
        resize_w, resize_h (int): 模型內部實際將圖縮放到的寬高（與抓到的 grid_h/grid_w 對應）。
        patch_size (int): patch 的空間尺寸（通常是 14）。
    Returns:
        list: 所有包含物件的 patch token index（+1 抵掉 class token）。
    """
    # 1. 尺度換算：將原始 bbox 映射到 resize_w×resize_h
    bx, by, bw, bh = bbox
    resize_w = grid_w * patch_size
    resize_h = grid_h * patch_size
    scale_x = resize_w / orig_w
    scale_y = resize_h / orig_h

    x1, y1 = bx * scale_x, by * scale_y
    x2, y2 = (bx + bw) * scale_x, (by + bh) * scale_y

    # 2. floor 除以 patch_size 得到格子座標
    x_start = int(math.floor(x1 / patch_size))
    x_end   = int(math.floor((x2 - 1) / patch_size))
    y_start = int(math.floor(y1 / patch_size))
    y_end   = int(math.floor((y2 - 1) / patch_size))

    # 3. clamp 至合法範圍
    x_start = max(0, min(grid_w - 1, x_start))
    x_end   = max(0, min(grid_w - 1, x_end))
    y_start = max(0, min(grid_h - 1, y_start))
    y_end   = max(0, min(grid_h - 1, y_end))

    # 4. class token 在最前面，所以 patch token 從 index 1 開始
    patch_indices = [
        1 + y * grid_w + x
        for y in range(y_start, y_end + 1)
        for x in range(x_start, x_end + 1)
    ]
    return patch_indices

def capture_obj_emb(inp_source, layer_source):
    image_activations = {}

    def extract_image_activations(module, input, output):
        if "layer_output" not in image_activations:
            image_activations["layer_output"] = output.clone()

    capture_hook = mt.model.visual.blocks[layer_source].register_forward_hook(extract_image_activations)

    with torch.no_grad():
        _ = mt.model(**inp_source)

    capture_hook.remove()

    return image_activations["layer_output"]

def patch_obj(inp_target, layer_target, obj_emb, patch_indices):

    patched = False

    def patch_hs(module, input, output):
        nonlocal patched
        nonlocal obj_emb
        nonlocal patch_indices

        if patched:
            return output

        
        for idx in patch_indices:
            output[idx - 1] = obj_emb[idx - 1]
        
        patched = True

        return output

    patch_hook = mt.model.visual.blocks[layer_target].register_forward_hook(patch_hs)
    output_ids = mt.model.generate(
                **inp_target,
                max_new_tokens=5
            )
    output = mt.processor.batch_decode(
                [out_ids[len(in_ids):] for in_ids, out_ids in zip(inp_target["input_ids"], output_ids)],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
    
    patch_hook.remove()
    return output

if __name__ == "__main__":
    # Set device
    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    model_path = "Qwen/Qwen2-VL-7B-Instruct"
    results_path = "/home/sc305/VLM/Final Experiments/obj-patching-ve/lg_results/qwen_results.json"
    sample_trainset = load_from_disk("../data/full_sample")
    category_mapping = load_cate_mapping("../data/instances_category_map.json")

    # Load vision-language model
    mt = load_vlm(model_path, device)
    
    # Load target image
    target_image = Image.open("../DeepSeek-VL2/images/room.png")

    exp_results = []

    pbar = tqdm(sample_trainset)
    count = 0

    for sample in pbar:
        results = []
        category = category_mapping[str(sample["annotations"]["category_id"][0])]
        prompt = f"Is there a {category} in the image? Only answer with yes or no."


        # prepare inputs
        if isinstance(sample["image"], Image.Image):
            sample["image"] = sample["image"].convert('RGB')

        inp_source = prepare_inputs("", sample["image"], mt.processor, mt.device)
        inp_target = prepare_inputs(prompt, target_image.resize(sample["image"].size), mt.processor, mt.device)

        # object patching
        _, grid_h, grid_w = inp_source["image_grid_thw"][0]
        patch_indices = determine_object_patches(sample["annotations"]["bbox"][0], sample["image"].size[0], sample["image"].size[1], grid_h, grid_w)

        try:
            for layer_source in range(0, 32, 3):
                layer_results = []
                obj_emb = capture_obj_emb(inp_source, layer_source)

                for layer_target in range(0, 32, 3):
                    result = patch_obj(inp_target, layer_target, obj_emb, patch_indices)
                    prediction = 1 if "yes" in result.lower() else 0
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
    
    
    