import os
import sys
from pathlib import Path

# Get the absolute path of the current script
current_script_path = Path(__file__).parent.absolute()
# Add the LLaVA directory to Python path
sys.path.append("/home/sc305/VLM/DeepSeek-VL2")
# Change to LLaVA directory
os.chdir("/home/sc305/VLM/DeepSeek-VL2")

from PIL import Image
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from pycocotools import mask as mask_utils

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images
from general_utils import ModelAndTokenizer, prepare_inputs
from patchscopes_utils import inspect_vision
import json
from datasets import load_from_disk
import cv2
from tqdm import tqdm



def load_vlm(device):
    # Load vision-language model
    model_path = "deepseek-ai/deepseek-vl2-tiny"
    vl_chat_processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    mt = ModelAndTokenizer(
        "deepseek-ai/deepseek-vl2-7b-tiny",
        model=vl_gpt,
        tokenizer=tokenizer,
        processor=vl_chat_processor,
        low_cpu_mem_usage=False,
        device=device
    )
    mt.model = mt.model.to(mt.device)
    return mt

def load_cate_mapping(mapping_path):
    with open(mapping_path, "r") as f:
        category_map = json.load(f)

    return category_map


def determine_object_patches(
    bbox, W, H,
    resize_longest=480,
    crop_size=384,
    grid_size=27
):
    """
    Args:
        bbox: [x, y, w, h] in 原始影像座標 (COCO format)
        W, H: 原始影像寬高
        resize_longest: ResizeLongestSide 大小 (預設 480)
        crop_size: FiveCrop 的 crop_size (預設 384)
        grid_size: crop_size 被切成 grid_size x grid_size patches (預設 27)
    Returns:

    """
    # 1. 計算 ResizeLongestSide 後的尺寸 (保留長寬比)
    if W >= H:
        new_w = resize_longest
        new_h = int(H * resize_longest / W)
    else:
        new_h = resize_longest
        new_w = int(W * resize_longest / H)

    # 2. 根據 scale 把 bbox 映射到 resized 尺寸
    scale_x = new_w / W
    scale_y = new_h / H
    bx, by, bw, bh = bbox
    x1s = bx * scale_x
    y1s = by * scale_y
    x2s = (bx + bw) * scale_x
    y2s = (by + bh) * scale_y

    # 3. 計算五個 crop 在 resized 影像上的 offset
    dx = new_w - crop_size
    dy = new_h - crop_size
    offsets = [
        (0, 0),            # 左上
        (dx, 0),           # 右上
        (0, dy),           # 左下
        (dx, dy),          # 右下
        (dx//2, dy//2),    # 中央
    ]

    patch_size = crop_size / grid_size
    results = {}

    # 4. 對每個 crop 計算 local bbox 並轉成 patch index
    for idx, (off_x, off_y) in enumerate(offsets):
        # 4.1 物件在這張 crop 的 local bbox
        lx1 = max(x1s - off_x, 0)
        ly1 = max(y1s - off_y, 0)
        lx2 = min(x2s - off_x, crop_size)
        ly2 = min(y2s - off_y, crop_size)

        # 若 bbox 與 crop 沒交集就跳過
        if lx2 <= lx1 or ly2 <= ly1:
            continue

        # 4.2 計算 patch 範圍 (floor)
        x_start = int(np.floor(lx1 / patch_size))
        x_end   = int(np.floor((lx2 - 1) / patch_size))
        y_start = int(np.floor(ly1 / patch_size))
        y_end   = int(np.floor((ly2 - 1) / patch_size))

        # 4.3 clamp 到 [0, grid_size-1]
        x_start = max(0, min(grid_size-1, x_start))
        x_end   = max(0, min(grid_size-1, x_end))
        y_start = max(0, min(grid_size-1, y_start))
        y_end   = max(0, min(grid_size-1, y_end))

        # 4.4 產生 patch index list
        patch_idxs = [
            y * grid_size + x
            for y in range(y_start, y_end+1)
            for x in range(x_start, x_end+1)
        ]
        results[idx] = patch_idxs

    return results

def capture_obj_emb(inp_source, layer_source):

    image_activations = {}

    def extract_image_activations(module, input, output):
        if "layer_output" not in image_activations:
            image_activations["layer_output"] = output.clone()

    capture_hook = mt.model.vision.blocks[layer_source].register_forward_hook(extract_image_activations)



    with torch.no_grad():
        _ = mt.model.generate(
            inputs_embeds=mt.model.prepare_inputs_embeds(**inp_source),
            attention_mask=inp_source["attention_mask"],
            pad_token_id=mt.tokenizer.pad_token_id,
            eos_token_id=mt.tokenizer.eos_token_id,
            bos_token_id=mt.tokenizer.bos_token_id,
            max_new_tokens=1,
            do_sample=False,
        )

    capture_hook.remove()

    return image_activations["layer_output"]

def patch_obj(inp_target, layer_target, image_activations, patch_map):

    patched = False

    def patch_hs(module, input, output):
        nonlocal patched

        if patched:
            return output

        
        for crop_i, idxs in patch_map.items():
            for id in idxs:
                output[crop_i][id] = image_activations[crop_i][id]
        
        patched = True

        return output

    patch_hook = mt.model.vision.blocks[layer_target].register_forward_hook(patch_hs)

    with torch.no_grad():
        output_ids = mt.model.generate(
            inputs_embeds=mt.model.prepare_inputs_embeds(**inp_target),
            attention_mask=inp_target["attention_mask"],
            pad_token_id=mt.tokenizer.pad_token_id,
            eos_token_id=mt.tokenizer.eos_token_id,
            bos_token_id=mt.tokenizer.bos_token_id,
            max_new_tokens=10,
            do_sample=False
        )

    output = mt.processor.tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    patch_hook.remove()
    return output

if __name__ == "__main__":
    # Set device
    device = "cuda:4" if torch.cuda.is_available() else "cpu"
    sample_trainset = load_from_disk("../data/full_sample")
    category_mapping = load_cate_mapping("../data/instances_category_map.json")
    results_path = "/home/sc305/VLM/Final Experiments/obj-patching-ve/lg_results/ds_results.json"
    # Load vision-language model
    mt = load_vlm(device)
    
    # Load target image
    target_image = Image.open("./images/room.png")

    exp_results = []

    pbar = tqdm(sample_trainset)
    count = 0

    for sample in pbar:
        results = []
        category = category_mapping[str(sample["annotations"]["category_id"][0])]
        prompt = f"Is there a {category} in the image? Only answer with yes or no."

        # object patching
        patch_indices = determine_object_patches(sample["annotations"]["bbox"][0], sample["image"].size[0], sample["image"].size[1])

        # prepare inputs
        if isinstance(sample["image"], Image.Image):
            sample["image"] = sample["image"].convert('RGB')

        inp_source = prepare_inputs("", sample["image"], mt.processor, mt.device)
        inp_target = prepare_inputs(prompt, target_image.resize(sample["image"].size), mt.processor, mt.device)


        try:
            for layer_source in range(0, 27, 3):
                layer_results = []
                edge_case = False
                obj_emb = capture_obj_emb(inp_source, layer_source)

                for layer_target in range(0, 27, 3):
                    if obj_emb.shape[0] != 5:
                        edge_case = True
                        break
                    result = patch_obj(inp_target, layer_target, obj_emb, patch_indices)
                    prediction = 1 if "yes" in result.lower() else 0
                    layer_results.append(prediction)
                
                if edge_case:
                    break
                
                results.append(layer_results)
        except Exception as e:
            print(f"Error processing sample {count}: {e}")
            continue
        
        if count % 100 == 0:
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
    
    
    
