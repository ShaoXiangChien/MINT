import os
import sys
from pathlib import Path

# Get the absolute path of the current script
current_script_path = Path(__file__).parent.absolute()
# Add the LLaVA directory to Python path
sys.path.append("/home/sc305/VLM/LLaVA")
# Change to LLaVA directory
os.chdir("/home/sc305/VLM/LLaVA")

from PIL import Image
import numpy as np
import torch
from llava.mm_utils import tokenizer_image_token, process_images
from llava.constants import IMAGE_TOKEN_INDEX
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from patchscopes_utils import inspect_vision_in_lm
import torch
from general_utils import ModelAndTokenizer
import math
import json
from datasets import load_from_disk
import cv2
from tqdm import tqdm
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates
import re

def load_vlm(model_path, device):
    # Load vision-language model
    torch.cuda.set_device(device)
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path),
        device=device
    )

    # 更徹底地確保所有模型組件都在正確的設備上
    model = model.to(device)
    
    # 遷移視覺模型
    if hasattr(model, 'model'):
        if hasattr(model.model, 'vision_tower'):
            # 確保主視覺塔在正確設備上
            model.model.vision_tower = model.model.vision_tower.to(device)
            
            # 遷移內部視覺塔
            if hasattr(model.model.vision_tower, 'vision_tower'):
                model.model.vision_tower.vision_tower = model.model.vision_tower.vision_tower.to(device)
                
                # 遷移視覺模型的所有子模塊
                if hasattr(model.model.vision_tower.vision_tower, 'vision_model'):
                    for module in model.model.vision_tower.vision_tower.vision_model.modules():
                        module.to(device)
                        
                # 遷移視覺投影器
                if hasattr(model.model.vision_tower, 'mm_projector'):
                    model.model.vision_tower.mm_projector = model.model.vision_tower.mm_projector.to(device)

    # 驗證設備遷移
    def verify_device(module, target_device):
        for name, child in module.named_children():
            for param in child.parameters():
                assert param.device.type == target_device.split(':')[0] and \
                       param.device.index == int(target_device.split(':')[1]), \
                       f"Parameter in {name} found on {param.device}, should be on {target_device}"
            verify_device(child, target_device)

    # 驗證所有組件都在正確的設備上
    try:
        verify_device(model, device)
    except AssertionError as e:
        print(f"Warning: Device verification failed: {e}")
        # 可以選擇是否在這裡 raise 錯誤
    
    dtype = torch.float16

    mt = ModelAndTokenizer(
        model_path,
        model=model,
        tokenizer=tokenizer,
        low_cpu_mem_usage=False,
        image_processor=image_processor,
        device=device,
        torch_dtype=dtype
    )

    mt.model.eval()
    return mt

def load_cate_mapping(mapping_path):
    with open(mapping_path, "r") as f:
        category_map = json.load(f)

    return category_map

def text_prompt_to_qs(model, model_path, prompt):
    qs = prompt
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    
    if "llama-2" in model_path.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_path.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_path.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_path.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_path.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()

if __name__ == "__main__":
    # Set device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    result_dir = "/home/sc305/VLM/Final Experiments/global-img-ld/lg_results/llava-results.json"
    model_path = "liuhaotian/llava-v1.5-7b"

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
        source_prompt = text_prompt_to_qs(mt.model, model_path, "")
        target_prompt = text_prompt_to_qs(mt.model, model_path, prompt)


        try:
            for layer_source in range(0, 32, 3):
                layer_results = []
                for layer_target in range(0, 32, 3):
                    result = inspect_vision_in_lm(mt, source_prompt, target_prompt, [sample["image"]], [target_image], layer_source, layer_target)
                    prediction = 1 if category in result.lower() else 0
                    layer_results.append(prediction)
                
                results.append(layer_results)
        except Exception as e:
            print(f"Error processing sample {count}: {e}")
            continue
        
        # save the results after each sample
        with open(result_dir, "w") as f:
            json.dump(exp_results, f)

        if len(results) != 0:
            exp_results.append({
                "sample_id": count,
                "category": category,
                "results": results
            })

        count += 1


    print(f"Satisfied samples: {len(exp_results)}")

    with open(result_dir, "w") as f:
        json.dump(exp_results, f)
    
    
    
