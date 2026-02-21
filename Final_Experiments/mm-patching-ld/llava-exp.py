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


def load_vlm(model_path, device):
    # Load vision-language model

    torch.cuda.set_device(device)

    

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path),
        device_map={"": device}
    )

    model = model.to(device)
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


def prepare_inputs(prompt, image, tokenizer, image_processor, model, device, model_path):
    image_sizes = [image.size]
    qs = text_prompt_to_qs(model, model_path, prompt)
    input_ids = tokenizer_image_token(qs, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).to(device)
    images_tensor = process_images([image], image_processor, model.config).to(device, dtype=torch.float16)
    
    return {"input_ids": input_ids, "images": images_tensor, "image_sizes": image_sizes}


def capture_obj_emb(inp_source, layer_source):

    extracted_activations = {}

    def capture_hs(module, input, output):
        nonlocal extracted_activations
        if "layer_output" not in extracted_activations:
            extracted_activations["layer_output"] = output[0].clone()

    capture_hs_hook = mt.model.model.layers[layer_source].register_forward_hook(capture_hs)

    with torch.no_grad():
        _ = mt.model(**inp_source)

    capture_hs_hook.remove()

    return extracted_activations["layer_output"]

def patch_multimodal(inp_target, layer_target, cache_hs):
    patched = False

    def patch_hs(module, input, output):
        nonlocal patched
        
        if patched:
            return output
        
        tgt_len = len(inp_target["input_ids"][0])
        img_idx = torch.where(inp_target["input_ids"][0] == -200)[0][0] 
        sep_idx = torch.where(inp_target["input_ids"][0] == 2056)[0][0]
        after_sep_len = tgt_len - sep_idx 


        for i in range(img_idx, output[0].shape[1] - after_sep_len):
            output[0][0][i] = cache_hs[0][i]


        patched = True
        

        return output

    patch_hook_handle = mt.model.model.layers[layer_target].register_forward_hook(patch_hs)
        
    with torch.no_grad():
        output_ids = mt.model.generate(
            inp_target["input_ids"],
            images=inp_target["images"],
            image_sizes=inp_target["image_sizes"],
            max_new_tokens=20,
            pad_token_id=mt.tokenizer.pad_token_id,
        )

    patch_hook_handle.remove()

    output = mt.tokenizer.decode(output_ids[0][1:])

    return output

if __name__ == "__main__":
    # Set device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_path = "liuhaotian/llava-v1.5-7b"
    results_path = "/home/sc305/VLM/Final Experiments/mm-patching-ld/lg_results/llava_results.json"
    sample_trainset = load_from_disk("../data/full_sample")
    category_mapping = load_cate_mapping("../data/instances_category_map.json")

    # Load vision-language model
    mt = load_vlm(model_path, device)

    
    # Load target image

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


        # prepare inputs
        if isinstance(sample["image"], Image.Image):
            sample["image"] = sample["image"].convert('RGB')

        inp_source = prepare_inputs(source_prompt, sample["image"], mt.tokenizer, mt.image_processor, mt.model, device, model_path)
        inp_target = prepare_inputs(merged_prompt, target_image.resize(sample["image"].size), mt.tokenizer, mt.image_processor, mt.model, device, model_path)

        # Silence source prompt in merged prompt
        img_idx = torch.where(inp_target["input_ids"][0] == -200)[0][0]
        sep_idx = torch.where(inp_target["input_ids"][0] == 2056)[0][0]
        for idx in range(img_idx + 1, sep_idx):
            inp_target["input_ids"][0][idx] = 0

        try:
            for layer_source in range(0, 24, 2):
                layer_results = []
                cache_hs = capture_obj_emb(inp_source, layer_source)

                for layer_target in range(0, 24, 2):
                    result = patch_multimodal(inp_target, layer_target, cache_hs)
                    prediction = 1 if "yes" in result.lower() else 0
                    layer_results.append(prediction)

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
    
    
    