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


def determine_object_patches(
    bbox: list,
    orig_w: int,
    orig_h: int,
    patch_size: int = 14,
    resize: int = 336
) -> list:
    """
    Determine which image patches correspond to an object in LLaVA-1.5-7b's vision encoder.

    Args:
        bbox (list): COCO bbox format [x, y, w, h] in original image pixels.
        orig_w (int): Original image width.
        orig_h (int): Original image height.
        patch_size (int): Size of each patch (default: 14 for CLIP-ViT-L-336px).
        resize (int): Resolution to which the image is resized (default: 336).

    Returns:
        list: Indices (in the encoder output of shape [1, 577, 1024]) of patches containing the object.
    """
    # Compute scaling factors
    scale_x = resize / orig_w
    scale_y = resize / orig_h

    # Extract and scale bbox corners
    bx, by, bw, bh = bbox
    x1, y1 = bx * scale_x, by * scale_y
    x2, y2 = (bx + bw) * scale_x, (by + bh) * scale_y

    # Grid parameters
    grid_size = int(resize / patch_size)  # 336/14 = 24

    # Compute patch grid indices
    x_start = int(math.floor(x1 / patch_size))
    x_end   = int(math.floor((x2 - 1) / patch_size))
    y_start = int(math.floor(y1 / patch_size))
    y_end   = int(math.floor((y2 - 1) / patch_size))

    # Clamp to valid range [0, grid_size-1]
    x_start = max(0, min(grid_size - 1, x_start))
    x_end   = max(0, min(grid_size - 1, x_end))
    y_start = max(0, min(grid_size - 1, y_start))
    y_end   = max(0, min(grid_size - 1, y_end))

    # The class token occupies index 0; patch tokens start from index 1
    patch_indices = [
        1 + y * grid_size + x
        for y in range(y_start, y_end + 1)
        for x in range(x_start, x_end + 1)
    ]

    return patch_indices

def prepare_inputs(prompt, image, tokenizer, image_processor, model, device, model_path):
    image_sizes = [image.size]
    qs = text_prompt_to_qs(model, model_path, prompt)
    input_ids = tokenizer_image_token(qs, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).to(device)
    images_tensor = process_images([image], image_processor, model.config).to(device, dtype=torch.float16)
    
    return {"input_ids": input_ids, "images": images_tensor, "image_sizes": image_sizes}


def capture_obj_emb(inp_source, layer_source):

    image_activations = {}

    def extract_image_activations(module, input, output):
        if "layer_output" not in image_activations:
            image_activations["layer_output"] = output[0].clone()

    capture_hook = mt.model.model.vision_tower.vision_tower.vision_model.encoder.layers[layer_source].register_forward_hook(extract_image_activations)



    with torch.no_grad():
        _ = mt.model(**inp_source)

    capture_hook.remove()

    return image_activations["layer_output"]

def patch_obj(inp_target, layer_target, obj_emb, patch_indices):

    patched = False

    def patch_hs(module, input, output):
        nonlocal patched
        nonlocal obj_emb

        if patched:
            return output

        
        for idx in patch_indices:
            output[0][0][idx] = obj_emb[0][idx]
        
        patched = True

        return output

    patch_hook = mt.model.model.vision_tower.vision_tower.vision_model.encoder.layers[layer_target].register_forward_hook(patch_hs)

    with torch.no_grad():
        output_ids = mt.model.generate(
            inp_target["input_ids"],
            images=inp_target["images"],
            image_sizes=inp_target["image_sizes"],
            max_new_tokens=100,
            pad_token_id=mt.tokenizer.pad_token_id,
        )

    output = mt.tokenizer.decode(output_ids[0][1:])
    patch_hook.remove()
    return output

if __name__ == "__main__":
    # Set device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_path = "liuhaotian/llava-v1.5-7b"
    results_path = "/home/sc305/VLM/Final Experiments/obj-patching-ve/lg_results/llava_results.json"
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

        # object patching
        patch_indices = determine_object_patches(sample["annotations"]["bbox"][0], sample["image"].size[0], sample["image"].size[1])

        # prepare inputs
        if isinstance(sample["image"], Image.Image):
            sample["image"] = sample["image"].convert('RGB')

        inp_source = prepare_inputs("", sample["image"], mt.tokenizer, mt.image_processor, mt.model, device, model_path)
        inp_target = prepare_inputs(prompt, target_image.resize(sample["image"].size), mt.tokenizer, mt.image_processor, mt.model, device, model_path)


        try:
            for layer_source in range(0, 24, 2):
                layer_results = []
                obj_emb = capture_obj_emb(inp_source, layer_source)

                for layer_target in range(0, 24, 2):
                    result = patch_obj(inp_target, layer_target, obj_emb, patch_indices)
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
    
    
    