import os
import sys
from pathlib import Path

# Get the absolute path of the current script
current_script_path = Path(__file__).parent.absolute()
# Add the LLaVA directory to Python path
sys.path.append("/home/sc305/VLM/DeepSeek-VL2")
# Change to LLaVA directory
os.chdir("/home/sc305/VLM/DeepSeek-VL2")

import torch
from transformers import AutoModelForCausalLM

from deepseek_vl2.models import DeepseekVLV2Processor
from general_utils import ModelAndTokenizer, prepare_inputs
import json
from datasets import load_from_disk
from tqdm import tqdm
from PIL import Image

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
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    mt.model = mt.model.to(mt.device)
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

    capture_hook = mt.model.language.model.layers[layer_source].register_forward_hook(extract_image_activations)


    with torch.no_grad():
        _ = mt.model(
            input_ids=inp_source["input_ids"],
            attention_mask=inp_source["attention_mask"],
            output_hidden_states=True,
            use_cache=True
        )

    capture_hook.remove()

    return image_activations["layer_output"]

def patch_text(mt, inp_target, layer_target, image_activations):

    patched = False

    def patch_hs(module, input, output):
        nonlocal patched

        if patched:
            return output
        
        img_start_idx = torch.where(inp_target["input_ids"][0] == 128815)[0][0]
        sep_idx = torch.where(inp_target["input_ids"][0] == 3749)[0][0]

        for idx in range(img_start_idx, sep_idx):
            output[0][0][idx] = image_activations[0][idx]

        
        patched = True

        return output

    patch_hook = mt.model.language.model.layers[layer_target].register_forward_hook(patch_hs)

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
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    result_dir = "/home/sc305/VLM/Final Experiments/mm-patching-ld/lg_results/ds-results.json"

    sample_trainset = load_from_disk("../data/full_sample")
    category_mapping = load_cate_mapping("../data/instances_category_map.json")

    # load saved results
    if os.path.exists(result_dir):
        with open(result_dir, "r") as f:
            exp_results = json.load(f)
    else:
        exp_results = []
    
    processed_samples = [result["sample_id"] for result in exp_results]

    # Load vision-language model
    mt = load_vlm(device)
    

    pbar = tqdm(sample_trainset)
    count = 0

    for sample in pbar:
        if count in processed_samples:
            count += 1
            continue

        results = []
        category = category_mapping[str(sample["annotations"]["category_id"][0])]
        source_prompt = f"The main object is not a {category}"
        target_prompt = f"Is the main object in the image a {category}? only answer with yes or no"

        merged_prompt = f"{source_prompt} ; {target_prompt}"


        target_image = Image.new("RGB", sample["image"].size, (0, 0, 0))
        try:

            inp_source = prepare_inputs(source_prompt, sample["image"], mt.processor, mt.device)
            inp_target = prepare_inputs(merged_prompt, target_image, mt.processor, mt.device)


            for layer_source in range(0, 12, 2):
                layer_results = []
                cache_hs = capture_emb(mt, inp_source, layer_source)

                for layer_target in range(0, 12, 2):
                    result = patch_text(mt, inp_target, layer_target, cache_hs)
                    prediction = 1 if "yes" in result.lower() else 0
                    layer_results.append(prediction)
                
                results.append(layer_results)
        except Exception as e:
            print(f"Error processing sample {count}: {e}")
            continue
        
        # save the results after each sample
        if count % 100 == 0:
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
    
    
    