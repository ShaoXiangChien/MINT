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
        
        # src_replace_start = torch.where(inp_source["input_ids"][0] == 344)[0][0]
        # tgt_replace_start = torch.where(inp_target["input_ids"][0] == 344)[0][0]
        start_idx = 3
        n = len(inp_target["input_ids"][0])

        for i in range(n - start_idx):
            output[0][0][start_idx + i] = image_activations[0][-17 + i]

        
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

    result_dir = "/home/sc305/VLM/Final Experiments/text-patching-ld/results/ds-results.json"

    sample_trainset = load_from_disk("../data/full_sample")
    category_mapping = load_cate_mapping("../data/instances_category_map.json")

    # Load vision-language model
    mt = load_vlm(device)
    
    # Load target image
    target_image = None

    exp_results = []

    pbar = tqdm(sample_trainset)
    count = 0

    for sample in pbar:
        try:
            results = []
            category = category_mapping[str(sample["annotations"]["category_id"][0])]
            prompt = f"Is there a {category} in the image? only answer with yes or no"

            # Ensure source image is in RGB to avoid PIL color mode issues
            source_image = sample.get("image", None) if hasattr(sample, "get") else sample["image"]
            if isinstance(source_image, Image.Image) and source_image.mode != "RGB":
                source_image = source_image.convert("RGB")

            inp_source = prepare_inputs(prompt, source_image, mt.processor, mt.device)
            inp_target = prepare_inputs(prompt, target_image, mt.processor, mt.device)

            for layer_source in range(0, 12, 2):
                layer_results = []
                for layer_target in range(0, 12, 2):
                    cache_hs = capture_emb(mt, inp_source, layer_source)
                    result = patch_text(mt, inp_target, layer_target, cache_hs)
                    prediction = 1 if category in result.lower() else 0
                    layer_results.append(prediction)
                results.append(layer_results)

            if len(results) != 0:
                exp_results.append({
                    "sample_id": count,
                    "category": category,
                    "results": results
                })

            # save the results after appending current sample
            with open(result_dir, "w") as f:
                json.dump(exp_results, f)

        except Exception as e:
            print(f"Error processing sample {count}: {e}")
        finally:
            count += 1


    print(f"Satisfied samples: {len(exp_results)}")

    with open(result_dir, "w") as f:
        json.dump(exp_results, f)
    
    
    