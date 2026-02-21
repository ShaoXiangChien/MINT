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
import torch
from transformers import AutoModelForCausalLM

from deepseek_vl2.models import DeepseekVLV2Processor
from general_utils import ModelAndTokenizer
from patchscopes_utils import inspect_vision_in_lm
import json
from datasets import load_from_disk
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



if __name__ == "__main__":
    # Set device
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    result_dir = "/home/sc305/VLM/Final Experiments/global-img-ld/lg_results/ds-results.json"

    sample_trainset = load_from_disk("../data/full_sample")
    category_mapping = load_cate_mapping("../data/instances_category_map.json")

    # Load vision-language model
    mt = load_vlm(device)
    
    # Load target image
    target_image = Image.open("./images/apple.png")

    exp_results = []

    pbar = tqdm(sample_trainset)
    count = 0

    for sample in pbar:
        results = []
        category = category_mapping[str(sample["annotations"]["category_id"][0])]
        prompt = f"What is in the image? {category} or apple"

        try:
            for layer_source in range(0, 12, 2):
                layer_results = []
                for layer_target in range(0, 12, 2):
                    result = inspect_vision_in_lm(mt, prompt, sample["image"], target_image, layer_source, layer_target)
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
    
    
    