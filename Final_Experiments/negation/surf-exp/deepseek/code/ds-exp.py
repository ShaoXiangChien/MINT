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

PATH_PREFIX = "/home/sc305/VLM/Final Experiments/spatial-relationship/"

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

    results_path = "/home/sc305/VLM/Final Experiments/spatial-relationship/results/ds_results.json"


    # Load controlled images dataset
    with open("/home/sc305/VLM/Final Experiments/spatial-relationship/data/controlled_images_dataset.json") as f:
        data = json.load(f)

    # Load vision-language model
    mt = load_vlm(device)
    
    exp_results = []

    pbar = tqdm(data)
    count = 0

    for item in pbar:
        source_image = Image.open(PATH_PREFIX + item["image_path"])
        target_image = Image.new('RGB', source_image.size, (0, 0, 0))
        for idx, caption in enumerate(item["caption_options"]):
            prompt_target = f"Is {caption}? only answer YES or NO"
            results = []

            try:
                for layer_source in range(0, 12):
                    layer_results = []
                    for layer_target in range(0, 12):
                        result = inspect_vision_in_lm(mt, prompt_target, source_image, target_image, layer_source, layer_target)
                        prediction = 1 if "yes" in result.lower() else 0
                        correct = prediction == 1 if idx == 0 else prediction == 0
                        layer_results.append({
                            "layer_source": layer_source,
                            "layer_target": layer_target,
                            "result": result,
                            "correct": correct,
                            "prediction": prediction
                        })
                    
                    results.append(layer_results)
            except Exception as e:
                print(f"Error processing sample {count}: {e}")
                continue
            
            exp_results.append({
                "image_id": count,
                "label": "yes" if idx == 0 else "no",
                "results": results
            })

        count += 1



    with open(results_path, "w") as f:
        json.dump(exp_results, f)
    
    
    
    