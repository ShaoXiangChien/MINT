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
from patchscopes_utils import inspect_vision_in_lm
import json
from tqdm import tqdm

PATH_PREFIX = "/home/sc305/VLM/Final Experiments/spatial-relationship/"

def load_vlm(model_path, device):
    # Load vision-language model

    # 預設以自動精度和設備映射加載模型
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
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

def inference(mt, prompt, image):
    inp = prepare_inputs(prompt, image.resize((384, 384)), mt.processor, mt.device)
    with torch.no_grad():
        output_ids = mt.model.generate(**inp, max_new_tokens=10)

    output = mt.processor.batch_decode(
            [out_ids[len(in_ids):] for in_ids, out_ids in zip(inp["input_ids"], output_ids)],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

    return output


if __name__ == "__main__":
    # Set device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_path = "Qwen/Qwen2-VL-7B-Instruct"

    results_path = "/home/sc305/VLM/Final Experiments/spatial-relationship/results/qwen_baseline_results.json"


    # Load controlled images dataset
    with open("/home/sc305/VLM/Final Experiments/spatial-relationship/data/controlled_images_dataset.json") as f:
        data = json.load(f)

    # Load vision-language model
    mt = load_vlm(model_path, device)
    
    exp_results = []

    pbar = tqdm(data)
    count = 0

    for item in pbar:
        source_image = Image.open(PATH_PREFIX + item["image_path"])
        target_image = Image.new('RGB', source_image.size, (0, 0, 0))
        for idx, caption in enumerate(item["caption_options"]):
            prompt_target = f"Is {caption}? only answer YES or NO"

            result = inference(mt, prompt_target, source_image)
            prediction = 1 if "yes" in result.lower() else 0
            correct = prediction == 1 if idx == 0 else prediction == 0
            
            exp_results.append({
                "image_id": count,
                "label": "yes" if idx == 0 else "no",
                "prediction": prediction,
                "correct": correct,
                "result": result
            })

        count += 1



    with open(results_path, "w") as f:
        json.dump(exp_results, f)
    
    
    