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
import glob

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

def load_category_mapping(mapping_path):
    with open(mapping_path, "r") as f:
        return json.load(f)


def evaluate_response(response, category):
    """Evaluate if the response correctly identifies the category."""
    response_lower = response.lower().strip()
    if "yes" in response_lower and "no" not in response_lower:
        return 1
    elif "no" in response_lower and "yes" not in response_lower:
        return 0
    else:
        return int(category.lower() in response_lower)


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    CURRENT_SCRIPT_PATH = current_script_path
    results_path = os.path.join(CURRENT_SCRIPT_PATH, "./deepseek_results.json")

    dataset_dir = os.path.join(CURRENT_SCRIPT_PATH, "../../test_images_split/dataset")
    category_mapping = load_category_mapping(os.path.join(CURRENT_SCRIPT_PATH, "../../test_images_split/neg_object_map.json"))

    image_files = sorted(glob.glob(os.path.join(dataset_dir, "*.jpg")) + glob.glob(os.path.join(dataset_dir, "*.png")))

    if not image_files:
        print(f"No images found in {dataset_dir}")
        sys.exit(1)

    mt = load_vlm(device)
    
    # Use a different image as target instead of blank image
    # Find an image that doesn't have the same category as the source
    target_image = Image.new("RGB", (384, 384), (0, 0, 0))
    
    # Load previous results
    exp_results = []
    if os.path.exists(results_path):
        try:
            with open(results_path, "r") as f:
                exp_results = json.load(f)
            print(f"Loaded {len(exp_results)} previous results.")
        except Exception as e:
            print(f"Error loading previous results: {e}")

    processed_files = set([r["filename"] for r in exp_results])
    remaining_images = [img for img in image_files if os.path.basename(img) not in processed_files]

    sample_id = len(exp_results)

    for img_path in tqdm(remaining_images, desc="Processing"):
        fname = os.path.basename(img_path)
        
        # Extract category from filename (assuming format like "01_000000108864.jpg")
        # Get the number part before underscore
        base_name = os.path.splitext(fname)[0]
        if '_' in base_name:
            category_id = base_name.split('_')[0]
        else:
            category_id = "0"  # Default category
        
        category = category_mapping.get(category_id, "unknown")
        prompt_target = f"Is there a {category} in the image? Answer yes or no."

        try:
            source_image = Image.open(img_path).convert("RGB").resize((384, 384))
            layer_results = []

            for layer_source in range(0, 12):
                source_layer_results = []
                for layer_target in range(0, 12):
                            
                    result = inspect_vision_in_lm(
                        mt,
                        prompt_target=prompt_target,
                        source_image=source_image,  # Use object image as source
                        target_image=target_image,  # Use different image as target
                        layer_source=layer_source,
                        layer_target=layer_target,
                        max_gen_len=20,
                        verbose=False
                    )

                    # # Debug: Print the actual response
                    # if layer_source == 0 and layer_target == 0:
                    #     print(f"DEBUG - Category: {category}")
                    #     print(f"DEBUG - Prompt: {prompt_target}")
                    #     print(f"DEBUG - Model response: '{result}'")
                    #     print(f"DEBUG - Response contains '{category}': {category in result.lower()}")
                    
                    # Check if response indicates presence (yes) or absence (no)
                    response_lower = result.lower().strip()
                    if "yes" in response_lower and "no" not in response_lower:
                        prediction = 1  # Presence detected
                    elif "no" in response_lower and "yes" not in response_lower:
                        prediction = 0  # Absence detected
                    else:
                        # Fallback: check if category is mentioned positively
                        prediction = 1 if category in response_lower else 0
                    source_layer_results.append(prediction)

                layer_results.append(source_layer_results)

            exp_results.append({
                "sample_id": sample_id,
                "filename": fname,
                "category": category,
                "results": layer_results
            })
            sample_id += 1

            with open(results_path, "w") as f:
                json.dump(exp_results, f, indent=2)

        except Exception as e:
            print(f"Error processing {fname}: {e}")

    print(f"Finished processing {len(exp_results)} samples.")
    with open(results_path, "w") as f:
        json.dump(exp_results, f, indent=2)
