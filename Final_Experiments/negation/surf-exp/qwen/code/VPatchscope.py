import os
import sys
import glob
from pathlib import Path

# Get the absolute path of the current script
current_script_path = Path(__file__).parent.absolute()
# Add the LLaVA directory to Python path
sys.path.append("/home/sc305/VLM/Qwen2-VL")
# Change to LLaVA directory
os.chdir("/home/sc305/VLM/Qwen2-VL")



from PIL import Image
import numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
from general_utils import ModelAndTokenizer
from patchscopes_utils import inspect_vision_in_lm
import json
from tqdm import tqdm


def load_vlm(model_path, device):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
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


def load_category_mapping(mapping_path):
    with open(mapping_path, "r") as f:
        category_mapping = json.load(f)
    return category_mapping


def get_category_from_filename(filename):
    """Extract category ID from filename like '1000_000000508602.jpg'"""
    base = os.path.splitext(filename)[0]
    category_id = base.split("_")[0] if "_" in base else "0"
    return category_id


def evaluate_response(response, category):
    """Evaluate if the response indicates presence (yes) or absence (no)"""
    response_lower = response.lower().strip()
    
    # Check for explicit yes/no answers
    if "yes" in response_lower and "no" not in response_lower:
        return 1  # Presence detected
    elif "no" in response_lower and "yes" not in response_lower:
        return 0  # Absence detected
    else:
        # Fallback: check if category is mentioned positively
        if category.lower() in response_lower:
            return 1
        else:
            return 0





if __name__ == "__main__":
    # Device and model paths
    device = "cuda:7" if torch.cuda.is_available() else "cpu"
    model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    results_path = "./qwen2.5_results_tmp.json"

    CURRENT_SCRIPT_PATH = current_script_path

    # Data paths
    dataset_dir = os.path.join(CURRENT_SCRIPT_PATH, "../../test_images_split/dataset")  # Directory with source images
    test_images_dir = os.path.join(CURRENT_SCRIPT_PATH, "../../test_images_split/test_images")  # Directory with target images
    
    category_mapping_path = os.path.join(CURRENT_SCRIPT_PATH, "../../test_images_split/neg_object_map.json")
    # Load category mapping
    category_mapping = load_category_mapping(category_mapping_path)
    
    # Get all image files from dataset directory
    image_files = []
    if os.path.exists(dataset_dir):
        image_files = glob.glob(os.path.join(dataset_dir, "*.jpg")) + glob.glob(os.path.join(dataset_dir, "*.png"))
    
    if not image_files:
        print(f"No images found in {dataset_dir}")
        print("Available directories:")
        for root, dirs, files in os.walk("."):
            if any(f.endswith(('.jpg', '.png')) for f in files):
                print(f"  {root}: {len([f for f in files if f.endswith(('.jpg', '.png'))])} images")
        exit(1)

    # Model
    mt = load_vlm(model_path, device)
    target_image = Image.open(os.path.join(CURRENT_SCRIPT_PATH, "../../test_images_split/test_images/blank_384x384.png")).resize((384, 384))

    # Load existing results if available
    exp_results = []
    if os.path.exists(results_path):
        try:
            with open(results_path, "r") as f:
                exp_results = json.load(f)
            print(f"Loaded {len(exp_results)} existing results from {results_path}")
        except Exception as e:
            print(f"Error loading existing results: {e}")
            exp_results = []
    
    # Get list of already processed filenames
    processed_files = set([result['filename'] for result in exp_results])
    print(f"Already processed {len(processed_files)} images")
    
    # Filter out already processed images
    remaining_images = [img for img in image_files if os.path.basename(img) not in processed_files]
    print(f"Remaining images to process: {len(remaining_images)}")
    
    sample_id = len(exp_results)
    
    pbar = tqdm(remaining_images)

    for img_path in pbar:
        print(f"Processing {img_path}")
        # Extract filename from path
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
            # Load source image from file path
            source_image = Image.open(img_path).resize((384, 384))
            
            # Layer sweep for patching
            layer_results = []
            for layer_source in range(0, 28, 3):
                source_layer_results = []
                for layer_target in range(0, 28, 3):
                    result = inspect_vision_in_lm(
                        mt,
                        prompt_target=prompt_target,
                        source_image=source_image,  # Source image from dataset
                        target_image=target_image,  # Blank target image
                        layer_source=layer_source,
                        layer_target=layer_target,
                        max_gen_len=20
                    )
                    
                    # Debug: Print the actual response
                    if layer_source == 0 and layer_target == 0:
                        print(f"DEBUG - Category: {category}")
                        print(f"DEBUG - Prompt: {prompt_target}")
                        print(f"DEBUG - Model response: '{result}'")
                        print(f"DEBUG - Response contains '{category}': {category in result.lower()}")
                    
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

            # Save results for this sample
            exp_results.append({
                "sample_id": sample_id,
                "filename": fname,
                "category": category,
                "results": layer_results
            })

            # Save after each sample
            with open(results_path, "w") as f:
                json.dump(exp_results, f)
            
            sample_id += 1 

        except Exception as e:
            print(f"Error processing {fname}: {e}")
            continue

    print(f"Successfully processed {len(exp_results)} samples")
    
    # Final save
    with open(results_path, "w") as f:
        json.dump(exp_results, f)
