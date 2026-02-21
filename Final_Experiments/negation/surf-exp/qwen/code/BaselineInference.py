
import os
import sys
import glob
from pathlib import Path

# Get the absolute path of the current script
current_script_path = Path(__file__).parent.absolute()
# Add the Qwen2-VL directory to Python path
sys.path.append("/home/sc305/VLM/Qwen2-VL")
# Change to Qwen2-VL directory so that model/utils imports resolve consistently
os.chdir("/home/sc305/VLM/Qwen2-VL")

from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch
import json
from tqdm import tqdm

# Local utilities used throughout this project
from general_utils import ModelAndTokenizer
from patchscopes_utils import prepare_inputs  # for constructing model inputs


def load_vlm(model_path, device):
    """
    Load the Qwen2-VL model and processor, attach them to a ModelAndTokenizer helper,
    and set eval mode. Mirrors the loader used in VPatchscope.py for consistency.
    """
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)

    mt = ModelAndTokenizer(
        model_path,
        model=model,
        low_cpu_mem_usage=False,
        device=device,
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


def evaluate_response(response_text, category):
    """
    Convert a free-form model response into a binary prediction.
    Returns 1 if presence is detected, otherwise 0.
    """
    response_lower = response_text.lower().strip()
    if "yes" in response_lower and "no" not in response_lower:
        return 1
    if "no" in response_lower and "yes" not in response_lower:
        return 0
    return 1 if category.lower() in response_lower else 0


def run_baseline_inference(mt, image_path, prompt, max_new_tokens=20):
    """
    Perform a single, unpatched inference on the given image and prompt.
    Returns the decoded text response.
    """
    image = Image.open(image_path).resize((384, 384))
    inputs = prepare_inputs(prompt, image, mt.processor, mt.device)
    with torch.no_grad():
        output_ids = mt.model.generate(**inputs, max_new_tokens=max_new_tokens)
    decoded = mt.processor.batch_decode(
        [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], output_ids)],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return decoded


if __name__ == "__main__":
    # Device and model paths
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    model_path = "Qwen/Qwen2-VL-7B-Instruct"
    CURRENT_SCRIPT_PATH = current_script_path


    results_path = os.path.join(CURRENT_SCRIPT_PATH, "../results/qwen_results_baseline.json")


    # Data paths
    dataset_dir = os.path.join(CURRENT_SCRIPT_PATH, "../../test_images_split/dataset")
    test_images_dir = os.path.join(CURRENT_SCRIPT_PATH, "../../test_images_split/test_images")
    category_mapping_path = os.path.join(CURRENT_SCRIPT_PATH, "../../test_images_split/neg_object_map.json")

    # Load category mapping
    category_mapping = load_category_mapping(category_mapping_path)

    # Collect images
    image_files = []
    if os.path.exists(dataset_dir):
        image_files = glob.glob(os.path.join(dataset_dir, "*.jpg")) + glob.glob(os.path.join(dataset_dir, "*.png"))

    if not image_files:
        print(f"No images found in {dataset_dir}")
        print("Available directories:")
        for root, dirs, files in os.walk("."):
            if any(f.endswith((".jpg", ".png")) for f in files):
                count = len([f for f in files if f.endswith((".jpg", ".png"))])
                print(f"  {root}: {count} images")
        sys.exit(1)

    # Model
    mt = load_vlm(model_path, device)

    # Load existing results to support resume
    exp_results = []
    if os.path.exists(results_path):
        try:
            with open(results_path, "r") as f:
                exp_results = json.load(f)
            print(f"Loaded {len(exp_results)} existing results from {results_path}")
        except Exception as e:
            print(f"Error loading existing results: {e}")
            exp_results = []

    processed_files = set(result.get("filename") for result in exp_results)
    print(f"Already processed {len(processed_files)} images")
    remaining_images = [img for img in image_files if os.path.basename(img) not in processed_files]
    print(f"Remaining images to process: {len(remaining_images)}")

    sample_id = len(exp_results)

    for idx, img_path in enumerate(tqdm(remaining_images)):
        fname = os.path.basename(img_path)

        # Derive category from filename
        category_id = get_category_from_filename(fname)
        category = category_mapping.get(category_id, "unknown")
        prompt = f"Is there a {category} in the image? Answer yes or no."

        try:
            response_text = run_baseline_inference(mt, img_path, prompt, max_new_tokens=20)

            if idx == 0:
                print(f"DEBUG - Category: {category}")
                print(f"DEBUG - Prompt: {prompt}")
                print(f"DEBUG - Model response: '{response_text}'")

            prediction = evaluate_response(response_text, category)

            # Save per-sample result
            exp_results.append({
                "sample_id": sample_id,
                "filename": fname,
                "category": category,
                "response": response_text,
                "prediction": prediction,
            })

            # Persist after each sample to allow resume
            with open(results_path, "w") as f:
                json.dump(exp_results, f)

            sample_id += 1

        except Exception as e:
            print(f"Error processing {fname}: {e}")
            continue

    print(f"Successfully processed {len(exp_results)} samples")
    with open(results_path, "w") as f:
        json.dump(exp_results, f)


