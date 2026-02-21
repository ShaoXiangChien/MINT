import os
import sys
import glob
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch

current_script_path = Path(__file__).parent.absolute()
sys.path.append("/home/sc305/VLM/DeepSeek-VL2")

os.chdir("/home/sc305/VLM/DeepSeek-VL2")

from transformers import AutoModelForCausalLM
from deepseek_vl2.models.processing_deepseek_vl_v2 import DeepseekVLV2Processor

from general_utils import ModelAndTokenizer

from patchscopes_utils import prepare_inputs


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


def get_category_from_filename(filename):
    base = os.path.splitext(filename)[0]
    return base.split("_")[0] if "_" in base else "0"


def evaluate_response(response_text, category):
    response_lower = response_text.lower().strip()
    if "yes" in response_lower and "no" not in response_lower:
        return 1
    if "no" in response_lower and "yes" not in response_lower:
        return 0
    return int(category.lower() in response_lower)


def run_baseline_inference(mt, image_path, prompt, max_new_tokens=20):
    image = Image.open(image_path).convert("RGB").resize((384, 384))
    inputs = prepare_inputs(prompt, image, mt.processor, mt.device)

    with torch.no_grad():
        output_ids = mt.model.generate(
            inputs_embeds=mt.model.prepare_inputs_embeds(**inputs),
            attention_mask=inputs["attention_mask"],
            pad_token_id=mt.tokenizer.pad_token_id,
            eos_token_id=mt.tokenizer.eos_token_id,
            bos_token_id=mt.tokenizer.bos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # Decode only generated tokens if input_ids available
    try:
        gen_ids = output_ids[0][len(inputs["input_ids"][0]):]
    except Exception:
        gen_ids = output_ids[0]

    return mt.processor.tokenizer.decode(
        gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )


if __name__ == "__main__":
    # Device and model paths
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    model_path = "deepseek-ai/deepseek-vl2-small"

    CURRENT_SCRIPT_PATH = Path(__file__).parent.absolute()

    # Results path (create directory if needed)
    results_path = os.path.join(CURRENT_SCRIPT_PATH, "../results/deepseek_results_baseline.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    # Data paths
    dataset_dir = os.path.join(CURRENT_SCRIPT_PATH, "../../test_images_split/dataset")
    test_images_dir = os.path.join(CURRENT_SCRIPT_PATH, "../../test_images_split/test_images")
    category_mapping_path = os.path.join(CURRENT_SCRIPT_PATH, "../../test_images_split/neg_object_map.json")

    # Load category mapping
    category_mapping = load_category_mapping(category_mapping_path)

    # Collect images
    image_files = sorted(
        glob.glob(os.path.join(dataset_dir, "*.jpg")) + glob.glob(os.path.join(dataset_dir, "*.png"))
    )

    if not image_files:
        print(f"No images found in {dataset_dir}")
        sys.exit(1)

    # Model
    mt = load_vlm(device)

    # Load previous results if any
    exp_results = []
    if os.path.exists(results_path):
        try:
            with open(results_path, "r") as f:
                exp_results = json.load(f)
            print(f"Loaded {len(exp_results)} previous results from {results_path}")
        except Exception as e:
            print(f"Error loading previous results: {e}")

    processed_files = set(r.get("filename") for r in exp_results)
    remaining_images = [img for img in image_files if os.path.basename(img) not in processed_files]
    print(f"Already processed {len(processed_files)} images")
    print(f"Remaining images to process: {len(remaining_images)}")

    sample_id = len(exp_results)

    for idx, img_path in enumerate(tqdm(remaining_images, desc="Baseline DeepSeek")):
        fname = os.path.basename(img_path)
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

            exp_results.append({
                "sample_id": sample_id,
                "filename": fname,
                "category": category,
                "response": response_text,
                "prediction": prediction,
            })

            with open(results_path, "w") as f:
                json.dump(exp_results, f)

            sample_id += 1

        except Exception as e:
            print(f"Error processing {fname}: {e}")
            continue

    print(f"Finished baseline processing {len(exp_results)} samples.")
    with open(results_path, "w") as f:
        json.dump(exp_results, f)


