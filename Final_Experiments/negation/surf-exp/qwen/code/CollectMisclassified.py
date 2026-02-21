import os
import sys
import glob
import json
import shutil
from pathlib import Path
from typing import Optional, Dict

from PIL import Image
from tqdm import tqdm
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# --------------------------------------------------------------------------------
# Environment bootstrap (kept consistent with BaselineInference.py)
# --------------------------------------------------------------------------------
current_script_path = Path(__file__).parent.absolute()
sys.path.append("/home/sc305/VLM/Qwen2-VL")
os.chdir("/home/sc305/VLM/Qwen2-VL")

# Local utilities used throughout this project
from patchscopes_utils import prepare_inputs  # noqa: E402
from general_utils import ModelAndTokenizer  # noqa: E402


def load_vlm(model_path: str, device: str) -> ModelAndTokenizer:
	"""
	Load the Qwen2-VL model and processor, attach them to a ModelAndTokenizer helper,
	and set eval mode. Mirrors the loader used in BaselineInference.py for consistency.
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


def load_category_mapping(mapping_path: str) -> Dict[str, str]:
	with open(mapping_path, "r") as f:
		category_mapping = json.load(f)
	return category_mapping


def get_category_from_filename(filename: str) -> str:
	"""Extract category ID from filename like '1000_000000508602.jpg'"""
	base = os.path.splitext(filename)[0]
	category_id = base.split("_")[0] if "_" in base else "0"
	return category_id


def evaluate_response(response_text: str, category: str) -> int:
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


def run_single_inference(mt: ModelAndTokenizer, image_path: str, prompt: str, max_new_tokens: int = 20) -> str:
	"""
	Perform a single inference on the given image and prompt.
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


def ensure_dir(path: str) -> None:
	os.makedirs(path, exist_ok=True)


def load_processed_set(results_jsonl_path: str) -> set:
	"""
	Read a JSONL file of results and return the set of filenames already recorded.
	If file does not exist, return empty set.
	"""
	if not os.path.exists(results_jsonl_path):
		return set()
	seen = set()
	with open(results_jsonl_path, "r") as f:
		for line in f:
			try:
				record = json.loads(line)
				fname = record.get("filename")
				if fname:
					seen.add(fname)
			except Exception:
				continue
	return seen


def append_jsonl(path: str, obj: dict) -> None:
	with open(path, "a") as f:
		f.write(json.dumps(obj) + "\n")


def build_sft_example(image_path: str, prompt: str, expected_label: int) -> dict:
	"""
	Construct a simple SFT-style example for Qwen2-VL where the assistant provides the correct answer.
	The expected text is 'Yes.' for 1 and 'No.' for 0.
	"""
	expected_text = "Yes." if expected_label == 1 else "No."
	return {
		"messages": [
			{
				"role": "user",
				"content": [
					{"type": "image", "image": image_path},
					{"type": "text", "text": prompt},
				],
			},
			{
				"role": "assistant",
				"content": [
					{"type": "text", "text": expected_text}
				],
			},
		]
	}


def main(
	model_path: str,
	device: str,
	dataset_dir: str,
	category_mapping_path: str,
	results_jsonl_path: str,
	errors_dir: str,
	errors_jsonl_path: str,
	errors_sft_jsonl_path: Optional[str],
	max_new_tokens: int,
	expected_label: int,
) -> None:
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
		return

	# Load model
	mt = load_vlm(model_path, device)

	# Resume support
	processed = load_processed_set(results_jsonl_path)
	print(f"Already processed {len(processed)} images")
	to_process = [p for p in image_files if os.path.basename(p) not in processed]
	print(f"Remaining images to process: {len(to_process)}")

	# Ensure output dirs/files
	ensure_dir(os.path.dirname(results_jsonl_path))
	ensure_dir(errors_dir)
	if errors_sft_jsonl_path:
		ensure_dir(os.path.dirname(errors_sft_jsonl_path))
	ensure_dir(os.path.dirname(errors_jsonl_path))

	sample_id = len(processed)

	for idx, img_path in enumerate(tqdm(to_process)):
		fname = os.path.basename(img_path)
		try:
			# Derive category from filename and build prompt
			category_id = get_category_from_filename(fname)
			category = category_mapping.get(category_id, "unknown")
			prompt = f"Is there a {category} in the image? Answer yes or no."

			response_text = run_single_inference(mt, img_path, prompt, max_new_tokens=max_new_tokens)
			prediction = evaluate_response(response_text, category)

			# Persist per-sample result (for resume and auditing)
			append_jsonl(results_jsonl_path, {
				"sample_id": sample_id,
				"filename": fname,
				"image_path": img_path,
				"category_id": category_id,
				"category": category,
				"prompt": prompt,
				"response": response_text,
				"prediction": prediction,
				"expected_label": expected_label,
				"correct": int(prediction == expected_label),
			})

			# If incorrect, copy image and log to errors JSONL (and optional SFT JSONL)
			if prediction != expected_label:
				dest_path = os.path.join(errors_dir, fname)
				try:
					shutil.copy2(img_path, dest_path)
				except Exception as copy_err:
					print(f"Warning: failed to copy {img_path} -> {dest_path}: {copy_err}")
					dest_path = img_path  # fall back to original path

				error_record = {
					"sample_id": sample_id,
					"filename": fname,
					"image_path": dest_path,
					"category_id": category_id,
					"category": category,
					"prompt": prompt,
					"response": response_text,
					"prediction": prediction,
					"expected_label": expected_label,
				}
				append_jsonl(errors_jsonl_path, error_record)

				if errors_sft_jsonl_path:
					sft_example = build_sft_example(dest_path, prompt, expected_label)
					append_jsonl(errors_sft_jsonl_path, sft_example)

			sample_id += 1

		except Exception as e:
			print(f"Error processing {fname}: {e}")
			continue

	print("Done. Summary:")
	# Quick counts
	total = len(processed) + len(to_process)
	try:
		incorrect = 0
		if os.path.exists(errors_jsonl_path):
			with open(errors_jsonl_path, "r") as f:
				for _ in f:
					incorrect += 1
		print(f"  Total seen: {total}")
		print(f"  Incorrect saved: {incorrect}")
		print(f"  Errors dir: {errors_dir}")
		print(f"  Errors JSONL: {errors_jsonl_path}")
		if errors_sft_jsonl_path:
			print(f"  SFT JSONL: {errors_sft_jsonl_path}")
	except Exception:
		pass


if __name__ == "__main__":
	# Defaults aligned with BaselineInference.py and negation setup
	default_device = "cuda:1" if torch.cuda.is_available() else "cpu"
	default_model_path = "Qwen/Qwen2-VL-7B-Instruct"
	CURRENT_SCRIPT_PATH = current_script_path

	default_dataset_dir = os.path.join(CURRENT_SCRIPT_PATH, "../../test_images_split/dataset")
	default_category_mapping_path = os.path.join(CURRENT_SCRIPT_PATH, "../../test_images_split/neg_object_map.json")

	default_results_jsonl = os.path.join(CURRENT_SCRIPT_PATH, "../results/qwen_results_collect_all.jsonl")
	default_errors_dir = os.path.join(CURRENT_SCRIPT_PATH, "../results/misclassified_images")
	default_errors_jsonl = os.path.join(CURRENT_SCRIPT_PATH, "../results/qwen_misclassified.jsonl")
	default_errors_sft_jsonl = os.path.join(CURRENT_SCRIPT_PATH, "../results/qwen_misclassified_sft.jsonl")

	# For the negation experiment, the correct answer is typically "No" (0)
	default_expected_label = 0

	import argparse
	parser = argparse.ArgumentParser(description="Run inference and collect misclassified samples for fine-tuning.")
	parser.add_argument("--model_path", type=str, default=default_model_path, help="HF model id or local path")
	parser.add_argument("--device", type=str, default=default_device, help="Device, e.g., 'cuda:0' or 'cpu'")
	parser.add_argument("--dataset_dir", type=str, default=default_dataset_dir, help="Directory with .jpg/.png images")
	parser.add_argument("--category_mapping_path", type=str, default=default_category_mapping_path, help="Path to neg_object_map.json")
	parser.add_argument("--results_jsonl", type=str, default=default_results_jsonl, help="Path to write per-sample results (JSONL)")
	parser.add_argument("--errors_dir", type=str, default=default_errors_dir, help="Directory where misclassified images are copied")
	parser.add_argument("--errors_jsonl", type=str, default=default_errors_jsonl, help="Path to write misclassified samples (JSONL)")
	parser.add_argument("--errors_sft_jsonl", type=str, default=default_errors_sft_jsonl, help="Optional: write SFT-formatted JSONL for fine-tuning")
	parser.add_argument("--max_new_tokens", type=int, default=20, help="Max new tokens for generation")
	parser.add_argument("--expected_label", type=int, choices=[0, 1], default=default_expected_label, help="Ground-truth label for the dataset: 0 = No, 1 = Yes")
	args = parser.parse_args()

	main(
		model_path=args.model_path,
		device=args.device,
		dataset_dir=args.dataset_dir,
		category_mapping_path=args.category_mapping_path,
		results_jsonl_path=args.results_jsonl,
		errors_dir=args.errors_dir,
		errors_jsonl_path=args.errors_jsonl,
		errors_sft_jsonl_path=args.errors_sft_jsonl,
		max_new_tokens=args.max_new_tokens,
		expected_label=args.expected_label,
	)


