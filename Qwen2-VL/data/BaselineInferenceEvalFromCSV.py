"""
Baseline Inference Evaluation from NegBench CSV
================================================

Modified version of BaselineInferenceEval.py that reads directly from
NegBench CSV format instead of requiring renamed images and category mapping.

Supports both CSV input (new) and directory-based input (legacy).
"""

import os
import sys
import glob
import json
import ast
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Tuple

from PIL import Image
from tqdm import tqdm
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# Environment bootstrap
current_script_path = Path(__file__).parent.absolute()
sys.path.append("/home/sc305/VLM/Qwen2-VL")
os.chdir("/home/sc305/VLM/Qwen2-VL")

# Local utilities
from patchscopes_utils import prepare_inputs  # noqa: E402
from general_utils import ModelAndTokenizer  # noqa: E402


def load_vlm(model_path: str, device: str) -> ModelAndTokenizer:
	"""Load the Qwen2-VL model and processor."""
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


def parse_negative_objects(neg_objs_str):
	"""
	Parse negative_objects from CSV (could be string representation of list).
	
	Args:
		neg_objs_str: String or list from CSV column
		
	Returns:
		List of negative object names
	"""
	if pd.isna(neg_objs_str):
		return ["unknown"]
	
	# If it's already a list, return as is
	if isinstance(neg_objs_str, list):
		return neg_objs_str if neg_objs_str else ["unknown"]
	
	# Try to parse as Python list literal
	try:
		parsed = ast.literal_eval(str(neg_objs_str))
		if isinstance(parsed, list) and len(parsed) > 0:
			return parsed
	except:
		pass
	
	# Fallback: treat as single string
	return [str(neg_objs_str)] if neg_objs_str else ["unknown"]


def load_dataset_from_csv(csv_path: str, coco_base_dir: str = None):
	"""
	Load dataset from NegBench CSV format.
	
	Args:
		csv_path: Path to CSV file
		coco_base_dir: Base directory for COCO images (if filepath needs to be resolved)
		
	Returns:
		List of dicts with: image_path, category, image_id, negative_objects
	"""
	print(f"Loading dataset from CSV: {csv_path}")
	df = pd.read_csv(csv_path)
	
	print(f"   Loaded {len(df)} rows from CSV")
	
	dataset = []
	for idx, row in df.iterrows():
		# Parse negative objects
		neg_objs = parse_negative_objects(row.get('negative_objects', 'unknown'))
		category = neg_objs[0] if neg_objs else "unknown"
		
		# Get image path
		image_id = row.get('image_id', None)
		filepath = row.get('filepath', None)
		
		# Resolve image path - try multiple strategies
		img_path = None
		
		# Strategy 1: Use filepath if it's absolute and exists
		if filepath and os.path.isabs(filepath) and os.path.exists(filepath):
			img_path = filepath
		
		# Strategy 2: If filepath is relative, try to resolve it
		elif filepath and not os.path.isabs(filepath):
			# Extract filename from filepath (e.g., "data/coco/images/val2017/000000397133.jpg" -> "000000397133.jpg")
			filename = os.path.basename(filepath)
			
			# Try in coco_base_dir
			if coco_base_dir:
				candidate_path = os.path.join(coco_base_dir, filename)
				if os.path.exists(candidate_path):
					img_path = candidate_path
			
			# If still not found, try common COCO paths
			if not img_path:
				common_paths = [
					'/home/sc305/VLM/data/val2017',
					'/home/sc305/VLM/data/coco/val2017',
					'data/coco/images/val2017',
				]
				for base in common_paths:
					candidate = os.path.join(base, filename)
					if os.path.exists(candidate):
						img_path = candidate
						break
		
		# Strategy 3: Build from image_id if we have it
		if not img_path and image_id:
			# Try different filename formats
			candidates = [
				f"{image_id:012d}.jpg",  # 12-digit zero-padded
				f"{image_id:06d}.jpg",   # 6-digit zero-padded
				f"{image_id}.jpg",       # No padding
			]
			
			base_dirs = [
				coco_base_dir,
				'/home/sc305/VLM/data/val2017',
				'/home/sc305/VLM/data/coco/val2017',
			]
			
			for base_dir in base_dirs:
				if not base_dir:
					continue
				for candidate_filename in candidates:
					candidate_path = os.path.join(base_dir, candidate_filename)
					if os.path.exists(candidate_path):
						img_path = candidate_path
						break
				if img_path:
					break
		
		# If still not found, skip this row
		if not img_path:
			if idx < 10:  # Only print first 10 warnings
				print(f"   ⚠️  Warning: Row {idx} - Could not resolve image path (filepath={filepath}, image_id={image_id})")
			continue
		
		# Verify image exists
		if not os.path.exists(img_path):
			if idx < 10:
				print(f"   ⚠️  Warning: Image not found: {img_path}")
			continue
		
		dataset.append({
			'image_path': img_path,
			'category': category,
			'image_id': image_id,
			'negative_objects': neg_objs,
			'row_index': idx
		})
	
	print(f"   ✅ Successfully loaded {len(dataset)} valid entries")
	return dataset


def evaluate_response(response_text: str, category: str) -> int:
	"""Convert model's free-form text response into a binary prediction."""
	response_lower = response_text.lower().strip()
	
	if "yes" in response_lower and "no" not in response_lower:
		return 1
	if "no" in response_lower and "yes" not in response_lower:
		return 0
	
	return 1 if category.lower() in response_lower else 0


def run_single_inference(
	mt: ModelAndTokenizer,
	image_path: str,
	prompt: str,
	max_new_tokens: int = 20
) -> str:
	"""Perform inference on a single image with the given prompt."""
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
	"""Create directory if it doesn't exist."""
	if path and path.strip():  # Only create if path is non-empty
		os.makedirs(path, exist_ok=True)


def load_processed_set(results_jsonl_path: str) -> set:
	"""Load the set of already processed image paths."""
	if not os.path.exists(results_jsonl_path):
		return set()
	
	seen = set()
	with open(results_jsonl_path, "r") as f:
		for line in f:
			try:
				record = json.loads(line)
				img_path = record.get("image_path")
				if img_path:
					seen.add(img_path)
			except Exception:
				continue
	return seen


def append_jsonl(path: str, obj: dict) -> None:
	"""Append a JSON object as a new line to a JSONL file."""
	with open(path, "a") as f:
		f.write(json.dumps(obj) + "\n")


def compute_metrics(results: List[dict]) -> Dict[str, float]:
	"""Compute classification metrics from the results."""
	if not results:
		return {
			"accuracy": 0.0,
			"precision": 0.0,
			"recall": 0.0,
			"f1_score": 0.0,
			"true_positives": 0,
			"true_negatives": 0,
			"false_positives": 0,
			"false_negatives": 0,
			"total": 0,
		}
	
	tp = tn = fp = fn = 0
	
	for r in results:
		pred = r["prediction"]
		actual = r["expected_label"]
		
		if pred == 1 and actual == 1:
			tp += 1
		elif pred == 0 and actual == 0:
			tn += 1
		elif pred == 1 and actual == 0:
			fp += 1
		elif pred == 0 and actual == 1:
			fn += 1
	
	total = len(results)
	
	accuracy = (tp + tn) / total if total > 0 else 0.0
	precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
	recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
	f1_score = (
		2 * precision * recall / (precision + recall)
		if (precision + recall) > 0
		else 0.0
	)
	
	return {
		"accuracy": accuracy,
		"precision": precision,
		"recall": recall,
		"f1_score": f1_score,
		"true_positives": tp,
		"true_negatives": tn,
		"false_positives": fp,
		"false_negatives": fn,
		"total": total,
	}


def print_metrics(metrics: Dict[str, float]) -> None:
	"""Pretty-print the evaluation metrics."""
	print("\n" + "="*60)
	print("EVALUATION RESULTS")
	print("="*60)
	print(f"Total Samples:       {metrics['total']}")
	print(f"\nAccuracy:            {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
	print(f"Precision:           {metrics['precision']:.4f}")
	print(f"Recall:              {metrics['recall']:.4f}")
	print(f"F1 Score:            {metrics['f1_score']:.4f}")
	print(f"\nConfusion Matrix:")
	print(f"  True Positives:    {metrics['true_positives']}")
	print(f"  True Negatives:    {metrics['true_negatives']}")
	print(f"  False Positives:   {metrics['false_positives']}")
	print(f"  False Negatives:   {metrics['false_negatives']}")
	print("="*60 + "\n")


def main(
	model_path: str,
	device: str,
	csv_file: str,
	coco_base_dir: str,
	results_jsonl_path: str,
	metrics_json_path: str,
	max_new_tokens: int,
	expected_label: int,
) -> None:
	"""
	Main function to run baseline inference from CSV.
	
	Args:
		model_path: Path or HF ID of the model to evaluate
		device: Device to run on
		csv_file: Path to NegBench CSV file
		coco_base_dir: Base directory for COCO images
		results_jsonl_path: Where to save per-sample results
		metrics_json_path: Where to save overall metrics
		max_new_tokens: Max tokens to generate per sample
		expected_label: Ground truth label (0 or 1) for the dataset
	"""
	print("="*80)
	print("BASELINE INFERENCE EVALUATION FROM CSV")
	print("="*80)
	print(f"Model: {model_path}")
	print(f"Device: {device}")
	print(f"CSV file: {csv_file}")
	print(f"COCO base dir: {coco_base_dir}")
	
	# Load dataset from CSV
	dataset = load_dataset_from_csv(csv_file, coco_base_dir)
	
	if not dataset:
		print("❌ ERROR: No valid images found in CSV!")
		return
	
	# Load the VLM model
	print(f"\nLoading model from {model_path}...")
	mt = load_vlm(model_path, device)
	print("✅ Model loaded successfully")
	
	# Resume support: check which images have already been processed
	processed = load_processed_set(results_jsonl_path)
	print(f"\nAlready processed {len(processed)} images")
	to_process = [d for d in dataset if d['image_path'] not in processed]
	print(f"Remaining images to process: {len(to_process)}")
	
	# Ensure output directories exist
	ensure_dir(os.path.dirname(results_jsonl_path))
	ensure_dir(os.path.dirname(metrics_json_path))
	
	# Initialize sample counter
	sample_id = len(processed)
	
	# Process each image
	print("\n" + "="*80)
	print("Running inference...")
	print("="*80)
	
	for item in tqdm(to_process, desc="Processing images"):
		img_path = item['image_path']
		category = item['category']
		
		try:
			# Build the prompt
			prompt = f"Is there a {category} in the image? Answer yes or no."
			
			# Run model inference
			response_text = run_single_inference(mt, img_path, prompt, max_new_tokens=max_new_tokens)
			
			# Convert response to binary prediction
			prediction = evaluate_response(response_text, category)
			
			# Save detailed result
			result = {
				"sample_id": sample_id,
				"filename": os.path.basename(img_path),
				"image_path": img_path,
				"image_id": item.get('image_id'),
				"category": category,
				"negative_objects": item.get('negative_objects', []),
				"prompt": prompt,
				"response": response_text,
				"prediction": prediction,
				"expected_label": expected_label,
				"correct": int(prediction == expected_label),
			}
			append_jsonl(results_jsonl_path, result)
			
			sample_id += 1
			
		except Exception as e:
			print(f"\n⚠️  Error processing {os.path.basename(img_path)}: {e}")
			continue
	
	# Load all results and compute metrics
	print("\n" + "="*80)
	print("Computing metrics...")
	print("="*80)
	
	all_results = []
	with open(results_jsonl_path, "r") as f:
		for line in f:
			try:
				all_results.append(json.loads(line))
			except Exception:
				continue
	
	metrics = compute_metrics(all_results)
	
	# Save metrics to JSON file
	with open(metrics_json_path, "w") as f:
		json.dump(metrics, f, indent=2)
	
	print(f"\n✅ Metrics saved to: {metrics_json_path}")
	print(f"✅ Detailed results saved to: {results_jsonl_path}")
	
	# Print metrics to console
	print_metrics(metrics)


if __name__ == "__main__":
	import argparse
	
	default_device = "cuda:1" if torch.cuda.is_available() else "cpu"
	default_model_path = "Qwen/Qwen2-VL-7B-Instruct"
	
	parser = argparse.ArgumentParser(
		description="Run baseline inference and evaluation from NegBench CSV."
	)
	parser.add_argument(
		"--model_path",
		type=str,
		default=default_model_path,
		help="HuggingFace model ID or local path to model"
	)
	parser.add_argument(
		"--device",
		type=str,
		default=default_device,
		help="Device to run on (e.g., 'cuda:0', 'cuda:1', or 'cpu')"
	)
	parser.add_argument(
		"--csv_file",
		type=str,
		required=True,
		help="Path to NegBench CSV file"
	)
	parser.add_argument(
		"--coco_base_dir",
		type=str,
		default="/home/sc305/VLM/data/val2017",
		help="Base directory for COCO images"
	)
	parser.add_argument(
		"--results_jsonl",
		type=str,
		default="./baseline_results.jsonl",
		help="Path to save per-sample results (JSONL format)"
	)
	parser.add_argument(
		"--metrics_json",
		type=str,
		default="./baseline_metrics.json",
		help="Path to save evaluation metrics (JSON format)"
	)
	parser.add_argument(
		"--max_new_tokens",
		type=int,
		default=20,
		help="Maximum number of tokens to generate per response"
	)
	parser.add_argument(
		"--expected_label",
		type=int,
		choices=[0, 1],
		default=0,
		help="Ground truth label for the dataset (0=No/absent, 1=Yes/present)"
	)
	
	args = parser.parse_args()
	
	# Run the main evaluation
	main(
		model_path=args.model_path,
		device=args.device,
		csv_file=args.csv_file,
		coco_base_dir=args.coco_base_dir,
		results_jsonl_path=args.results_jsonl,
		metrics_json_path=args.metrics_json,
		max_new_tokens=args.max_new_tokens,
		expected_label=args.expected_label,
	)

