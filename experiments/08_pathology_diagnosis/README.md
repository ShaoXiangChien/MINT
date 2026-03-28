# Pathology Diagnosis (Section 5.2)

This directory contains the code for Section 5.2 of the MINT framework: **Pathology (Localizing the Roots of Multimodal Failures)**.

The goal of this experiment is to prove that when modern Vision-Language Models (VLMs) fail on hard tasks, the failure is often due to bottlenecks in their "Fusion Band" (e.g., Late Activation for spatial reasoning, or Early Prior Override for hallucinations).

We use causal hidden-state patching to diagnose this:
1. **Target Run (The Failure):** `[Image] + [Tricky/Adversarial Question]`. The model relies on language priors or fails to process 3D spatial relations, resulting in an incorrect answer.
2. **Source Run (The Cure):** `[Image] + "" (Empty Text Prompt)`. The model extracts pure, unpoisoned visual features without being biased by the tricky question.
3. **Intervention:** We patch the hidden states from the Source Run into the Target Run, layer by layer, and record the "Flip Rate" (when the model's incorrect answer flips to the correct one).

## Datasets

We utilize two State-of-the-Art (SOTA) datasets for this diagnosis:
1. **NaturalBench (NeurIPS 2024):** For diagnosing hallucinations and prior override. We focus on adversarial pairs where language priors contradict the visual evidence.
2. **MINDCUBE (ICCV 2025):** For diagnosing spatial blindness and late activation in 3D mental modeling.

## Step-by-Step Setup Guide

### 1. Environment Setup

Ensure you have installed the necessary dependencies for your target models. The new dataset preparation script for NaturalBench requires the Hugging Face `datasets` library.

```bash
pip install datasets
```

### 2. Download and Prepare NaturalBench

The `prepare_naturalbench.py` script automatically downloads the dataset from Hugging Face and formats it into a flat JSON list.

```bash
python data/prepare/prepare_naturalbench.py \
    --output_dir data/naturalbench
```

This will create `data/naturalbench/naturalbench_pathology.json` and download the corresponding images to `data/naturalbench/images/`.

### 3. Download and Prepare MINDCUBE

For MINDCUBE, you first need to download the raw data from Hugging Face using their CLI, then extract it.

```bash
# Download the raw dataset
huggingface-cli download Inevitablevalor/MindCube data.zip --repo-type dataset
unzip data.zip -d data/mindcube_raw

# Run the preparation script
python data/prepare/prepare_mindcube.py \
    --data_dir data/mindcube_raw/data \
    --output_dir data/mindcube
```

This will create `data/mindcube/mindcube_pathology.json`.

### 4. Expected Data Structure

Both preparation scripts output a flat JSON list where each item follows this schema:

```json
[
  {
    "image_file": "/absolute/path/to/image.jpg",
    "target_question": "Is the person holding the bat without swinging?",
    "expected_answer": "Yes",
    "meta": {
      "index": 42,
      "task_type": "spatial_reasoning"
    }
  }
]
```

### 5. Running the Experiment

Run the patching sweep using the `run_experiment.py` script. You can specify the model and the prepared dataset path.

**Evaluating Qwen2.5-VL on NaturalBench:**

```bash
python -m experiments.08_pathology_diagnosis.run_experiment \
    --model qwen25 \
    --device cuda:0 \
    --dataset_path data/naturalbench/naturalbench_pathology.json \
    --output results/08_pathology/qwen25_naturalbench.json \
    --max_samples 200
```

**Evaluating LLaVA-OneVision on MINDCUBE:**

```bash
python -m experiments.08_pathology_diagnosis.run_experiment \
    --model llava_onevision \
    --device cuda:0 \
    --dataset_path data/mindcube/mindcube_pathology.json \
    --output results/08_pathology/llava_onevision_mindcube.json \
    --max_samples 200
```

### Testing the Pipeline

You can use the `--test` flag to run a quick verification with just 2 samples and a larger layer step:

```bash
python -m experiments.08_pathology_diagnosis.run_experiment \
    --model qwen25 \
    --dataset_path data/naturalbench/naturalbench_pathology.json \
    --output results/08_pathology/test_run.json \
    --test
```
