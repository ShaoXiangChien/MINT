# Synthetic Dataset Construction for Negation Finetuning

This directory contains all the scripts and template bash files required to create fine-tuning datasets for improving negation understanding in vision-language models. The process builds on the CC12M dataset, augmenting it with negated captions and multiple-choice questions to construct two datasets: **CC12M-NegCap** and **CC12M-NegMCQ**, along with their combined variant, **CC12M-NegFull**. 

The pipeline is structured into multiple steps, each performed using the Python and bash scripts provided in this directory. Example bash scripts are provided to demonstrate how these steps can be executed on nodes with 8 GPUs, but users must replace placeholder values with paths and variables suited to their specific environment.

---

## Prerequisites

### Conda Environment
The required Python environment is the same as in `synthetic_datasets/evaluation/`. Ensure you have the environment configured as follows:

```bash
conda create -n negbench python=3.10 -y
conda activate negbench
pip install vllm transformers
```

### Input Data
The process requires the **CC12M dataset** with approximately 10 million image-caption pairs in CSV format.

### Output Structure
The outputs include:
- **CC12M-NegCap**: Three negated captions per image (~30 million captions).
- **CC12M-NegMCQ**: Four captions per image (1 correct + 3 hard negatives) (~40 million captions).
- **CC12M-NegFull**: A combination of both datasets.

---

## Instructions for Running the Pipeline

Follow the steps below in sequence to generate the datasets.

### Step 1: Extract Positive and Negative Objects
Run `run_all_extraction.sh` to extract positive and negative objects using LLaMA 3.1. This script processes the input CC12M captions and outputs annotations for each image.

```bash
bash run_all_extraction.sh <NODE_NUM>
```

### Step 2: Propose Negative Objects
Run `run_node_negative.sh` to propose negative objects for each image. This script uses LLaMA 3.1 to generate objects contextually relevant but absent from the image.

```bash
bash run_node_negative.sh <NODE_NUM>
```

### Step 3: Filter Negative Objects
Run `run_node_object_filtering.sh` to verify the presence of positive objects and absence of negative objects using an open-vocabulary object detector.

```bash
bash run_node_object_filtering.sh <NODE_NUM>
```

### Step 4: Validate Object Lists
Run `run_node_validate_filtered.sh` to validate that the extracted positive and negative object lists contain only valid objects. This step ensures that the dataset has high-quality annotations.

```bash
bash run_node_validate_filtered.sh <NODE_NUM>
```

### Step 4.1: Combine Validated CSVs
Combine the validated object files into a single CSV using `combine_csv_files.py`.

```bash
python combine_csv_files.py --csv_type cc12m_images_pos_neg_validated
```

### Step 5: Generate Negated Captions
Run `run_node_caption_generation.sh` to generate captions incorporating negated objects using LLaMA 3.1. This script creates the **CC12M-NegCap** dataset.

```bash
bash run_node_caption_generation.sh <NODE_NUM>
```

### Step 6: Create Multiple-Choice Questions (MCQs)
Run `create_mcq.py` to generate MCQs with one correct caption and three hard negatives per image. This script creates the templated version of the **CC12M-NegMCQ** dataset.

```bash
python create_mcq.py --input_file /path/to/cc12m_images_pos_neg_validated.csv \
                     --output_file /path/to/final/cc12m_mcq_captions.csv
```

### Step 7: Paraphrase MCQ Captions
Run `run_node_paraphrase_mcq.sh` to paraphrase the generated MCQs, ensuring linguistic diversity in the captions. This script creates the final version of the **CC12M-NegMCQ** dataset.

```bash
bash run_node_paraphrase_mcq.sh <NODE_NUM>
```

---

## Customizing the Bash Scripts

### Placeholder Variables
The example bash scripts are designed for an 8-GPU node setup. Replace the following placeholders with your system-specific values:
- `<CLUSTER_BASE_DIR>`: Base directory where the dataset and scripts are stored.
- `<PATH_TO_CONDA>`: Path to the Conda installation.
- `<CLUSTER_TMP_DIR>`: Directory for temporary files.
- `<CLUSTER_PIP_CACHE_DIR>`: Directory for the pip cache.
- `<CLUSTER_HF_HOME>`: Directory for Hugging Face model files.
- `<HF_ACCESS_TOKEN>`: Hugging Face access token.

### Logs
Logs for each job are saved in the `logs/` directory with descriptive filenames (e.g., `extraction_<JOB_INDEX>.log`).

---

## Output Datasets
1. **CC12M-NegCap**: Located in the `cc12m_images_captioned/` directory.
2. **CC12M-NegMCQ**: Located in the `final/` directory.
3. **CC12M-NegFull**: Combine both datasets if needed for experiments.

---

## Notes
- The provided scripts are templates and may require adaptation for your specific cluster or GPU setup.
- Always validate the outputs of each step before proceeding to the next.
- Contact the authors for questions or clarifications about the dataset creation process.