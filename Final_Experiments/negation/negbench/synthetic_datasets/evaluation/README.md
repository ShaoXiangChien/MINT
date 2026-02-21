# Synthetic Dataset Construction for NegBench Evaluation

This folder contains scripts for generating evaluation datasets required by NegBench, as described in Section 4 of the paper. The goal is to create synthetic datasets with negation-specific captions and multiple-choice questions (MCQs) to evaluate vision-language models' understanding of negation.

---

## Overview of Steps

1. **Process Captions to Extract Positive and Negative Concepts (Images and Videos)**  
   - For images: Use `process_caption_objects.py` to identify objects/concepts present in the images (`positive objects`) and propose related `negative objects`.  
   - For videos: Use `process_video_tasks.py` with the `concepts` task to generate two positive concepts and one negative concept.

2. **Verify Negative Objects (Supported for Images Only)**  
   Use `filter_negative_objects.py` to verify that proposed `negative objects` are not detected in the images.

3. **Create MCQs**  
   - Use `create_mcq.py` to generate multiple-choice questions for images or videos, including Affirmation, Negation, and Hybrid templates.

4. **Paraphrase Captions (Images and Videos)**  
   - Use `paraphrase_captions.py` for images.  
   - Use `process_video_tasks.py` with the `retrieval` or `mcq` task for videos.

---

## Prerequisites

### 1. Environment Setup
The following libraries are required:
- `vllm`
- `transformers`

Create and activate the `negbench` conda environment:
```bash
conda create -n negbench python=3.10 -y
conda activate negbench
pip install vllm transformers
```

### 2. Input CSV File
The input CSV file must contain:
- `filepath`: Path to the image or video.
- `caption`: Descriptive captions of the image or video content.

For video tasks, additional columns include:
- `all_captions`: A list of captions describing the video (in string format).

---

## Step-by-Step Instructions

### 1. Extract Positive and Negative Objects
#### For Images
Run `process_caption_objects.py` to process captions and extract `positive objects` and propose related `negative objects`.

```bash
python process_caption_objects.py \
    --input_file path/to/input.csv \
    --output_base path/to/output_base \
    --task_type extraction  # For positive objects
```

Then, run the script again to propose `negative objects`:
```bash
python process_caption_objects.py \
    --input_file path/to/output_base_extracted.csv \
    --output_base path/to/output_base_negatives \
    --task_type negative
```

**Output:**  
A CSV file with columns:
- `filepath`
- `caption`
- `positive_objects`
- `negative_objects`

#### For Videos
Run `process_video_tasks.py` with the `concepts` task to extract positive and negative concepts and generate negated captions for retrieval tasks.

```bash
python process_video_tasks.py \
    --input_file path/to/video_input.csv \
    --output_base path/to/output_base_concepts \
    --task concepts \
    --model mixtral
```

**Output:**  
A CSV file with columns:
- `positive_concept1`
- `positive_concept2`
- `negative_concept`
- `captions` (with negated concepts added)

---

### 2. Verify Negative Objects (Images Only)
Run `filter_negative_objects.py` to validate that proposed `negative objects` are absent from the images using an open-vocabulary object detector.

```bash
python filter_negative_objects.py \
    --input_file path/to/output_base_negatives.csv \
    --output_file path/to/filtered_output.csv
```

**Output:**  
A CSV file with verified `negative_objects`.

---

### 3. Generate MCQs
Run `create_mcq.py` to create multiple-choice questions based on verified `positive_objects` and `negative_objects`.

#### For Images
```bash
python create_mcq.py \
    --task image \
    --input_file path/to/filtered_output.csv \
    --output_file path/to/mcq_output.csv
```

#### For Videos
```bash
python create_mcq.py \
    --task video \
    --input_file path/to/video_concepts_output.csv \
    --output_file path/to/video_mcq_output.csv
```

**Output:**  
A CSV file with columns:
- `correct_answer`: Index of the correct caption.
- `caption_0` to `caption_3`: Four answer choices.
- `correct_answer_template`: The template used (Affirmation, Negation, or Hybrid).
- `image_path`: Path to the image or video.

---

### 4. Paraphrase Captions
#### For Images
Run `paraphrase_captions.py` to generate paraphrased captions for linguistic diversity.

##### For Retrieval Captions
```bash
python paraphrase_captions.py \
    --model mixtral \
    --task retrieval \
    --input_file path/to/retrieval_input.csv
```

##### For MCQ Captions
```bash
python paraphrase_captions.py \
    --model llama3.1 \
    --task mcq \
    --input_file path/to/mcq_output.csv
```

Optional: Add `--use_affirmation_negation_guideline` to preserve affirmation/negation order in captions.

#### For Videos
Run `process_video_tasks.py` with the `retrieval` or `mcq` task.

##### Retrieval Task
```bash
python process_video_tasks.py \
    --input_file path/to/video_retrieval_templated.csv \
    --output_base path/to/video_retrieval_rephrased \
    --task retrieval \
    --use_affirmation_negation_guideline
```

##### MCQ Task
```bash
python process_video_tasks.py \
    --input_file path/to/video_mcq_templated.csv \
    --output_base path/to/video_mcq_rephrased \
    --task mcq \
    --model llama3.1
```

---

## Notes

1. **File Naming Conventions**  
   - Outputs from each script should use descriptive names to reflect their processing stage.
   - For example:
     - `captions_extracted_pos.csv` for positive objects.
     - `captions_filtered_neg.csv` for verified negative objects.
     - `mcq_output.csv` for multiple-choice questions.

2. **LLM Configuration**  
   - The scripts support `mixtral` and `llama3.1` for paraphrasing.
   - Ensure sufficient GPU resources are available for running LLMs.

3. **Chunk Processing**  
   - Adjust `index_start` and `index_end` arguments to process files in chunks if required.

---

For further details, refer to the comments in each script.