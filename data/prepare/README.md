# Data Preparation Scripts for MINT Baseline Fusion Experiment

This directory contains scripts to download and format the three datasets used in the new **Section 5.1 Baseline Fusion** experiment.

## Overview

| Dataset | Dimension | Script | Output |
| :--- | :--- | :--- | :--- |
| POPE (random split) | Object Existence | `prepare_pope.py` | `data/pope/pope_minimal_pairs.json` |
| GQA (val scene graphs) | Attribute (colour/material) | `prepare_gqa.py` | `data/gqa/gqa_attribute_minimal_pairs.json` |
| What's Up (Controlled_Images A) | Spatial Relationship | `prepare_whatsup.py` | `data/whatsup/whatsup_spatial_minimal_pairs.json` |

---

## Step-by-Step Download Instructions

### 1. COCO val2014 Images (for POPE)

POPE questions are generated from COCO val2014 images.

```bash
wget http://images.cocodataset.org/zips/val2014.zip
python data/prepare/unzip.py val2014.zip --dest /path/to/coco/
# Result: /path/to/coco/val2014/COCO_val2014_*.jpg
```

### 2. GQA Scene Graphs + Images (for GQA)

The scene graphs zip only contains two JSON files and a README — this is normal.
The images are a separate download.

```bash
# Scene graphs (~42 MB) — produces train_sceneGraphs.json + val_sceneGraphs.json
wget https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip
python data/prepare/unzip.py sceneGraphs.zip --dest /path/to/gqa/

# Images (~20 GB) — download from:
# https://cs.stanford.edu/people/dorarad/gqa/download.html
# Then extract:
python data/prepare/unzip.py images.zip --dest /path/to/gqa/
```

### 3. What's Up Controlled_Images (for What's Up)

Download **only these two files** from the [What's Up Google Drive](https://drive.google.com/drive/folders/164q6X9hrvP-QYpi3ioSnfMuyHpG5oRkZ):

| File | Size | Purpose |
| :--- | :--- | :--- |
| `controlled_images_dataset.json` | 86 KB | Index JSON with captions |
| `controlled_images.tar.gz` | 90.7 MB | The actual images |

You do **not** need `val2017.zip`, `vg_images.tar.gz`, or any other file.

```bash
# After downloading, place both files in the same directory, e.g. /path/to/whatsup/
# Then extract the images:
cd /path/to/whatsup/
tar -xzf controlled_images.tar.gz
# Result: /path/to/whatsup/controlled_images/*.jpg
```

If `tar` is not available on your server:
```bash
python data/prepare/unzip.py controlled_images.tar.gz --dest /path/to/whatsup/
# Note: unzip.py also handles .tar.gz via Python's tarfile module
```

---

## Running the Preparation Scripts

```bash
# From the MINT root directory:

# 1. POPE (Object Existence)
python data/prepare/prepare_pope.py \
    --coco_image_dir /path/to/coco/val2014 \
    --output_dir     data/pope \
    --split          random

# 2. GQA (Attribute)
python data/prepare/prepare_gqa.py \
    --scene_graphs   /path/to/gqa/val_sceneGraphs.json \
    --gqa_image_dir  /path/to/gqa/images \
    --output_dir     data/gqa \
    --max_samples    500

# 3. What's Up (Spatial)
#    --root_dir must contain both controlled_images_dataset.json
#    and the controlled_images/ folder
python data/prepare/prepare_whatsup.py \
    --root_dir   /path/to/whatsup \
    --output_dir data/whatsup
```

---

## Output Format

All three scripts produce a unified JSON format compatible with the `07_baseline_fusion` experiment:

```json
[
  {
    "image_file": "/absolute/path/to/image.jpg",
    "positive": {
      "question": "Is there a person in the image?",
      "label":    "yes"
    },
    "negative": {
      "question": "Is there a refrigerator in the image?",
      "label":    "no"
    }
  }
]
```

Each entry is a **minimal pair**: the same image with a positive question (correct answer: "yes") and a negative question (correct answer: "no"). This design ensures that any difference in the model's response is attributable solely to the visual content, not to linguistic priors.

---

## Utility: Python-based Extraction

If `unzip` or `tar` is not available on your server, use the included Python utility:

```bash
# For .zip files
python data/prepare/unzip.py file.zip --dest /output/dir/

# For .tar.gz files (also supported)
python data/prepare/unzip.py file.tar.gz --dest /output/dir/
```
