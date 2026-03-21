# Data Preparation Scripts for MINT Baseline Fusion Experiment

This directory contains scripts to download and format the three datasets used in the new **Section 5.1 Baseline Fusion** experiment.

## Overview

| Dataset | Dimension | Script | Output |
| :--- | :--- | :--- | :--- |
| POPE (random split) | Object Existence | `prepare_pope.py` | `data/pope/pope_minimal_pairs.json` |
| GQA (val scene graphs) | Attribute (colour/material) | `prepare_gqa.py` | `data/gqa/gqa_attribute_minimal_pairs.json` |
| What's Up (Controlled_Images A) | Spatial Relationship | `prepare_whatsup.py` | `data/whatsup/whatsup_spatial_minimal_pairs.json` |

---

## Prerequisites

You will need to download the following external data manually before running the scripts:

### 1. COCO val2014 Images (for POPE)
```bash
# ~6 GB download
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip -d /path/to/coco/
```

### 2. GQA Scene Graphs + Images (for GQA)
```bash
# Scene graphs (~42 MB)
wget https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip
unzip sceneGraphs.zip  # produces val_sceneGraphs.json

# Images (~20 GB, only needed for the images referenced in val_sceneGraphs.json)
# Download from: https://cs.stanford.edu/people/dorarad/gqa/download.html
```

### 3. What's Up Controlled Images (for What's Up)
```bash
git clone https://github.com/amitakamath/whatsup_vlms.git
cd whatsup_vlms
# Then follow their README to download the data via Google Drive
# The JSON index files will be downloaded automatically by our script if not present
```

---

## Running the Scripts

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
python data/prepare/prepare_whatsup.py \
    --data_dir   /path/to/whatsup_vlms/data \
    --output_dir data/whatsup \
    --subset     A
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
