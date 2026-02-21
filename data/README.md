# Data Preparation

This directory contains scripts and utilities for preparing the datasets
used in MINT experiments.  **Large dataset files are not included in
this repository** and must be downloaded separately.

## COCO 2017 Validation Set

Used by experiments 01--04 (object patching, multimodal fusion, text patching,
global image fusion).

```bash
python -m data.download_coco --output_dir data/
```

This downloads and extracts ~5000 images to `data/val2017/`.

## Pre-processed Sample Dataset

The `full_sample/` directory should contain a HuggingFace `Dataset` object
saved via `dataset.save_to_disk()`.  This includes COCO images paired with
bounding-box annotations.  See the paper appendix for dataset construction
details.

## Spatial Relationship Dataset

The `controlled_images_dataset.json` and associated images are used by
experiment 05 (spatial reasoning).  See the experiment README for generation
instructions.

## NegBench (External)

The negation benchmarks are provided by [NegBench](https://github.com/...negbench).
Install separately:

```bash
pip install negbench  # or clone the repository
```

See experiment 06 README for details.
