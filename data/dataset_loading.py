"""Dataset loading utilities for MINT experiments.

Provides helpers to load the datasets used across experiments:
- COCO-based object detection samples (``full_sample``)
- Category mapping (``instances_category_map.json``)
- Controlled spatial-relationship image datasets
- NegBench / SURF negation benchmarks
"""

import json
from pathlib import Path
from typing import Dict, Any

from datasets import load_from_disk


def load_coco_samples(data_dir: str = "data/"):
    """Load the pre-processed COCO sample dataset.

    Expects ``data_dir/full_sample/`` to contain a HuggingFace ``Dataset``
    saved via ``dataset.save_to_disk()``.

    Returns:
        A HuggingFace ``Dataset`` object.
    """
    return load_from_disk(str(Path(data_dir) / "full_sample"))


def load_category_mapping(data_dir: str = "data/") -> Dict[str, str]:
    """Load COCO category id -> name mapping.

    Returns:
        Dict mapping category id (as string) to category name.
    """
    path = Path(data_dir) / "instances_category_map.json"
    with open(path) as f:
        return json.load(f)


def load_spatial_dataset(dataset_path: str) -> list:
    """Load the controlled spatial-relationship image dataset.

    Args:
        dataset_path: Path to ``controlled_images_dataset.json``.

    Returns:
        List of dicts, each with ``image_path`` and ``caption_options``.
    """
    with open(dataset_path) as f:
        return json.load(f)
