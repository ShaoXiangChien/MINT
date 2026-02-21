#!/usr/bin/env python3
"""
Plot average accuracy per source layer for MM-Patching experiments.

This script reads per-sample square matrices from ./lg_results/*.json, computes
the average matrix per model, then plots the mean across target layers for each
source layer as a single line per model. Outputs are saved to ./plots/.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


# Matplotlib appearance
plt.rcParams['font.size'] = 18
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['figure.dpi'] = 100


def load_avg_matrix(file_path: Path) -> Optional[np.ndarray]:
    """Load a list of per-sample result matrices and return their average.

    Returns None if the file does not exist or has no valid samples.
    """
    if not file_path.exists():
        return None

    with open(file_path, "r") as f:
        samples = json.load(f)

    if not samples:
        return None

    matrices: List[np.ndarray] = [np.array(sample["results"]) for sample in samples]
    return np.mean(matrices, axis=0)


def collect_models(results_dir: Path) -> Dict[str, Path]:
    """Map model display names to their expected results JSON paths."""
    return {
        "DeepSeek-VL2": results_dir / "ds-results.json",
        "LLaVA-1.5": results_dir / "llava_results.json",
        "QWEN2-VL": results_dir / "qwen_results.json",
    }


# Explicit source-layer indices (x-axis positions) per model
# DeepSeek-VL2: 0,2,4,6,8,10 (len=6)
# LLaVA-1.5: 0,2,4,...,22 (len=12)
# QWEN2-VL: 0,3,6,...,27 (len=10)
SOURCE_LAYER_INDICES: Dict[str, List[int]] = {
    "DeepSeek-VL2": list(range(0, 12, 2)),
    "LLaVA-1.5": list(range(0, 24, 2)),
    "QWEN2-VL": list(range(0, 28, 3)),
}


def compute_source_layer_series(avg_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Compute mean and std across target layers for each source layer.

    Returns (mean_per_source, std_per_source, layer_indices)
    """
    size = avg_matrix.shape[0]
    layers = list(range(size))
    mean_per_source = np.mean(avg_matrix, axis=1)
    std_per_source = np.std(avg_matrix, axis=1)
    return mean_per_source, std_per_source, layers


def plot_source_layer_avg(models_to_paths: Dict[str, Path], save_dir: Path) -> None:
    save_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    any_plotted = False
    for model_name, json_path in models_to_paths.items():
        avg_matrix = load_avg_matrix(json_path)
        if avg_matrix is None:
            print(f"Warning: skipping {model_name}; no data at {json_path}")
            continue

        mean_s, std_s, _layers = compute_source_layer_series(avg_matrix)

        planned_indices = SOURCE_LAYER_INDICES.get(model_name, list(range(len(mean_s))))
        if len(planned_indices) != len(mean_s):
            print(
                f"Warning: {model_name} source length ({len(mean_s)}) != planned indices ({len(planned_indices)}). Aligning to min length."
            )
            new_len = min(len(planned_indices), len(mean_s))
            planned_indices = planned_indices[:new_len]
            mean_s = mean_s[:new_len]
            std_s = std_s[:new_len]

        ax.plot(planned_indices, mean_s, marker="o", linewidth=2.5, markersize=7, label=model_name)
        ax.fill_between(planned_indices, mean_s - std_s / 2, mean_s + std_s / 2, alpha=0.2)
        any_plotted = True

    ax.set_xlabel("Source Layer Index")
    ax.set_ylabel("Average Accuracy")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    if any_plotted:
        ax.legend()

    plt.tight_layout()

    png_path = save_dir / "source_layer_avg_only.png"
    pdf_path = save_dir / "source_layer_avg_only.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

    plt.show()


def main() -> None:
    project_root = Path(__file__).parent
    results_dir = project_root / "lg_results"
    plots_dir = project_root / "plots"

    models = collect_models(results_dir)
    plot_source_layer_avg(models, plots_dir)


if __name__ == "__main__":
    main()


