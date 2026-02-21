#!/usr/bin/env python3
"""
Plot Average Accuracy per Target Layer (Single Figure)

This script loads experimental results from `lg_results/` and generates
one line plot that shows the average accuracy per target layer across
all source layers, for each available model.

Usage: python plot_target_layer_average.py
Outputs:
 - plots/target_layer_average_only.png
 - plots/target_layer_average_only.pdf
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Matplotlib aesthetics similar to quick_analysis
plt.rcParams['font.size'] = 18
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['figure.dpi'] = 100

MODELS = {
    'DeepSeek-VL2': {'file': 'ds-results.json', 'layers': list(range(0, 12, 2))},
    'LLaVA-1.5': {'file': 'llava-results.json', 'layers': list(range(0, 32, 3))},
    'Qwen2-VL': {'file': 'qwen_results.json', 'layers': list(range(0, 28, 3))},
    'Qwen2.5-VL': {'file': 'qwen2.5_results.json', 'layers': list(range(0, 28, 3))},
}


def load_experimental_avg_matrices(results_dir: Path):
    """Load experimental matrices and compute per-model average matrix.

    Returns dict: { model_name: { 'avg_matrix': np.ndarray, 'layers': list[int], 'samples': int } }
    """
    data = {}
    for model_name, info in MODELS.items():
        file_path = results_dir / info['file']
        if not file_path.exists():
            print(f"Warning: {file_path} not found; skipping {model_name}")
            continue
        with open(file_path, 'r') as f:
            entries = json.load(f)
        if not entries:
            print(f"Warning: {model_name} has no entries; skipping")
            continue
        matrices = [np.array(sample['results']) for sample in entries]
        avg_matrix = np.mean(matrices, axis=0)
        data[model_name] = {
            'avg_matrix': avg_matrix,
            'layers': info['layers'],
            'samples': len(matrices),
        }
        print(f"Loaded {model_name}: {len(matrices)} samples, matrix shape {avg_matrix.shape}")
    return data


def plot_target_layer_average(data, save_dir: Path):
    """Create a single line plot: average accuracy per target layer for each model."""
    if not data:
        print("No data available to plot.")
        return

    save_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Plot average (across source layers) per target layer; add half-std band
    for model_name, model_data in data.items():
        layers = model_data['layers']
        avg_matrix = model_data['avg_matrix']
        target_avg = np.mean(avg_matrix, axis=0)
        target_std = np.std(avg_matrix, axis=0)

        ax.plot(layers, target_avg, marker='o', label=model_name, linewidth=2.5, markersize=8)
        ax.fill_between(layers, target_avg - target_std / 2.0, target_avg + target_std / 2.0, alpha=0.2)

    ax.set_xlabel('Target Layer')
    ax.set_ylabel('Average Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    out_png = save_dir / 'target_layer_average_only.png'
    out_pdf = save_dir / 'target_layer_average_only.pdf'
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")
    plt.show()


def main():
    results_dir = Path('./lg_results/')
    save_dir = Path('./plots/')

    print("Generating single target-layer average line plot...")
    data = load_experimental_avg_matrices(results_dir)
    plot_target_layer_average(data, save_dir)


if __name__ == '__main__':
    main()
