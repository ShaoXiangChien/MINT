#!/usr/bin/env python3
"""
Plot Average Accuracy per Target Layer (Text Patching, Single Figure)

Loads experimental results from `results/` and generates a single
line plot showing the average accuracy per target layer (averaged
across source layers) for each available model.

Outputs:
 - plots/target_layer_average_only.png
 - plots/target_layer_average_only.pdf

Usage: python plot_target_layer_average.py
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Styling similar to quick_analysis
plt.rcParams['font.size'] = 18
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['figure.dpi'] = 100

MODELS = {
    'DeepSeek-VL2': {'file': 'ds-results.json'},
    'Qwen2-VL': {'file': 'qwen_results.json'},
}

# Explicit target-layer indices per model (x-axis positions)
# DeepSeek-VL2: tested target layers 0,2,4,...,10
# Qwen2-VL: tested target layers 0,3,6,...,27
TARGET_LAYER_INDICES = {
    'DeepSeek-VL2': list(range(0, 12, 2)),
    'Qwen2-VL': list(range(0, 28, 3)),
}


def load_avg_matrices(results_dir: Path):
    """Load results and compute average matrix per model.

    Returns: { model_name: { 'avg_matrix': np.ndarray, 'layers_target': list[int], 'samples': int } }
    Handles possible non-square matrices by deriving target length from columns.
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
        matrices = [np.array(sample['results'], dtype=float) for sample in entries]
        # Defensive shape alignment if mismatched shapes appear
        shapes = [m.shape for m in matrices]
        if len(set(shapes)) != 1:
            target_shape = max(shapes, key=lambda s: s[0] * s[1])
            fixed = []
            for m in matrices:
                if m.shape == target_shape:
                    fixed.append(m)
                else:
                    r, c = m.shape
                    tr, tc = target_shape
                    out = np.zeros(target_shape, dtype=float)
                    rr = min(r, tr)
                    cc = min(c, tc)
                    out[:rr, :cc] = m[:rr, :cc]
                    fixed.append(out)
            matrices = fixed
        avg_matrix = np.mean(matrices, axis=0)
        planned_indices = TARGET_LAYER_INDICES.get(model_name)
        # Fallback if not specified
        if planned_indices is None:
            planned_indices = list(range(avg_matrix.shape[1]))

        # Align matrix columns with planned indices length if they differ
        cols = avg_matrix.shape[1]
        if cols != len(planned_indices):
            print(
                f"Warning: column count ({cols}) != planned index length ({len(planned_indices)}) for {model_name}. Aligning..."
            )
            if cols > len(planned_indices):
                avg_matrix = avg_matrix[:, : len(planned_indices)]
            else:
                planned_indices = planned_indices[:cols]

        layers_target = planned_indices
        data[model_name] = {
            'avg_matrix': avg_matrix,
            'layers_target': layers_target,
            'samples': len(matrices),
        }
        print(f"Loaded {model_name}: {len(matrices)} samples, matrix shape {avg_matrix.shape}")
    return data


def plot_target_layer_average(data, save_dir: Path):
    if not data:
        print("No data to plot.")
        return
    save_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for model_name, model_data in data.items():
        layers = model_data['layers_target']
        avg_matrix = model_data['avg_matrix']
        target_avg = np.mean(avg_matrix, axis=0)
        target_std = np.std(avg_matrix, axis=0)

        ax.plot(layers, target_avg, marker='o', label=model_name, linewidth=2.5, markersize=8)
        ax.fill_between(layers, target_avg - target_std / 2.0, target_avg + target_std / 2.0, alpha=0.2)

    ax.set_xlabel('Target Layer Index')
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
    results_dir = Path('./results/')
    save_dir = Path('./plots/')

    print("Generating single target-layer average line plot (text-patching)...")
    data = load_avg_matrices(results_dir)
    plot_target_layer_average(data, save_dir)


if __name__ == '__main__':
    main()
