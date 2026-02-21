#!/usr/bin/env python3
"""
Plot Average Effectiveness per Source Layer (Text Patching, Single Figure)

Loads experimental results from `results/` and generates a single
line plot showing the average effectiveness per source layer (averaged
across target layers) for each available model.

Outputs:
 - plots/source_layer_average_only.png
 - plots/source_layer_average_only.pdf

Usage: python plot_source_layer_average.py
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
    'LLaVA': {'file': 'llava_results.json'},
}

# Explicit source-layer indices per model (x-axis positions)
# These are inferred from the matrix dimensions (rows)
# If matrices are square, source layers typically match target layers
# DeepSeek-VL2: typically tested source layers 0,2,4,...,10
# Qwen2-VL: typically tested source layers 0,3,6,...,27
# LLaVA: will be inferred from matrix dimensions
SOURCE_LAYER_INDICES = {
    'DeepSeek-VL2': list(range(0, 12, 2)),
    'Qwen2-VL': list(range(0, 28, 3)),
    # LLaVA will be inferred from matrix dimensions
}


def load_avg_matrices(results_dir: Path):
    """Load results and compute average matrix per model.

    Returns: { model_name: { 'avg_matrix': np.ndarray, 'layers_source': list[int], 'samples': int } }
    Handles possible non-square matrices by deriving source length from rows.
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
        
        # Infer source layer indices from matrix rows
        num_source_layers = avg_matrix.shape[0]
        planned_indices = SOURCE_LAYER_INDICES.get(model_name)
        
        # Fallback: infer from matrix dimensions if not specified
        if planned_indices is None:
            planned_indices = list(range(num_source_layers))
        else:
            # Align planned indices with actual matrix rows if they differ
            if num_source_layers != len(planned_indices):
                print(
                    f"Warning: row count ({num_source_layers}) != planned index length ({len(planned_indices)}) for {model_name}. Aligning..."
                )
                if num_source_layers > len(planned_indices):
                    # If we have more rows than planned indices, infer from dimensions
                    planned_indices = list(range(num_source_layers))
                else:
                    planned_indices = planned_indices[:num_source_layers]

        layers_source = planned_indices
        data[model_name] = {
            'avg_matrix': avg_matrix,
            'layers_source': layers_source,
            'samples': len(matrices),
        }
        print(f"Loaded {model_name}: {len(matrices)} samples, matrix shape {avg_matrix.shape}")
    return data


def plot_source_layer_average(data, save_dir: Path):
    """Plot average effectiveness per source layer (averaged across target layers)."""
    if not data:
        print("No data to plot.")
        return
    save_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for model_name, model_data in data.items():
        layers = model_data['layers_source']
        avg_matrix = model_data['avg_matrix']
        # Average across target layers (axis=1) to get source layer effectiveness
        source_avg = np.mean(avg_matrix, axis=1)
        source_std = np.std(avg_matrix, axis=1)

        ax.plot(layers, source_avg, marker='o', label=model_name, linewidth=2.5, markersize=8)
        ax.fill_between(layers, source_avg - source_std / 2.0, source_avg + source_std / 2.0, alpha=0.2)

    ax.set_xlabel('Source Layer Index')
    ax.set_ylabel('Average Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    out_png = save_dir / 'source_layer_average_only.png'
    out_pdf = save_dir / 'source_layer_average_only.pdf'
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")
    plt.show()


def main():
    results_dir = Path('./results/')
    save_dir = Path('./plots/')

    print("Generating single source-layer average line plot (text-patching)...")
    data = load_avg_matrices(results_dir)
    plot_source_layer_average(data, save_dir)


if __name__ == '__main__':
    main()

