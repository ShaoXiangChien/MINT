#!/usr/bin/env python3
"""
Plot average value per target layer across models for spatial patching.

Reads per-model matrices (correct-rate or flip-rate) from
results/spatial-relationship/{ds,llava,qwen} and produces a single line plot
showing the mean across source layers for each target layer, one line per model.

Usage:
  python plot_target_layer_avg_spatial.py \
    --metric flip \
    --base_dir "/home/sc305/VLM/Final Experiments/spatial-relationship/results/spatial-relationship" \
    --output_dir "/home/sc305/VLM/Final Experiments/spatial-relationship/results/comparisons" \
    --save_pdf
"""

from pathlib import Path
from typing import Dict, List, Optional
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Matplotlib appearance
plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'figure.dpi': 100,
})


MODEL_PREFIX_TO_LABEL: Dict[str, str] = {
    'ds': 'DeepSeek-VL2',
    'llava': 'LLaVA-1.5',
    'qwen': 'QWEN2-VL',
}


def load_matrix(csv_path: Path) -> Optional[pd.DataFrame]:
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path, index_col=0)
    except Exception:
        return None
    return df


def compute_target_series(df: pd.DataFrame) -> Optional[Dict[str, np.ndarray]]:
    if df is None or df.empty:
        return None

    # Columns are target layers; rows are source layers.
    # Ensure numeric ordering of target layers for plotting consistency.
    try:
        target_layers_numeric = pd.to_numeric(df.columns, errors='coerce')
    except Exception:
        target_layers_numeric = df.columns

    order = np.argsort(target_layers_numeric)
    ordered_columns = [df.columns[i] for i in order]
    df_ordered = df[ordered_columns]

    values = df_ordered.values.astype(float)
    mean_per_target = np.nanmean(values, axis=0)
    std_per_target = np.nanstd(values, axis=0)

    # Convert ordered column labels back to ints when possible for x-axis
    try:
        layers_for_plot = [int(x) for x in ordered_columns]
    except Exception:
        layers_for_plot = list(range(len(ordered_columns)))

    return {
        'layers': np.array(layers_for_plot, dtype=int),
        'mean': mean_per_target,
        'std': std_per_target,
    }


def plot_target_layer_avg(
    base_dir: Path,
    output_dir: Path,
    metric: str = 'flip',
    models: Optional[List[str]] = None,
    save_pdf: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if models is None:
        models = ['ds', 'llava', 'qwen']

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    any_plotted = False

    for model in models:
        label = MODEL_PREFIX_TO_LABEL.get(model, model)
        subdir = base_dir / model
        if metric == 'correct':
            csv_name = f"{model}_correct_rate_matrix.csv"
            ylabel = "Average Correct Rate"
            out_suffix = "correct"
        else:
            csv_name = f"{model}_flip_rate_matrix.csv"
            ylabel = "Average Flip Rate"
            out_suffix = "flip"

        csv_path = subdir / csv_name
        df = load_matrix(csv_path)
        if df is None:
            print(f"[WARN] Missing or invalid CSV for {label}: {csv_path}")
            continue

        series = compute_target_series(df)
        if series is None:
            print(f"[WARN] Empty data for {label}: {csv_path}")
            continue

        layers = series['layers']
        mean_t = series['mean']
        std_t = series['std']

        ax.plot(layers, mean_t, marker='o', linewidth=2.5, markersize=7, label=label)
        ax.fill_between(layers, mean_t - std_t / 2.0, mean_t + std_t / 2.0, alpha=0.2)
        any_plotted = True

    ax.set_xlabel("Target Layer")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    if any_plotted:
        ax.legend()

    plt.tight_layout()

    png_path = output_dir / f"target_layer_avg_only_{out_suffix}.png"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    if save_pdf:
        pdf_path = output_dir / f"target_layer_avg_only_{out_suffix}.pdf"
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"[OK] Saved: {png_path} and {pdf_path}")
    else:
        print(f"[OK] Saved: {png_path}")

    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot average per target layer for spatial patching results")
    parser.add_argument(
        "--metric",
        choices=["flip", "correct"],
        default="flip",
        help="Which matrix to use: flip rate or correct rate",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/home/sc305/VLM/Final Experiments/spatial-relationship/results/spatial-relationship",
        help="Base directory containing per-model subfolders (ds/ llava/ qwen/)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/sc305/VLM/Final Experiments/spatial-relationship/results/comparisons",
        help="Where to save the plot(s)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ds", "llava", "qwen"],
        help="Model prefixes to include",
    )
    parser.add_argument(
        "--save_pdf",
        action="store_true",
        help="Also save a PDF alongside the PNG",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")
    plot_target_layer_avg(
        base_dir=base_dir,
        output_dir=output_dir,
        metric=args.metric,
        models=args.models,
        save_pdf=args.save_pdf,
    )


if __name__ == "__main__":
    main()


