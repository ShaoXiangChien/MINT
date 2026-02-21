#!/usr/bin/env python3
"""
Appendix Combined Figure Generator

Creates a multi-panel figure for the appendix that summarizes bootstrap CI results:
- Left column (per model): Target-layer intervention effect lines with 95% CI bands
- Right column (per model): Effect heatmap (Exp − Base) with significance overlay (CI excludes 0)

Usage:
    python appendix_figure.py \
      --summary ./analysis_results/bootstrap_ci/bootstrap_summary.json \
      --out_dir ./analysis_results/bootstrap_ci/report/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt


def load_summary(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def get_layers_for_model(model_name: str) -> List[int]:
    if model_name == "DeepSeek-VL2":
        return list(range(0, 12, 2))
    if model_name == "LLaVA-1.5":
        return list(range(0, 32, 3))
    if model_name == "QWEN2-VL":
        return list(range(0, 28, 3))
    if model_name == "Qwen2.5-VL":
        return list(range(0, 28, 3))
    return []


def plot_row(ax_left, ax_right, model: str, data: Dict[str, Any], unified_vmax: float) -> None:
    tgt = data["effect"]["target_mean"]
    mean = np.array(tgt["mean"], dtype=float)
    lo = np.array(tgt["ci_lower"], dtype=float)
    hi = np.array(tgt["ci_upper"], dtype=float)
    layers = get_layers_for_model(model)
    if not layers:
        layers = list(range(len(mean)))

    ax_left.plot(layers, mean, marker="o", linewidth=2)
    ax_left.fill_between(layers, lo, hi, alpha=0.2)
    ax_left.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax_left.set_xlabel("Target Layer")
    ax_left.set_ylabel("Effect (Exp − Base)")
    ax_left.grid(True, alpha=0.3)

    eff = data["effect"]["matrix"]
    eff_mean = np.array(eff["mean"], dtype=float)
    eff_sig = np.array(eff["significant"], dtype=int)
    vmax = unified_vmax if unified_vmax is not None else (np.max(np.abs(eff_mean)) if eff_mean.size else 1.0)
    im = ax_right.imshow(eff_mean, vmin=-vmax, vmax=vmax, cmap="RdBu_r")
    ax_right.set_xlabel("Target Layer")
    ax_right.set_ylabel("Source Layer")
    ax_right.set_xticks(range(len(layers)))
    ax_right.set_xticklabels(layers, rotation=45)
    ax_right.set_yticks(range(len(layers)))
    ax_right.set_yticklabels(layers[::-1])
    for i in range(eff_mean.shape[0]):
        for j in range(eff_mean.shape[1]):
            if eff_sig[i, j] == 1:
                ax_right.text(j, i, "•", ha="center", va="center", color="black", fontsize=8)
    # Add a per-axis colorbar so each heatmap has its own legend
    plt.colorbar(im, ax=ax_right, fraction=0.046, pad=0.04)
    return im


def main() -> None:
    parser = argparse.ArgumentParser(description="Appendix combined figure for bootstrap results")
    parser.add_argument("--summary", type=str, default="./analysis_results/bootstrap_ci/bootstrap_summary.json")
    parser.add_argument("--out_dir", type=str, default="./analysis_results/bootstrap_ci/report/")
    args = parser.parse_args()

    summary = load_summary(Path(args.summary))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = [
        m for m in ["DeepSeek-VL2", "LLaVA-1.5", "QWEN2-VL", "Qwen2.5-VL"] if m in summary
    ]
    if not models:
        models = list(summary.keys())

    nrows = len(models)
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 4 * nrows))
    if nrows == 1:
        axes = np.array([axes])

    # Compute a unified color scale across models for consistent comparison
    global_vmax = 0.0
    for m in models:
        eff_mean = np.array(summary[m]["effect"]["matrix"]["mean"], dtype=float)
        if eff_mean.size:
            global_vmax = max(global_vmax, float(np.max(np.abs(eff_mean))))
    if global_vmax == 0.0:
        global_vmax = 1.0

    cbar_im = None
    for r, model in enumerate(models):
        ax_left = axes[r, 0]
        ax_right = axes[r, 1]
        im = plot_row(ax_left, ax_right, model, summary[model], unified_vmax=global_vmax)
        cbar_im = im

    # Per-axis colorbars are already attached to each heatmap; no shared colorbar

    # Use constrained layout and remove figure/title-level headings for a clean appendix figure
    # (titles removed per request)

    out_png = out_dir / "appendix_combined.png"
    out_pdf = out_dir / "appendix_combined.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()


