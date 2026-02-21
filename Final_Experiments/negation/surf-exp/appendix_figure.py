#!/usr/bin/env python3
"""
Appendix Combined Figure (surf-exp negation)

Builds a multi-panel figure per model: left = target-layer flip-rate line with CI,
right = flip-rate heatmap with significance overlay.

Usage:
  python appendix_figure.py \
    --summary ./analysis_results/bootstrap_ci/bootstrap_summary.json \
    --out_dir ./analysis_results/bootstrap_ci/report/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import matplotlib.pyplot as plt


def load_summary(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def get_layers_for_model(model_name: str, length: int) -> List[int]:
    if model_name == "DeepSeek-VL2":
        return list(range(0, length * 2, 2))[:length]
    if model_name.upper().startswith("QWEN"):
        return list(range(0, length * 3, 3))[:length]
    return list(range(length))


def main() -> None:
    ap = argparse.ArgumentParser(description="Appendix combined figure (surf-exp negation)")
    ap.add_argument("--summary", type=str, default="./analysis_results/bootstrap_ci/bootstrap_summary.json")
    ap.add_argument("--out_dir", type=str, default="./analysis_results/bootstrap_ci/report/")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = load_summary(Path(args.summary))

    models = list(summary.keys())
    if not models:
        print("No models in summary.")
        return

    fig, axes = plt.subplots(nrows=len(models), ncols=2, figsize=(12, 4 * len(models)))
    if len(models) == 1:
        axes = np.array([axes])

    for i, model in enumerate(models):
        data = summary[model]
        # Left: target-layer flip-rate line
        flip_t = data["flip"]["target_mean"]
        mean = np.array(flip_t["mean"], dtype=float)
        lo = np.array(flip_t["ci_lower"], dtype=float)
        hi = np.array(flip_t["ci_upper"], dtype=float)
        layers = get_layers_for_model(model, len(mean))
        ax = axes[i, 0]
        ax.plot(layers, mean, marker="o", linewidth=2)
        ax.fill_between(layers, lo, hi, alpha=0.2)
        ax.axhline(0, color="black", linestyle="--", alpha=0.5)
        ax.set_title(f"{model}: Target-layer flip-rate")
        ax.set_xlabel("Target Layer")
        ax.set_ylabel("Flip Rate")
        ax.grid(True, alpha=0.3)

        # Right: flip-rate heatmap with significance overlay
        flip_m = data["flip"]["matrix"]
        m = np.array(flip_m["mean"], dtype=float)
        sig = np.array(flip_m["significant"], dtype=int)
        ax = axes[i, 1]
        vmax = np.nanmax(np.abs(m)) if m.size > 0 else 1.0
        if np.isnan(vmax) or vmax == 0:
            vmax = 1.0
        im = ax.imshow(m, vmin=0, vmax=vmax, cmap="viridis")
        ax.set_title(f"{model}: Flip Rate")
        ax.set_xlabel("Target Layer")
        ax.set_ylabel("Source Layer")
        ax.set_xticks(range(m.shape[1]))
        ax.set_yticks(range(m.shape[0]))
        for r in range(m.shape[0]):
            for c in range(m.shape[1]):
                if sig[r, c] == 1:
                    ax.text(c, r, "•", ha="center", va="center", color="black", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    for ext in ["png", "pdf"]:
        path = out_dir / f"appendix_combined.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()


