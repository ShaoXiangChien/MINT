#!/usr/bin/env python3
"""
Appendix Combined Figure (Text-Patching)

Combines the key per-model plots into a single multi-panel figure for the paper
appendix. Expects the same summary JSON as the other reporting scripts.

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
    if model_name.upper().startswith("QWEN"):
        return list(range(0, 28, 3))
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Appendix combined figure (text-patching)")
    parser.add_argument("--summary", type=str, default="./analysis_results/bootstrap_ci/bootstrap_summary.json")
    parser.add_argument("--out_dir", type=str, default="./analysis_results/bootstrap_ci/report/")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = load_summary(Path(args.summary))

    # Panels: for each model, left = target effect line with CI; right = effect heatmap
    models = list(summary.keys())
    n = len(models)
    if n == 0:
        print("No models found in summary.")
        return

    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(12, 4 * n))
    if n == 1:
        axes = np.array([axes])

    for i, model in enumerate(models):
        data = summary[model]
        layers = get_layers_for_model(model)

        # Left: target-layer effect line with CI if effect exists
        ax = axes[i, 0]
        eff_t = data.get("effect", {}).get("target_mean")
        if eff_t is not None:
            mean = np.array(eff_t["mean"], dtype=float)
            lo = np.array(eff_t["ci_lower"], dtype=float)
            hi = np.array(eff_t["ci_upper"], dtype=float)
            if not layers:
                layers = list(range(len(mean)))
            ax.plot(layers, mean, marker="o", linewidth=2)
            ax.fill_between(layers, lo, hi, alpha=0.2)
            ax.axhline(0, color="black", linestyle="--", alpha=0.5)
        else:
            # Fallback: experimental target means (no zero line)
            exp_t = data["experimental"]["target_mean"]
            mean = np.array(exp_t["mean"], dtype=float)
            lo = np.array(exp_t["ci_lower"], dtype=float)
            hi = np.array(exp_t["ci_upper"], dtype=float)
            if not layers:
                layers = list(range(len(mean)))
            ax.plot(layers, mean, marker="o", linewidth=2)
            ax.fill_between(layers, lo, hi, alpha=0.2)
        ax.set_title(f"{model}: Target-layer")
        ax.set_xlabel("Target Layer")
        ax.set_ylabel("Effect" if eff_t is not None else "Accuracy")
        ax.grid(True, alpha=0.3)

        # Right: effect heatmap with significance if available, else experimental mean
        ax = axes[i, 1]
        eff_m = data.get("effect", {}).get("matrix")
        if eff_m is not None:
            mean = np.array(eff_m["mean"], dtype=float)
            sig = np.array(eff_m["significant"], dtype=int)
            vmax = np.max(np.abs(mean)) if mean.size > 0 else 1.0
            im = ax.imshow(mean, vmin=-vmax, vmax=vmax, cmap="RdBu_r")
            for r in range(mean.shape[0]):
                for c in range(mean.shape[1]):
                    if sig[r, c] == 1:
                        ax.text(c, r, "•", ha="center", va="center", color="black", fontsize=9)
        else:
            exp_m = data["experimental"]["matrix"]
            mean = np.array(exp_m["mean"], dtype=float)
            im = ax.imshow(mean, vmin=np.min(mean), vmax=np.max(mean), cmap="viridis")
        ax.set_title(f"{model}: {'Effect' if eff_m is not None else 'Experimental'}")
        ax.set_xlabel("Target Layer")
        ax.set_ylabel("Source Layer")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    for ext in ["png", "pdf"]:
        path = out_dir / f"appendix_combined.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()


