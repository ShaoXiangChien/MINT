#!/usr/bin/env python3
"""Generate paper-ready figures from bootstrap analysis results.

Produces the main fusion-band heatmaps, target-layer comparison plots,
and appendix multi-panel figures referenced in the paper.

Script-to-figure mapping (update with final paper numbering):

- Figure 3:  Fusion band heatmaps        -> ``heatmap_*.pdf``
- Figure 4:  Target-layer comparison      -> ``target_layer_comparison.pdf``
- Table 5:   Bootstrap override accuracy  -> ``overall_metrics.csv``
- Appendix:  Full grid plots              -> ``appendix_*.pdf``

Usage::

    python -m evaluation.generate_paper_figures \
        --experiment_dirs experiments/02_multimodal_fusion/analysis \
                          experiments/04_global_image_fusion/analysis \
                          experiments/05_spatial_reasoning/analysis \
        --out_dir figures/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def load_all_summaries(dirs: List[str]) -> Dict[str, dict]:
    """Load bootstrap_summary.json from each experiment directory."""
    summaries = {}
    for d in dirs:
        p = Path(d) / "bootstrap_summary.json"
        if p.exists():
            with open(p) as f:
                summaries[Path(d).parent.name] = json.load(f)
    return summaries


def generate_combined_heatmap(summaries, out_dir):
    """Multi-panel heatmap: one panel per experiment type, each showing all models."""
    n_exps = len(summaries)
    if n_exps == 0:
        return
    fig, axes = plt.subplots(1, n_exps, figsize=(6 * n_exps, 5))
    if n_exps == 1:
        axes = [axes]

    for ax, (exp_name, summary) in zip(axes, summaries.items()):
        # Use first available model's matrix
        for model, data in summary.items():
            mat = data["experimental"].get("matrix", {}).get("mean")
            if mat is not None:
                sns.heatmap(np.array(mat), annot=False, cmap="YlOrRd",
                            ax=ax, vmin=0, vmax=1)
                ax.set_title(f"{exp_name}\n({model})")
                ax.set_xlabel("Target Layer")
                ax.set_ylabel("Source Layer")
                break

    fig.tight_layout()
    fig.savefig(out_dir / "combined_heatmaps.pdf", dpi=300)
    plt.close(fig)
    print(f"Combined heatmap -> {out_dir / 'combined_heatmaps.pdf'}")


def generate_target_layer_comparison(summaries, out_dir):
    """Cross-experiment target-layer comparison plot."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for exp_name, summary in summaries.items():
        for model, data in summary.items():
            tgt = data["experimental"].get("target_mean")
            if not tgt:
                continue
            means = np.array(tgt["mean"])
            x = np.arange(len(means))
            ax.plot(x, means, marker="o", markersize=3, label=f"{exp_name}/{model}")

    ax.set_xlabel("Target Layer Index")
    ax.set_ylabel("Override Accuracy")
    ax.set_title("Cross-Experiment Target-Layer Comparison")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "target_layer_comparison.pdf", dpi=300)
    plt.close(fig)
    print(f"Target-layer comparison -> {out_dir / 'target_layer_comparison.pdf'}")


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--experiment_dirs", nargs="+", required=True)
    parser.add_argument("--out_dir", type=str, default="figures/")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = load_all_summaries(args.experiment_dirs)
    print(f"Loaded {len(summaries)} experiment summaries")

    generate_combined_heatmap(summaries, out_dir)
    generate_target_layer_comparison(summaries, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
