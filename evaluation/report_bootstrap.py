#!/usr/bin/env python3
"""Bootstrap Reporting -- tables, charts, and heatmaps.

Reads a ``bootstrap_summary.json`` produced by :mod:`evaluation.bootstrap_ci`
and generates paper-ready artefacts:

- Overall metrics table (CSV + Markdown)
- Intervention-effect bar chart with 95 % CIs
- Per-model target-layer line plots with CI bands
- Per-model (source x target) significance heatmaps

Usage::

    python -m evaluation.report_bootstrap \
        --summary experiments/04_global_image_fusion/analysis/bootstrap_summary.json \
        --out_dir experiments/04_global_image_fusion/analysis/report/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def load_summary(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def make_overall_table(summary, out_dir):
    rows = []
    for model, data in summary.items():
        exp = data["experimental"]["overall"]
        eff = (data.get("effect") or {}).get("overall", {})
        rows.append({
            "Model": model,
            "Override Accuracy": exp.get("mean", float("nan")),
            "CI Lower": (exp.get("ci") or [float("nan")])[0],
            "CI Upper": (exp.get("ci") or [0, float("nan")])[1],
            "Effect": eff.get("mean", ""),
            "Significant": eff.get("significant", ""),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "overall_metrics.csv", index=False)
    df.to_markdown(out_dir / "overall_metrics.md", index=False)
    print(f"  Table -> {out_dir / 'overall_metrics.csv'}")
    return df


def plot_effect_bar(summary, out_dir):
    models, means, lowers, uppers = [], [], [], []
    for model, data in summary.items():
        eff = (data.get("effect") or {}).get("overall")
        if not eff:
            continue
        models.append(model)
        means.append(eff["mean"])
        ci = eff["ci"]
        lowers.append(eff["mean"] - ci[0])
        uppers.append(ci[1] - eff["mean"])
    if not models:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(models))
    ax.bar(x, means, yerr=[lowers, uppers], capsize=5, color="steelblue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Effect (Override - Baseline)")
    ax.set_title("Intervention Effect with 95% CI")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_dir / "effect_bar.pdf", dpi=150)
    plt.close(fig)
    print(f"  Bar chart -> {out_dir / 'effect_bar.pdf'}")


def plot_target_layer_lines(summary, out_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    for model, data in summary.items():
        tgt = data["experimental"].get("target_mean")
        if not tgt:
            continue
        means = np.array(tgt["mean"])
        ci_lo = np.array(tgt["ci_lower"])
        ci_hi = np.array(tgt["ci_upper"])
        x = np.arange(len(means))
        ax.plot(x, means, marker="o", label=model, markersize=4)
        ax.fill_between(x, ci_lo, ci_hi, alpha=0.15)
    ax.set_xlabel("Target Layer Index")
    ax.set_ylabel("Override Accuracy")
    ax.set_title("Target-Layer Override Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "target_layer_lines.pdf", dpi=150)
    plt.close(fig)
    print(f"  Line plot -> {out_dir / 'target_layer_lines.pdf'}")


def plot_heatmaps(summary, out_dir):
    for model, data in summary.items():
        mat = data["experimental"].get("matrix")
        if not mat or not mat.get("mean"):
            continue
        arr = np.array(mat["mean"])
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(arr, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax,
                    vmin=0, vmax=1)
        ax.set_xlabel("Target Layer")
        ax.set_ylabel("Source Layer")
        ax.set_title(f"{model} -- Override Accuracy Heatmap")
        fig.tight_layout()
        safe_name = model.replace(" ", "_").replace("/", "_")
        fig.savefig(out_dir / f"heatmap_{safe_name}.pdf", dpi=150)
        plt.close(fig)
        print(f"  Heatmap -> {out_dir / f'heatmap_{safe_name}.pdf'}")


def main():
    parser = argparse.ArgumentParser(description="Bootstrap reporting")
    parser.add_argument("--summary", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    summary = load_summary(Path(args.summary))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating reports...")
    make_overall_table(summary, out_dir)
    plot_effect_bar(summary, out_dir)
    plot_target_layer_lines(summary, out_dir)
    plot_heatmaps(summary, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
