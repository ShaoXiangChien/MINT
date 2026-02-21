#!/usr/bin/env python3
"""
Bootstrap Reporting Utilities

Reads analysis_results/bootstrap_ci/bootstrap_summary.json and produces:
- Overall metrics table (CSV + Markdown)
- Overall intervention effect bar chart with 95% CIs
- Per-model target-layer intervention effect line plot with CI bands
- Per-model significance heatmap (binary CI-excludes-0) and effect heatmap with significance overlay

Usage:
    python report_bootstrap.py \
      --summary ./analysis_results/bootstrap_ci/bootstrap_summary.json \
      --out_dir ./analysis_results/bootstrap_ci/report/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
    # Fallback: infer length from data caller if needed
    return []


def make_overall_table(summary: Dict[str, Any], out_dir: Path) -> pd.DataFrame:
    rows = []
    for model, data in summary.items():
        exp_overall = data["experimental"]["overall"]
        base_overall = data["baseline"]["overall"]
        eff_overall = data["effect"]["overall"]
        rows.append({
            "Model": model,
            "Experimental Mean": exp_overall["mean"],
            "Experimental 95% CI Lower": exp_overall["ci"][0],
            "Experimental 95% CI Upper": exp_overall["ci"][1],
            "Baseline Mean": base_overall["mean"],
            "Baseline 95% CI Lower": base_overall["ci"][0],
            "Baseline 95% CI Upper": base_overall["ci"][1],
            "Effect Mean": eff_overall["mean"],
            "Effect 95% CI Lower": eff_overall["ci"][0],
            "Effect 95% CI Upper": eff_overall["ci"][1],
            "Effect Significant": eff_overall.get("significant", False),
        })
    df = pd.DataFrame(rows)
    # Save CSV
    csv_path = out_dir / "overall_metrics.csv"
    df.to_csv(csv_path, index=False)
    # Save Markdown (no external tabulate dependency)
    md_path = out_dir / "overall_metrics.md"
    headers = list(df.columns)
    align = ["---" for _ in headers]
    def fmt(v):
        if isinstance(v, (int, bool)):
            return str(v)
        try:
            return f"{float(v):.3f}"
        except Exception:
            return str(v)
    with open(md_path, "w") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(align) + " |\n")
        for _, row in df.iterrows():
            f.write("| " + " | ".join(fmt(row[h]) for h in headers) + " |\n")
    return df


def plot_overall_effect_bars(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(df))
    means = df["Effect Mean"].values
    lower = df["Effect 95% CI Lower"].values
    upper = df["Effect 95% CI Upper"].values
    err_lower = means - lower
    err_upper = upper - means
    ax.bar(x, means, yerr=[err_lower, err_upper], capsize=6,
           color=["green" if m > 0 else "red" for m in means], alpha=0.8)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"].values, rotation=0)
    ax.set_ylabel("Intervention Effect (Exp − Base)")
    ax.set_title("Overall Intervention Effect with 95% CIs")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "overall_effect_bars.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "overall_effect_bars.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_target_effect_line(model: str, data: Dict[str, Any], out_dir: Path) -> None:
    eff = data["effect"]["target_mean"]
    mean = np.array(eff["mean"], dtype=float)
    lo = np.array(eff["ci_lower"], dtype=float)
    hi = np.array(eff["ci_upper"], dtype=float)
    layers = get_layers_for_model(model)
    if not layers:
        layers = list(range(len(mean)))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(layers, mean, marker="o", linewidth=2)
    ax.fill_between(layers, lo, hi, alpha=0.2)
    ax.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Target Layer")
    ax.set_ylabel("Effect (Exp − Base)")
    ax.set_title(f"{model}: Target-layer Intervention Effect with 95% CIs")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{model}_target_effect_line.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_effect_heatmap_with_significance(model: str, data: Dict[str, Any], out_dir: Path) -> None:
    eff = data["effect"]["matrix"]
    mean = np.array(eff["mean"], dtype=float)
    sig = np.array(eff["significant"], dtype=int)
    layers = get_layers_for_model(model)
    if not layers:
        layers = list(range(mean.shape[0]))

    # Heatmap of mean effect
    fig, ax = plt.subplots(figsize=(6, 5))
    vmax = np.max(np.abs(mean))
    im = ax.imshow(mean, vmin=-vmax, vmax=vmax, cmap="RdBu_r")
    ax.set_title(f"{model}: Effect (Exp − Base)")
    ax.set_xlabel("Target Layer")
    ax.set_ylabel("Source Layer")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=45)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers[::-1])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Effect")
    # Overlay significance markers
    for i in range(mean.shape[0]):
        for j in range(mean.shape[1]):
            if sig[i, j] == 1:
                ax.text(j, i, "•", ha="center", va="center", color="black", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / f"{model}_effect_heatmap_with_sig.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Binary significance heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    im2 = ax.imshow(sig, vmin=0, vmax=1, cmap="Greens")
    ax.set_title(f"{model}: Significance (CI excludes 0)")
    ax.set_xlabel("Target Layer")
    ax.set_ylabel("Source Layer")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=45)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers[::-1])
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / f"{model}_significance_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_markdown_blurb(df: pd.DataFrame, out_dir: Path) -> None:
    md = out_dir / "paper_blurb.md"
    lines = [
        "# Bootstrap CI Reporting Snippets\n",
        "\n",
        "Use these ready-to-paste sentences (edit numbers as needed):\n\n",
    ]
    for _, row in df.iterrows():
        lines.append(
            (
                f"- {row['Model']}: Overall intervention effect was "
                f"{row['Effect Mean']:.3f} (95% CI: {row['Effect 95% CI Lower']:.3f} to "
                f"{row['Effect 95% CI Upper']:.3f}); "
                f"baseline accuracy {row['Baseline Mean']:.3f} (95% CI: {row['Baseline 95% CI Lower']:.3f} to "
                f"{row['Baseline 95% CI Upper']:.3f}).\n"
            )
        )
    with open(md, "w") as f:
        f.writelines(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report figures and tables for bootstrap CIs")
    parser.add_argument("--summary", type=str, default="./analysis_results/bootstrap_ci/bootstrap_summary.json")
    parser.add_argument("--out_dir", type=str, default="./analysis_results/bootstrap_ci/report/")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(summary_path)

    # 1) Overall metrics table
    df = make_overall_table(summary, out_dir)

    # 2) Overall effect bar chart with CIs
    plot_overall_effect_bars(df, out_dir)

    # 3) Per-model target-layer effect lines and heatmaps
    for model, data in summary.items():
        plot_target_effect_line(model, data, out_dir)
        plot_effect_heatmap_with_significance(model, data, out_dir)

    # 4) Helper blurb for paper
    write_markdown_blurb(df, out_dir)

    print(f"Saved report to: {out_dir}")


if __name__ == "__main__":
    main()


