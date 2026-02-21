#!/usr/bin/env python3
"""
Bootstrap Reporting Utilities (surf-exp negation, flip-rate)

Reads analysis_results/bootstrap_ci/bootstrap_summary.json and produces:
- Overall flip-rate table (CSV + Markdown)
- Overall flip-rate bar chart with 95% CIs
- Per-model target-layer flip-rate line plot with CI bands
- Per-model flip-rate heatmap with significance overlay and binary significance heatmap

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
import seaborn as sns  # noqa: F401


def load_summary(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def get_layers_for_model(model_name: str, length: int) -> List[int]:
    # Provide pleasant default labeling if we know typical sampling
    if model_name == "DeepSeek-VL2":
        return list(range(0, length * 2, 2))[:length]
    if model_name.upper().startswith("QWEN"):
        return list(range(0, length * 3, 3))[:length]
    if model_name.startswith("LLaVA"):
        return list(range(length))
    return list(range(length))


def make_overall_table(summary: Dict[str, Any], out_dir: Path) -> pd.DataFrame:
    rows = []
    for model, data in summary.items():
        flip_overall = data["flip"]["overall"]
        rows.append({
            "Model": model,
            "Flip Mean": flip_overall.get("mean", np.nan),
            "Flip 95% CI Lower": (flip_overall.get("ci") or [np.nan, np.nan])[0],
            "Flip 95% CI Upper": (flip_overall.get("ci") or [np.nan, np.nan])[1],
            "Flip Significant": flip_overall.get("significant", False),
            "Samples": data.get("samples", np.nan),
        })
    df = pd.DataFrame(rows)

    csv_path = out_dir / "overall_flip_metrics.csv"
    df.to_csv(csv_path, index=False)

    md_path = out_dir / "overall_flip_metrics.md"
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


def plot_overall_bars(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(df))
    means = df["Flip Mean"].values
    lower = df["Flip 95% CI Lower"].values
    upper = df["Flip 95% CI Upper"].values
    err_lower = means - lower
    err_upper = upper - means
    ax.bar(x, means, yerr=[err_lower, err_upper], capsize=6, color="steelblue", alpha=0.85)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"].values, rotation=0)
    ax.set_ylabel("Flip Rate (baseline wrong → patched correct)")
    ax.set_title("Overall Flip Rate with 95% CIs")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "overall_flip_rate_bars.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "overall_flip_rate_bars.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_target_flip_line(model: str, data: Dict[str, Any], out_dir: Path) -> None:
    flip = data["flip"]["target_mean"]
    mean = np.array(flip["mean"], dtype=float)
    lo = np.array(flip["ci_lower"], dtype=float)
    hi = np.array(flip["ci_upper"], dtype=float)
    layers = get_layers_for_model(model, len(mean))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(layers, mean, marker="o", linewidth=2)
    ax.fill_between(layers, lo, hi, alpha=0.2)
    ax.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Target Layer")
    ax.set_ylabel("Flip Rate")
    ax.set_title(f"{model}: Target-layer Flip Rate with 95% CIs")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{model}_target_flip_line.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_flip_heatmap_with_significance(model: str, data: Dict[str, Any], out_dir: Path) -> None:
    eff = data["flip"]["matrix"]
    mean = np.array(eff["mean"], dtype=float)
    sig = np.array(eff["significant"], dtype=int)

    fig, ax = plt.subplots(figsize=(6, 5))
    vmax = np.nanmax(np.abs(mean)) if mean.size > 0 else 1.0
    if np.isnan(vmax) or vmax == 0:
        vmax = 1.0
    im = ax.imshow(mean, vmin=0, vmax=vmax, cmap="viridis")
    ax.set_title(f"{model}: Flip Rate")
    ax.set_xlabel("Target Layer")
    ax.set_ylabel("Source Layer")
    ax.set_xticks(range(mean.shape[1]))
    ax.set_yticks(range(mean.shape[0]))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Flip")
    for i in range(mean.shape[0]):
        for j in range(mean.shape[1]):
            if sig[i, j] == 1:
                ax.text(j, i, "•", ha="center", va="center", color="black", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / f"{model}_flip_heatmap_with_sig.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    im2 = ax.imshow(sig, vmin=0, vmax=1, cmap="Greens")
    ax.set_title(f"{model}: Significance (CI lower > 0)")
    ax.set_xlabel("Target Layer")
    ax.set_ylabel("Source Layer")
    ax.set_xticks(range(sig.shape[1]))
    ax.set_yticks(range(sig.shape[0]))
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / f"{model}_significance_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_markdown_blurb(df: pd.DataFrame, out_dir: Path) -> None:
    md = out_dir / "paper_blurb.md"
    lines = [
        "# Bootstrap CI Reporting Snippets (surf-exp negation)\n",
        "\n",
        "Use these ready-to-paste sentences (edit numbers as needed):\n\n",
    ]
    for _, row in df.iterrows():
        lines.append(
            (
                f"- {row['Model']}: Overall flip-rate was "
                f"{float(row['Flip Mean']):.3f} (95% CI: {float(row['Flip 95% CI Lower']):.3f} to "
                f"{float(row['Flip 95% CI Upper']):.3f}) based on {int(row['Samples'])} samples.\n"
            )
        )
    with open(md, "w") as f:
        f.writelines(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report figures and tables for bootstrap flip-rate CIs (surf-exp negation)")
    parser.add_argument("--summary", type=str, default="./analysis_results/bootstrap_ci/bootstrap_summary.json")
    parser.add_argument("--out_dir", type=str, default="./analysis_results/bootstrap_ci/report/")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(summary_path)

    df = make_overall_table(summary, out_dir)
    plot_overall_bars(df, out_dir)
    for model, data in summary.items():
        plot_target_flip_line(model, data, out_dir)
        plot_flip_heatmap_with_significance(model, data, out_dir)
    write_markdown_blurb(df, out_dir)

    print(f"Saved report to: {out_dir}")


if __name__ == "__main__":
    main()
