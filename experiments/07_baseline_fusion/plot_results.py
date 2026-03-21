"""Visualization for Baseline Fusion Experiment (Section 5.1).

Generates two figures:
  1. Override Accuracy heatmaps for each dimension (Object, Attribute, Spatial)
     plotted side-by-side to show that the Fusion Band is consistent across tasks.
  2. Cross-metric corroboration: mean Override Accuracy (diagonal of the
     patching heatmap) vs. mean text-to-image attention weight per layer.

Usage::

    python -m experiments.07_baseline_fusion.plot_results \\
        --patching_results results/baseline_fusion_qwen.json \\
        --attention_results results/attention_corroboration_qwen.json \\
        --output_dir       figures/baseline_fusion \\
        --model_name       "Qwen2-VL-7B"
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


DIMENSION_LABELS = {
    "object":    "Object Existence\n(POPE)",
    "attribute": "Attribute\n(GQA)",
    "spatial":   "Spatial Relationship\n(What's Up)",
}
DIMENSION_ORDER = ["object", "attribute", "spatial"]
DIMENSION_COLORS = {
    "object":    "#2196F3",   # blue
    "attribute": "#FF9800",   # orange
    "spatial":   "#4CAF50",   # green
}


def load_patching(path: str):
    with open(path) as f:
        return json.load(f)


def load_attention(path: str):
    with open(path) as f:
        return json.load(f)


def compute_mean_heatmap(results, dimension: str):
    """Average the positive_sweep heatmaps for a given dimension."""
    matrices = [
        np.array(r["positive_sweep"])
        for r in results
        if r["dimension"] == dimension and "positive_sweep" in r
    ]
    if not matrices:
        return None
    return np.mean(matrices, axis=0)


def compute_diagonal_oa(heatmap):
    """Extract the diagonal (source_layer == target_layer) Override Accuracy."""
    n = min(heatmap.shape)
    return [heatmap[i, i] for i in range(n)]


def compute_mean_attn(attn_results, dimension: str):
    """Average attention-by-layer vectors for a given dimension."""
    vectors = [
        r["attn_by_layer"]
        for r in attn_results
        if r["dimension"] == dimension and "attn_by_layer" in r
    ]
    if not vectors:
        return None
    max_len = max(len(v) for v in vectors)
    padded = [v + [float("nan")] * (max_len - len(v)) for v in vectors]
    arr = np.array(padded, dtype=float)
    return np.nanmean(arr, axis=0)


def plot_heatmaps(patching_results, output_dir: Path, model_name: str):
    """Figure 1: Side-by-side Override Accuracy heatmaps."""
    dims = [d for d in DIMENSION_ORDER
            if any(r["dimension"] == d for r in patching_results)]
    if not dims:
        print("No patching data found.")
        return

    fig, axes = plt.subplots(1, len(dims), figsize=(5 * len(dims), 4.5))
    if len(dims) == 1:
        axes = [axes]

    for ax, dim in zip(axes, dims):
        heatmap = compute_mean_heatmap(patching_results, dim)
        if heatmap is None:
            ax.set_visible(False)
            continue

        im = ax.imshow(heatmap, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto",
                       origin="lower")
        ax.set_title(DIMENSION_LABELS.get(dim, dim), fontsize=12, fontweight="bold")
        ax.set_xlabel("Target Layer", fontsize=10)
        ax.set_ylabel("Source Layer", fontsize=10)
        plt.colorbar(im, ax=ax, label="Override Accuracy")

    fig.suptitle(
        f"Fusion Band Consistency Across Visual Dimensions\n({model_name})",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    out = output_dir / "heatmaps_by_dimension.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    print(f"Saved heatmaps → {out}")
    plt.close()


def plot_corroboration(patching_results, attn_results,
                       output_dir: Path, model_name: str):
    """Figure 2: Override Accuracy diagonal vs. Attention Weight per layer."""
    dims = [d for d in DIMENSION_ORDER
            if any(r["dimension"] == d for r in patching_results)]
    if not dims:
        return

    fig, axes = plt.subplots(1, len(dims), figsize=(5 * len(dims), 4))
    if len(dims) == 1:
        axes = [axes]

    for ax, dim in zip(axes, dims):
        heatmap = compute_mean_heatmap(patching_results, dim)
        if heatmap is None:
            ax.set_visible(False)
            continue

        oa_diag = compute_diagonal_oa(heatmap)
        layers = list(range(len(oa_diag)))

        color = DIMENSION_COLORS.get(dim, "steelblue")
        ax.plot(layers, oa_diag, color=color, linewidth=2,
                label="Override Accuracy (causal)")
        ax.set_ylabel("Override Accuracy", color=color, fontsize=10)
        ax.tick_params(axis="y", labelcolor=color)

        if attn_results:
            attn_vec = compute_mean_attn(attn_results, dim)
            if attn_vec is not None:
                # Normalise attention to [0, 1] for overlay
                attn_norm = np.array(attn_vec[:len(layers)], dtype=float)
                finite = attn_norm[np.isfinite(attn_norm)]
                if len(finite) > 0:
                    attn_norm = (attn_norm - finite.min()) / (
                        finite.max() - finite.min() + 1e-9)
                ax2 = ax.twinx()
                ax2.plot(layers, attn_norm[:len(layers)], color="gray",
                         linewidth=1.5, linestyle="--",
                         label="Text→Image Attention (normalised)")
                ax2.set_ylabel("Attention Weight (normalised)", color="gray",
                               fontsize=9)
                ax2.tick_params(axis="y", labelcolor="gray")

        ax.set_title(DIMENSION_LABELS.get(dim, dim), fontsize=11,
                     fontweight="bold")
        ax.set_xlabel("Decoder Layer", fontsize=10)
        ax.set_xlim(0, len(oa_diag) - 1)

    fig.suptitle(
        f"Cross-Metric Corroboration: Causal Patching vs. Attention\n({model_name})",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    out = output_dir / "corroboration_causal_vs_attention.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    print(f"Saved corroboration plot → {out}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot baseline fusion results")
    parser.add_argument("--patching_results",  type=str, required=True)
    parser.add_argument("--attention_results", type=str, default=None)
    parser.add_argument("--output_dir",        type=str, default="figures/baseline_fusion")
    parser.add_argument("--model_name",        type=str, default="VLM")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    patching = load_patching(args.patching_results)
    attn = load_attention(args.attention_results) if args.attention_results else []

    plot_heatmaps(patching, out_dir, args.model_name)
    plot_corroboration(patching, attn, out_dir, args.model_name)


if __name__ == "__main__":
    main()
