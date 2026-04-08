"""Visualization for Baseline Fusion Experiment (Section 5.1).

Generates a combined multi-panel figure showing the Fusion Band is consistent
across three visual dimensions (Object, Attribute, Spatial) and across models.
Each panel shows Override Accuracy curves for the three dimensions plus an
aggregate text-to-image attention overlay.

Optionally also generates per-model OA heatmaps and corroboration plots
(the original figures) via --also_heatmaps.

Usage::

    python experiments/07_baseline_fusion/plot_results.py \\
        --model_labels "LLaVA-1.5-7B" "Qwen2-VL-7B" "InternVL2.5-8B" \\
        --model_keys   llava qwen internvl \\
        --patching_results  results/baseline_fusion_llava.json \\
                            results/baseline_fusion_qwen.json \\
                            results/baseline_fusion_internvl.json \\
        --attention_results results/attention_corroboration_llava.json \\
                            results/attention_corroboration_qwen.json \\
                            results/attention_corroboration_internvl.json \\
        --output_dir results/figures/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from evaluation.plot_utils import set_paper_style, MODEL_LAYERS, get_layer_ticks


# ---------------------------------------------------------------------------
# Constants (unchanged from original)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Data loading (unchanged from original)
# ---------------------------------------------------------------------------

def load_patching(path: str):
    with open(path) as f:
        return json.load(f)


def load_attention(path: str):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Computation helpers (unchanged from original)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# New computation helpers
# ---------------------------------------------------------------------------

def compute_oa_curves(patching_results):
    """Compute mean diagonal OA curve per dimension.

    Returns a dict mapping dimension name -> 1-D np.ndarray of mean Override
    Accuracy values (one entry per diagonal index).  Values are in [0, 1].
    Returns None for a dimension if no samples exist.
    """
    curves = {}
    for dim in DIMENSION_ORDER:
        diagonals = []
        for r in patching_results:
            if r.get("dimension") != dim:
                continue
            sweep = r.get("positive_sweep")
            if sweep is None:
                continue
            mat = np.array(sweep, dtype=float)
            if mat.ndim != 2 or mat.size == 0:
                continue
            n = min(mat.shape)
            diagonals.append(np.array([mat[i, i] for i in range(n)]))

        if not diagonals:
            curves[dim] = None
            continue

        max_len = max(len(d) for d in diagonals)
        padded = np.full((len(diagonals), max_len), np.nan)
        for i, d in enumerate(diagonals):
            padded[i, :len(d)] = d
        curves[dim] = np.nanmean(padded, axis=0)

    return curves


def detect_fusion_band(oa_curves, model_key, band_start_override=None, band_end_override=None):
    """Detect the Normal Fusion Band layer range in actual layer numbers.

    If both overrides are provided, returns them directly.  Otherwise detects
    the contiguous region where the cross-dimension mean OA exceeds 50% of
    its peak value and converts diagonal indices to actual layer numbers.

    Returns (band_start, band_end) in actual layer numbers.
    """
    if band_start_override is not None and band_end_override is not None:
        return band_start_override, band_end_override

    step = MODEL_LAYERS.get(model_key, {"step": 1})["step"]

    valid = [c for c in oa_curves.values() if c is not None]
    if not valid:
        print("WARNING: No OA curves available for fusion band detection; using full range.")
        total = MODEL_LAYERS.get(model_key, {"total": 28})["total"]
        return 0, total - 1

    max_len = max(len(c) for c in valid)
    padded = np.full((len(valid), max_len), np.nan)
    for i, c in enumerate(valid):
        padded[i, :len(c)] = c
    mean_curve = np.nanmean(padded, axis=0)

    peak = np.nanmax(mean_curve)
    if peak == 0 or np.isnan(peak):
        print("WARNING: Peak OA is zero or NaN; using full range for fusion band.")
        return 0, int((max_len - 1) * step)

    threshold = 0.5 * peak
    above = np.where(mean_curve >= threshold)[0]

    # Handle partial overrides
    if band_start_override is not None:
        start_layer = band_start_override
    else:
        start_layer = int(above[0]) * step

    if band_end_override is not None:
        end_layer = band_end_override
    else:
        end_layer = int(above[-1]) * step

    return start_layer, end_layer


# ---------------------------------------------------------------------------
# New combined multi-panel figure
# ---------------------------------------------------------------------------

def plot_combined_fusion_band(per_model_data, band_start, band_end, output_dir):
    """Generate the combined multi-panel fusion band figure (Section 5.1).

    Parameters
    ----------
    per_model_data : list of (label, model_key, oa_curves, attn_results)
    band_start     : int, actual layer number
    band_end       : int, actual layer number
    output_dir     : Path
    """
    n = len(per_model_data)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    ax2_last = None  # track rightmost twin axis for label

    for panel_idx, (ax, (label, model_key, oa_curves, attn_results)) in enumerate(
        zip(axes, per_model_data)
    ):
        step = MODEL_LAYERS.get(model_key, {"step": 1})["step"]

        # Determine x range for this model
        valid_curves = [c for c in oa_curves.values() if c is not None]
        num_diag = max((len(c) for c in valid_curves), default=0)
        max_layer = num_diag * step  # exclusive upper bound

        # Fusion band shading (drawn first so it's behind everything)
        ax.axvspan(band_start, band_end, color="#FFFDE7", alpha=0.5,
                   label="Normal Fusion Band", zorder=0)

        # OA curves — one per dimension
        for dim in DIMENSION_ORDER:
            curve = oa_curves.get(dim)
            if curve is None:
                continue
            x = np.arange(len(curve)) * step
            ax.plot(x, curve,
                    color=DIMENSION_COLORS[dim],
                    linewidth=2,
                    label=DIMENSION_LABELS[dim].replace("\n", " "),
                    zorder=2)

        # Aggregate attention overlay (single dashed gray line)
        if attn_results:
            all_vecs = [r["attn_by_layer"] for r in attn_results
                        if "attn_by_layer" in r]
            if all_vecs:
                max_len = max(len(v) for v in all_vecs)
                padded = np.full((len(all_vecs), max_len), np.nan)
                for i, v in enumerate(all_vecs):
                    padded[i, :len(v)] = v
                mean_attn = np.nanmean(padded, axis=0)
                x_attn = np.arange(len(mean_attn)) * step

                ax2 = ax.twinx()
                ax2.plot(x_attn, mean_attn,
                         color="#555555", linestyle="--", linewidth=1.5,
                         label="Text→Image Attention", zorder=1)
                ax2.set_ylim(bottom=0)
                ax2.tick_params(axis="y", labelcolor="#555555", labelsize=9)

                # Only the rightmost panel gets the secondary Y label
                if panel_idx == n - 1:
                    ax2.set_ylabel("Attention Weight", fontsize=14,
                                   color="#555555")
                else:
                    ax2.set_yticklabels([])

                ax2_last = ax2

        # X-axis ticks
        ticks, tick_labels = get_layer_ticks(model_key, max_layers=max_layer)
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels, fontsize=9)
        ax.set_xlim(0, max_layer - step)

        # Y-axis
        ax.set_ylim(0, 1)
        ax.set_xlabel("Decoder Layer", fontsize=14)
        if panel_idx == 0:
            ax.set_ylabel("Override Accuracy (OA)", fontsize=14)

        ax.set_title(label, fontsize=14, fontweight="bold")

    # Combined legend on first panel
    handles, labels_leg = axes[0].get_legend_handles_labels()
    if ax2_last is not None:
        h2, l2 = ax2_last.get_legend_handles_labels()
        handles += h2
        labels_leg += l2
    axes[0].legend(handles, labels_leg, fontsize=11, loc="upper left",
                   framealpha=0.8)

    fig.suptitle("Fusion Band Consistency Across Models",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    stem = "fusion_band_combined"
    fig.savefig(output_dir / f"{stem}.pdf")
    fig.savefig(output_dir / f"{stem}.png")
    plt.close(fig)
    print(f"Saved combined fusion band figure → {output_dir / stem}.pdf")


# ---------------------------------------------------------------------------
# Legacy figure functions (unchanged from original, used by --also_heatmaps)
# ---------------------------------------------------------------------------

def plot_heatmaps(patching_results, output_dir: Path, model_name: str):
    """Figure: Side-by-side Override Accuracy heatmaps."""
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
    """Figure: Override Accuracy diagonal vs. Attention Weight per layer."""
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot baseline fusion results (Section 5.1 — Anatomy)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Multi-model inputs
    parser.add_argument(
        "--model_labels", nargs="+", required=True,
        metavar="LABEL",
        help='Display names, one per model (e.g. "LLaVA-1.5-7B")',
    )
    parser.add_argument(
        "--model_keys", nargs="+", required=True,
        metavar="KEY",
        choices=list(MODEL_LAYERS.keys()),
        help=f"Model keys for layer config; one per model. Choices: {list(MODEL_LAYERS.keys())}",
    )
    parser.add_argument(
        "--patching_results", nargs="+", required=True,
        metavar="FILE",
        help="Path to baseline_fusion_<model>.json, one per model",
    )
    parser.add_argument(
        "--attention_results", nargs="*", default=[],
        metavar="FILE",
        help="Path to attention_corroboration_<model>.json, one per model (optional)",
    )

    # Fusion band override
    parser.add_argument("--band_start", type=int, default=None,
                        help="Override fusion band start (actual layer number)")
    parser.add_argument("--band_end",   type=int, default=None,
                        help="Override fusion band end (actual layer number)")

    # Output
    parser.add_argument("--output_dir", type=str, default="results/figures/",
                        help="Directory to save figures")

    # Legacy heatmap option
    parser.add_argument("--also_heatmaps", action="store_true",
                        help="Also generate per-model OA heatmap and corroboration PDFs")

    args = parser.parse_args()

    # Validate list lengths
    n = len(args.model_labels)
    if len(args.model_keys) != n or len(args.patching_results) != n:
        parser.error(
            "--model_labels, --model_keys, and --patching_results must all have "
            "the same number of entries."
        )

    # Pad attention_results with empty strings if shorter
    attn_paths = list(args.attention_results) + [""] * (n - len(args.attention_results))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_paper_style()

    # Load data and build per-model structures
    per_model_data = []
    first_oa_curves = None
    first_model_key = None

    for label, key, p_path, a_path in zip(
        args.model_labels, args.model_keys, args.patching_results, attn_paths
    ):
        patching = load_patching(p_path)
        attn = load_attention(a_path) if a_path else []
        oa_curves = compute_oa_curves(patching)
        per_model_data.append((label, key, oa_curves, attn))

        if first_oa_curves is None:
            first_oa_curves = oa_curves
            first_model_key = key

    # Detect (or accept) fusion band from the first model's data
    band_start, band_end = detect_fusion_band(
        first_oa_curves, first_model_key,
        args.band_start, args.band_end,
    )
    print(f"Normal Fusion Band: layers {band_start}–{band_end}")
    print(f"  (pass --band_start {band_start} --band_end {band_end} to plot_pathology.py)")

    # Main combined figure
    plot_combined_fusion_band(per_model_data, band_start, band_end, out_dir)

    # Optional legacy per-model heatmaps
    if args.also_heatmaps:
        for label, key, p_path, a_path in zip(
            args.model_labels, args.model_keys, args.patching_results, attn_paths
        ):
            model_out = out_dir / key
            model_out.mkdir(parents=True, exist_ok=True)
            patching = load_patching(p_path)
            attn = load_attention(a_path) if a_path else []
            plot_heatmaps(patching, model_out, label)
            if attn:
                plot_corroboration(patching, attn, model_out, label)


if __name__ == "__main__":
    main()
