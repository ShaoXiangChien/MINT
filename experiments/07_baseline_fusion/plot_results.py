"""Visualization for Baseline Fusion Experiment (Section 5.1).

Generates TWO complementary figures per run:

1. ``fusion_band_diagnostic_curve.pdf``  (Main text)
   Combined multi-panel 1D line chart — one panel per model.
   Each panel overlays three Override Accuracy (OA) diagonal curves
   (Object, Attribute, Spatial) plus an aggregate text-to-image attention
   line, and shades the auto-detected Normal Fusion Band.

2. ``appendix_heatmaps.pdf``  (Appendix)
   Full 2D patching heatmaps arranged as a grid of
   (dimensions × models) panels, coloured with ``viridis``.
   A white dashed diagonal marks where the 1D curve was extracted.

Because ``run_experiment.py`` generates one JSON per dataset
(pope / gqa / whatsup), ``--patching_results`` and
``--attention_results`` each accept a comma-separated list of files
as a single quoted argument per model.

Usage::

    python experiments/07_baseline_fusion/plot_results.py \\
        --model_labels "Qwen2.5-VL-7B" "LLaVA-OneVision-7B" "InternVL2.5-8B" \\
        --model_keys    qwen25           llava_onevision       internvl25 \\
        --patching_results \\
            "results/baseline_fusion_qwen25_pope.json,results/baseline_fusion_qwen25_gqa.json,results/baseline_fusion_qwen25_whatsup.json" \\
            "results/baseline_fusion_llava_onevision_pope.json,results/baseline_fusion_llava_onevision_gqa.json,results/baseline_fusion_llava_onevision_whatsup.json" \\
            "results/baseline_fusion_internvl25_pope.json,results/baseline_fusion_internvl25_gqa.json,results/baseline_fusion_internvl25_whatsup.json" \\
        --output_dir results/figures/

    # Once attention corroboration JSONs are ready, add:
        --attention_results \\
            "results/attention_corroboration_qwen25.json" \\
            "results/attention_corroboration_llava_onevision.json" \\
            "results/attention_corroboration_internvl25.json"
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json

import numpy as np
import matplotlib.pyplot as plt

from evaluation.plot_utils import set_paper_style, MODEL_LAYERS, get_layer_ticks


# ---------------------------------------------------------------------------
# Constants
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
# Data loading
# ---------------------------------------------------------------------------

def load_patching(path: str):
    with open(path) as f:
        return json.load(f)


def load_attention(path: str):
    with open(path) as f:
        return json.load(f)


def load_patching_group(group_str: str):
    """Merge one or more patching files given as a comma-separated string.

    Each file may cover a single dataset/dimension (pope / gqa / whatsup).
    All samples are concatenated so compute_oa_curves sees all three
    dimensions in one list.
    """
    merged = []
    for path in group_str.split(","):
        path = path.strip()
        if path:
            merged.extend(load_patching(path))
    return merged


def load_attention_group(group_str: str):
    """Merge one or more attention corroboration files."""
    merged = []
    for path in group_str.split(","):
        path = path.strip()
        if path:
            merged.extend(load_attention(path))
    return merged


# ---------------------------------------------------------------------------
# Computation helpers
# ---------------------------------------------------------------------------

def compute_mean_heatmap(results, dimension: str):
    """Average the positive_sweep matrices for a given dimension."""
    matrices = [
        np.array(r["positive_sweep"])
        for r in results
        if r.get("dimension") == dimension and "positive_sweep" in r
    ]
    if not matrices:
        return None
    return np.mean(matrices, axis=0)


def compute_diagonal_oa(heatmap):
    """Extract the diagonal (source_layer == target_layer) OA values."""
    n = min(heatmap.shape)
    return [heatmap[i, i] for i in range(n)]


def compute_mean_attn(attn_results, dimension: str):
    """Average attention-by-layer vectors for a given dimension."""
    vectors = [
        r["attn_by_layer"]
        for r in attn_results
        if r.get("dimension") == dimension and "attn_by_layer" in r
    ]
    if not vectors:
        return None
    max_len = max(len(v) for v in vectors)
    padded  = [v + [float("nan")] * (max_len - len(v)) for v in vectors]
    return np.nanmean(np.array(padded, dtype=float), axis=0)


def compute_oa_curves(patching_results):
    """Mean diagonal OA curve per dimension.

    Returns dict: dimension -> 1-D np.ndarray (diagonal index space),
    or None if no samples exist for that dimension.
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
        padded  = np.full((len(diagonals), max_len), np.nan)
        for i, d in enumerate(diagonals):
            padded[i, :len(d)] = d
        curves[dim] = np.nanmean(padded, axis=0)

    return curves


def detect_fusion_band(oa_curves, model_key,
                       band_start_override=None, band_end_override=None):
    """Return (band_start, band_end) in actual layer numbers.

    Auto-detects the contiguous region where cross-dimension mean OA ≥ 50%
    of its peak, then converts diagonal indices to actual layer numbers
    (index × step).  CLI overrides bypass detection entirely.
    """
    if band_start_override is not None and band_end_override is not None:
        return band_start_override, band_end_override

    step = MODEL_LAYERS.get(model_key, {"step": 1})["step"]
    valid = [c for c in oa_curves.values() if c is not None]

    if not valid:
        print("WARNING: No OA curves for band detection; using full range.")
        total = MODEL_LAYERS.get(model_key, {"total": 28})["total"]
        return 0, total - 1

    max_len = max(len(c) for c in valid)
    padded  = np.full((len(valid), max_len), np.nan)
    for i, c in enumerate(valid):
        padded[i, :len(c)] = c
    mean_curve = np.nanmean(padded, axis=0)

    peak = np.nanmax(mean_curve)
    if peak == 0 or np.isnan(peak):
        print("WARNING: Peak OA is zero/NaN; using full range for fusion band.")
        return 0, int((max_len - 1) * step)

    above = np.where(mean_curve >= 0.5 * peak)[0]
    start_layer = band_start_override if band_start_override is not None \
                  else int(above[0])  * step
    end_layer   = band_end_override   if band_end_override   is not None \
                  else int(above[-1]) * step
    return start_layer, end_layer


# ---------------------------------------------------------------------------
# Figure 1 (Main text): 1D diagnostic curve
# ---------------------------------------------------------------------------

def plot_fusion_band_diagnostic_curve(per_model_data, band_start, band_end,
                                      output_dir):
    """Multi-panel 1D OA + attention curves with Fusion Band shading.

    Parameters
    ----------
    per_model_data : list of (label, model_key, oa_curves, attn_results, patching)
    band_start, band_end : int, actual layer numbers
    output_dir : Path
    """
    n = len(per_model_data)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    ax2_last = None

    for panel_idx, (ax, (label, model_key, oa_curves, attn_results, _)) in \
            enumerate(zip(axes, per_model_data)):

        step = MODEL_LAYERS.get(model_key, {"step": 1})["step"]
        valid_curves = [c for c in oa_curves.values() if c is not None]
        num_diag  = max((len(c) for c in valid_curves), default=0)
        max_layer = num_diag * step   # exclusive upper bound in actual layers

        # Fusion band (drawn first so it sits behind everything)
        ax.axvspan(band_start, band_end,
                   color="#FFFDE7", alpha=0.5, label="Normal Fusion Band", zorder=0)

        # Three OA curves
        for dim in DIMENSION_ORDER:
            curve = oa_curves.get(dim)
            if curve is None:
                continue
            x = np.arange(len(curve)) * step
            ax.plot(x, curve,
                    color=DIMENSION_COLORS[dim], linewidth=2,
                    label=DIMENSION_LABELS[dim].replace("\n", " "),
                    zorder=2)

        # Aggregate attention overlay (single dashed gray line)
        if attn_results:
            all_vecs = [r["attn_by_layer"] for r in attn_results
                        if "attn_by_layer" in r]
            if all_vecs:
                pad_len = max(len(v) for v in all_vecs)
                padded  = np.full((len(all_vecs), pad_len), np.nan)
                for i, v in enumerate(all_vecs):
                    padded[i, :len(v)] = v
                mean_attn = np.nanmean(padded, axis=0)
                x_attn    = np.arange(len(mean_attn)) * step

                ax2 = ax.twinx()
                ax2.plot(x_attn, mean_attn,
                         color="#555555", linestyle="--", linewidth=1.5,
                         label="Text→Image Attention", zorder=1)
                ax2.set_ylim(bottom=0)
                ax2.tick_params(axis="y", labelcolor="#555555", labelsize=9)
                if panel_idx == n - 1:
                    ax2.set_ylabel("Attention Weight", fontsize=14, color="#555555")
                else:
                    ax2.set_yticklabels([])
                ax2_last = ax2

        # Ticks and labels
        ticks, tick_labels = get_layer_ticks(model_key, max_layers=max_layer)
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels, fontsize=9)
        ax.set_xlim(0, max(max_layer - step, 1))
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
    axes[0].legend(handles, labels_leg, fontsize=11,
                   loc="upper left", framealpha=0.85)

    fig.suptitle("Fusion Band Consistency Across Models",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    stem = "fusion_band_diagnostic_curve"
    fig.savefig(output_dir / f"{stem}.pdf")
    fig.savefig(output_dir / f"{stem}.png")
    plt.close(fig)
    print(f"Saved 1D diagnostic curve  → {output_dir / stem}.pdf")


# ---------------------------------------------------------------------------
# Figure 2 (Appendix): 2D heatmap grid
# ---------------------------------------------------------------------------

def plot_appendix_heatmaps(per_model_data, output_dir):
    """Grid of 2D Override Accuracy heatmaps for the appendix.

    Layout: ``n_dims`` rows (Object / Attribute / Spatial) ×
            ``n_models`` columns.
    Coloured with ``viridis`` (0 = dark blue, 1 = yellow).
    A white dashed diagonal marks the slice used for the 1D diagnostic curve.

    Parameters
    ----------
    per_model_data : list of (label, model_key, oa_curves, attn_results, patching)
    output_dir : Path
    """
    n_models = len(per_model_data)
    n_dims   = len(DIMENSION_ORDER)

    fig, axes = plt.subplots(
        n_dims, n_models,
        figsize=(4.5 * n_models, 4.0 * n_dims),
        squeeze=False,
    )

    im_ref = None

    for col_idx, (label, model_key, _, _, patching) in enumerate(per_model_data):
        step = MODEL_LAYERS.get(model_key, {"step": 1})["step"]

        for row_idx, dim in enumerate(DIMENSION_ORDER):
            ax      = axes[row_idx][col_idx]
            heatmap = compute_mean_heatmap(patching, dim)

            if heatmap is None:
                ax.set_visible(False)
                continue

            n_src, n_tgt = heatmap.shape

            im = ax.imshow(
                heatmap, vmin=0, vmax=1, cmap="viridis",
                aspect="auto", origin="lower",
            )
            im_ref = im

            # White dashed diagonal — shows where the 1D curve was taken from
            n_diag = min(n_src, n_tgt)
            ax.plot([0, n_diag - 1], [0, n_diag - 1],
                    color="white", linestyle="--", linewidth=1.2,
                    alpha=0.8, label="Diagonal (1D curve)")

            # Tick labels: show every pixel (actual layer numbers)
            # Use a stride if there are too many ticks (> 8)
            stride = max(1, n_tgt // 8)
            tgt_ticks = list(range(0, n_tgt, stride))
            src_ticks = list(range(0, n_src, stride))
            ax.set_xticks(tgt_ticks)
            ax.set_xticklabels(
                [str(t * step) for t in tgt_ticks], fontsize=8, rotation=45)
            ax.set_yticks(src_ticks)
            ax.set_yticklabels(
                [str(t * step) for t in src_ticks], fontsize=8)

            # Column header = model name (top row only)
            if row_idx == 0:
                ax.set_title(label, fontsize=13, fontweight="bold", pad=8)

            # Row label = dimension (leftmost column only)
            if col_idx == 0:
                ax.set_ylabel(
                    DIMENSION_LABELS[dim].replace("\n", " ") + "\n(Source Layer)",
                    fontsize=10,
                )

            # X-axis label (bottom row only)
            if row_idx == n_dims - 1:
                ax.set_xlabel("Target Layer", fontsize=11)

    # Single shared colorbar on the right
    if im_ref is not None:
        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.70])
        cb = fig.colorbar(im_ref, cax=cbar_ax)
        cb.set_label("Override Accuracy", fontsize=11)
        cb.ax.tick_params(labelsize=9)

    fig.suptitle(
        "Appendix — Override Accuracy Heatmaps (Source Layer × Target Layer)\n"
        "White dashed diagonal = slice used for 1D diagnostic curve",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 0.89, 0.96])

    stem = "appendix_heatmaps"
    fig.savefig(output_dir / f"{stem}.pdf")
    fig.savefig(output_dir / f"{stem}.png")
    plt.close(fig)
    print(f"Saved 2D appendix heatmaps → {output_dir / stem}.pdf")


# ---------------------------------------------------------------------------
# Legacy figure functions (used by --also_heatmaps for per-model corroboration)
# ---------------------------------------------------------------------------

def plot_heatmaps(patching_results, output_dir: Path, model_name: str):
    """Side-by-side OA heatmaps (original legacy figure)."""
    dims = [d for d in DIMENSION_ORDER
            if any(r.get("dimension") == d for r in patching_results)]
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
        im = ax.imshow(heatmap, vmin=0, vmax=1, cmap="RdYlGn",
                       aspect="auto", origin="lower")
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
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight")
    print(f"Saved legacy heatmaps → {out}")
    plt.close()


def plot_corroboration(patching_results, attn_results,
                       output_dir: Path, model_name: str):
    """OA diagonal vs. attention weight per layer (original legacy figure)."""
    dims = [d for d in DIMENSION_ORDER
            if any(r.get("dimension") == d for r in patching_results)]
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
        layers  = list(range(len(oa_diag)))
        color   = DIMENSION_COLORS.get(dim, "steelblue")

        ax.plot(layers, oa_diag, color=color, linewidth=2,
                label="Override Accuracy (causal)")
        ax.set_ylabel("Override Accuracy", color=color, fontsize=10)
        ax.tick_params(axis="y", labelcolor=color)

        if attn_results:
            attn_vec = compute_mean_attn(attn_results, dim)
            if attn_vec is not None:
                attn_norm  = np.array(attn_vec[:len(layers)], dtype=float)
                finite     = attn_norm[np.isfinite(attn_norm)]
                if len(finite) > 0:
                    attn_norm = (attn_norm - finite.min()) / \
                                (finite.max() - finite.min() + 1e-9)
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
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight")
    print(f"Saved legacy corroboration plot → {out}")
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

    parser.add_argument(
        "--model_labels", nargs="+", required=True,
        metavar="LABEL",
        help='Display names, one per model, e.g. "Qwen2.5-VL-7B"',
    )
    parser.add_argument(
        "--model_keys", nargs="+", required=True,
        metavar="KEY",
        choices=list(MODEL_LAYERS.keys()),
        help=f"Model keys for layer config; one per model. "
             f"Choices: {list(MODEL_LAYERS.keys())}",
    )
    parser.add_argument(
        "--patching_results", nargs="+", required=True,
        metavar="FILE[,FILE,...]",
        help=(
            "One entry per model — comma-separated list of "
            "baseline_fusion JSON files (pope/gqa/whatsup) as a single quoted "
            'string, e.g. "results/baseline_fusion_qwen25_pope.json,'
            'results/baseline_fusion_qwen25_gqa.json,'
            'results/baseline_fusion_qwen25_whatsup.json"'
        ),
    )
    parser.add_argument(
        "--attention_results", nargs="*", default=[],
        metavar="FILE[,FILE,...]",
        help=(
            "One entry per model (optional). Comma-separated list of "
            "attention_corroboration JSON files."
        ),
    )
    parser.add_argument("--band_start", type=int, default=None,
                        help="Override fusion band start (actual layer number)")
    parser.add_argument("--band_end",   type=int, default=None,
                        help="Override fusion band end (actual layer number)")
    parser.add_argument("--output_dir", type=str, default="results/figures/",
                        help="Directory to save figures")
    parser.add_argument(
        "--also_heatmaps", action="store_true",
        help="Also generate per-model legacy OA heatmap and corroboration PDFs",
    )

    args = parser.parse_args()

    n = len(args.model_labels)
    if len(args.model_keys) != n or len(args.patching_results) != n:
        parser.error(
            "--model_labels, --model_keys, and --patching_results must all "
            "have the same number of entries."
        )

    # Pad attention_results with empty strings if not supplied for all models
    attn_paths = list(args.attention_results) + [""] * (n - len(args.attention_results))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_paper_style()

    # ------------------------------------------------------------------
    # Load all data
    # ------------------------------------------------------------------
    per_model_data = []   # (label, model_key, oa_curves, attn_results, patching)
    first_oa_curves  = None
    first_model_key  = None

    for label, key, p_group, a_group in zip(
        args.model_labels, args.model_keys, args.patching_results, attn_paths
    ):
        patching = load_patching_group(p_group)
        attn     = load_attention_group(a_group) if a_group else []
        print(f"[{label}] {len(patching)} patching samples, "
              f"{len(attn)} attention samples")
        oa_curves = compute_oa_curves(patching)
        per_model_data.append((label, key, oa_curves, attn, patching))

        if first_oa_curves is None:
            first_oa_curves = oa_curves
            first_model_key = key

    # ------------------------------------------------------------------
    # Detect (or accept CLI override) fusion band from first model
    # ------------------------------------------------------------------
    band_start, band_end = detect_fusion_band(
        first_oa_curves, first_model_key,
        args.band_start, args.band_end,
    )
    print(f"\nNormal Fusion Band: layers {band_start}–{band_end}")
    print(f"  → pass  --band_start {band_start} --band_end {band_end}  "
          f"to plot_pathology.py\n")

    # ------------------------------------------------------------------
    # Figure 1 (Main text) — 1D diagnostic curve
    # ------------------------------------------------------------------
    plot_fusion_band_diagnostic_curve(
        per_model_data, band_start, band_end, out_dir)

    # ------------------------------------------------------------------
    # Figure 2 (Appendix) — 2D heatmap grid
    # ------------------------------------------------------------------
    plot_appendix_heatmaps(per_model_data, out_dir)

    # ------------------------------------------------------------------
    # Optional: legacy per-model corroboration figures
    # ------------------------------------------------------------------
    if args.also_heatmaps:
        for label, key, _, _, patching in per_model_data:
            # find matching attn group
            idx     = [lbl for lbl, *_ in per_model_data].index(label)
            a_group = attn_paths[idx]
            attn    = load_attention_group(a_group) if a_group else []

            model_out = out_dir / key
            model_out.mkdir(parents=True, exist_ok=True)
            plot_heatmaps(patching, model_out, label)
            if attn:
                plot_corroboration(patching, attn, model_out, label)


if __name__ == "__main__":
    main()
