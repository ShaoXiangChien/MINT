"""Visualization for Pathology Diagnosis Experiment (Section 5.2).

Generates TWO complementary figures per run:

1. ``pathology_diagnostic_curve.pdf``  (Main text)
   1D Flip Rate curves (mean ± 1 std across failed samples) per series,
   overlaid on the Normal Fusion Band from Section 5.1.
   Optional arrows label "Prior Override" (peak left of band) or
   "Late Activation" (peak right of band).

2. ``pathology_heatmap.pdf``  (Appendix)
   Full 2D mean Flip Rate heatmaps — one panel per series.
   Coloured with ``viridis``.  Vertical dotted lines mark the Normal Fusion
   Band boundaries.  A white dashed diagonal marks where the 1D curve was
   extracted.

Only samples where ``baseline_correct == false`` are processed.

Usage::

    python experiments/08_pathology_diagnosis/plot_pathology.py \\
        --results \\
            "NaturalBench-Qwen:results/08_pathology/qwen25_naturalbench.json" \\
            "MINDCUBE-Qwen:results/08_pathology/qwen25_mindcube.json" \\
            "NaturalBench-LLaVA:results/08_pathology/llava_onevision_naturalbench.json" \\
            "MINDCUBE-LLaVA:results/08_pathology/llava_onevision_mindcube.json" \\
            "NaturalBench-InternVL:results/08_pathology/internvl25_naturalbench.json" \\
            "MINDCUBE-InternVL:results/08_pathology/internvl25_mindcube.json" \\
        --band_start 9 --band_end 18 \\
        --model_key qwen25 --annotate \\
        --output_dir results/figures/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from evaluation.plot_utils import set_paper_style, MODEL_LAYERS, get_layer_ticks


# ---------------------------------------------------------------------------
# Argument helpers
# ---------------------------------------------------------------------------

def parse_label_path(token: str):
    """Parse a ``LABEL:PATH`` token into (label, Path)."""
    label, sep, path_str = token.partition(":")
    if not sep:
        raise argparse.ArgumentTypeError(
            f"Expected LABEL:PATH format, got: {token!r}"
        )
    return label, Path(path_str)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pathology_filtered(path: Path):
    """Load pathology JSON; return only samples where the model failed.

    Samples missing ``baseline_correct`` are treated as correct (excluded).
    """
    with open(path) as f:
        data = json.load(f)

    missing = sum(1 for s in data if "baseline_correct" not in s)
    if missing:
        print(f"  WARNING: {missing} sample(s) missing 'baseline_correct' — excluded.")

    filtered = [s for s in data if not s.get("baseline_correct", True)]
    print(f"  {len(filtered)} failed / {len(data)} total samples from {path.name}")
    return filtered


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def compute_flip_rate_curve(samples):
    """Mean and std of the diagonal flip rate across failed samples.

    Returns
    -------
    mean_curve, std_curve : np.ndarray, shape (num_diag_layers,)
        Both in [0, 1].  Returns two empty arrays if no valid samples.
    """
    diagonals = []
    for s in samples:
        sweep = s.get("patching_sweep")
        if sweep is None:
            continue
        mat = np.array(sweep, dtype=float)
        if mat.ndim != 2 or mat.size == 0:
            continue
        n = min(mat.shape)
        diagonals.append(np.array([mat[i, i] for i in range(n)]))

    if not diagonals:
        return np.array([]), np.array([])

    max_len = max(len(d) for d in diagonals)
    padded  = np.full((len(diagonals), max_len), np.nan)
    for i, d in enumerate(diagonals):
        padded[i, :len(d)] = d

    return np.nanmean(padded, axis=0), np.nanstd(padded, axis=0)


def compute_mean_flip_heatmap(samples):
    """Average the 2D ``patching_sweep`` matrices across failed samples.

    Returns an np.ndarray of shape (n_source_layers, n_target_layers),
    or None if no valid matrices found.
    """
    matrices = []
    for s in samples:
        sweep = s.get("patching_sweep")
        if sweep is None:
            continue
        mat = np.array(sweep, dtype=float)
        if mat.ndim == 2 and mat.size > 0:
            matrices.append(mat)

    if not matrices:
        return None

    # Pad to the largest shape encountered (handles edge cases)
    max_r = max(m.shape[0] for m in matrices)
    max_c = max(m.shape[1] for m in matrices)
    padded = np.full((len(matrices), max_r, max_c), np.nan)
    for i, m in enumerate(matrices):
        padded[i, :m.shape[0], :m.shape[1]] = m

    return np.nanmean(padded, axis=0)


# ---------------------------------------------------------------------------
# Figure 1 (Main text): 1D diagnostic curve
# ---------------------------------------------------------------------------

def plot_pathology_diagnostic_curve(series, band_start, band_end,
                                    output_dir, model_key, annotate):
    """1D Flip Rate curves with Fusion Band overlay.

    Parameters
    ----------
    series      : list of (label, mean_curve, std_curve)
    band_start  : int, actual layer number (left edge of Normal Fusion Band)
    band_end    : int, actual layer number (right edge)
    output_dir  : Path
    model_key   : str  (determines step and tick spacing)
    annotate    : bool (add Prior Override / Late Activation arrows)
    """
    step = MODEL_LAYERS.get(model_key, {"step": 1})["step"]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Fusion band background (drawn first)
    ax.axvspan(band_start, band_end, color="#FFFDE7", alpha=0.5,
               label="Normal Fusion Band", zorder=0)

    palette = sns.color_palette("colorblind", n_colors=max(len(series), 1))
    num_layers_max = 0

    for (label, mean_curve, std_curve), color in zip(series, palette):
        if len(mean_curve) == 0:
            print(f"  WARNING: empty curve for '{label}', skipping.")
            continue

        x = np.arange(len(mean_curve)) * step
        num_layers_max = max(num_layers_max, int(x[-1]) + step)

        ax.plot(x, mean_curve, color=color, linewidth=2, label=label, zorder=2)

        # ±1 std shading, clipped to [0, 1]
        ax.fill_between(
            x,
            np.clip(mean_curve - std_curve, 0.0, 1.0),
            np.clip(mean_curve + std_curve, 0.0, 1.0),
            color=color, alpha=0.15, zorder=1,
        )

        # Optional pathology annotations
        if annotate:
            peak_idx   = int(np.nanargmax(mean_curve))
            peak_layer = peak_idx * step
            peak_val   = float(mean_curve[peak_idx])

            if peak_layer < band_start:
                ax.annotate(
                    "Prior Override",
                    xy=(peak_layer, peak_val),
                    xytext=(max(peak_layer - 2 * step, 0), peak_val + 0.10),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
                    fontsize=10, ha="center", color="black",
                )
            elif peak_layer > band_end:
                ax.annotate(
                    "Late Activation",
                    xy=(peak_layer, peak_val),
                    xytext=(peak_layer + 2 * step, peak_val + 0.10),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
                    fontsize=10, ha="center", color="black",
                )

    if num_layers_max == 0:
        num_layers_max = MODEL_LAYERS.get(model_key, {"total": 28})["total"]

    ticks, tick_labels = get_layer_ticks(model_key, max_layers=num_layers_max)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_xlim(0, num_layers_max - step)
    ax.set_ylim(0, 1)

    ax.set_xlabel("Decoder Layer", fontsize=14)
    ax.set_ylabel("Flip Rate (Incorrect → Correct)", fontsize=14)
    ax.set_title("Pathology Diagnosis: Where Does Patching Cure Failures?",
                 fontsize=16, fontweight="bold")
    ax.legend(fontsize=12, loc="upper right", framealpha=0.85)
    plt.tight_layout()

    stem = "pathology_diagnostic_curve"
    fig.savefig(output_dir / f"{stem}.pdf")
    fig.savefig(output_dir / f"{stem}.png")
    plt.close(fig)
    print(f"Saved 1D diagnostic curve  → {output_dir / stem}.pdf")


# ---------------------------------------------------------------------------
# Figure 2 (Appendix): 2D heatmap panels
# ---------------------------------------------------------------------------

def plot_pathology_heatmaps(heatmap_series, band_start, band_end,
                             output_dir, model_key):
    """2D Flip Rate heatmaps for the appendix.

    One panel per series.  Layout: up to 3 columns, wrapping to extra rows
    when there are more than 3 series.

    The Normal Fusion Band is marked with vertical dotted lines on the
    Target Layer axis.  A white dashed diagonal marks the slice used for the
    1D diagnostic curve.

    Parameters
    ----------
    heatmap_series : list of (label, mean_heatmap_2d)
    band_start, band_end : int, actual layer numbers
    output_dir : Path
    model_key  : str
    """
    n = len(heatmap_series)
    if n == 0:
        print("WARNING: No heatmap data to plot.")
        return

    step  = MODEL_LAYERS.get(model_key, {"step": 1})["step"]
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.5 * ncols, 4.5 * nrows),
        squeeze=False,
    )

    im_ref = None

    for idx, (label, heatmap) in enumerate(heatmap_series):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        if heatmap is None:
            ax.set_visible(False)
            continue

        n_src, n_tgt = heatmap.shape

        im = ax.imshow(
            heatmap, vmin=0, vmax=1, cmap="viridis",
            aspect="auto", origin="lower",
        )
        im_ref = im

        # White dashed diagonal — shows where the 1D curve was extracted
        n_diag = min(n_src, n_tgt)
        ax.plot([0, n_diag - 1], [0, n_diag - 1],
                color="white", linestyle="--", linewidth=1.2, alpha=0.8)

        # Normal Fusion Band: vertical dotted lines on target-layer axis
        band_start_px = band_start / step
        band_end_px   = band_end   / step
        ax.axvline(band_start_px, color="#FFD700", linestyle=":",
                   linewidth=1.8, alpha=0.9, label="Band start")
        ax.axvline(band_end_px,   color="#FF8C00", linestyle=":",
                   linewidth=1.8, alpha=0.9, label="Band end")

        # Tick labels
        stride    = max(1, n_tgt // 8)
        tgt_ticks = list(range(0, n_tgt, stride))
        src_ticks = list(range(0, n_src, stride))
        ax.set_xticks(tgt_ticks)
        ax.set_xticklabels(
            [str(t * step) for t in tgt_ticks], fontsize=8, rotation=45)
        ax.set_yticks(src_ticks)
        ax.set_yticklabels(
            [str(t * step) for t in src_ticks], fontsize=8)

        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.set_xlabel("Target Layer", fontsize=11)
        ax.set_ylabel("Source Layer", fontsize=11)

        # Legend on first panel only
        if idx == 0:
            ax.legend(fontsize=8, loc="upper left", framealpha=0.7)

    # Hide any unused axes
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    # Shared colorbar on the right
    if im_ref is not None:
        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.70])
        cb = fig.colorbar(im_ref, cax=cbar_ax)
        cb.set_label("Flip Rate (Incorrect → Correct)", fontsize=11)
        cb.ax.tick_params(labelsize=9)

    fig.suptitle(
        "Appendix — Pathology Flip Rate Heatmaps (Source Layer × Target Layer)\n"
        "Dotted lines = Normal Fusion Band  |  Dashed diagonal = 1D curve slice",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 0.89, 0.95])

    stem = "pathology_heatmap"
    fig.savefig(output_dir / f"{stem}.pdf")
    fig.savefig(output_dir / f"{stem}.png")
    plt.close(fig)
    print(f"Saved 2D pathology heatmaps → {output_dir / stem}.pdf")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot pathology diagnosis results (Section 5.2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--results", nargs="+", required=True,
        type=parse_label_path,
        metavar="LABEL:PATH",
        help='One or more LABEL:PATH pairs, e.g. '
             '"NaturalBench-Qwen:results/08_pathology/qwen25_naturalbench.json"',
    )
    parser.add_argument(
        "--band_start", type=int, required=True,
        help="Left boundary of Normal Fusion Band (actual layer number). "
             "Copy from plot_results.py console output.",
    )
    parser.add_argument(
        "--band_end", type=int, required=True,
        help="Right boundary of Normal Fusion Band (actual layer number).",
    )
    parser.add_argument(
        "--model_key", type=str, default="qwen25",
        choices=list(MODEL_LAYERS.keys()),
        help=f"Model key for tick spacing. Choices: {list(MODEL_LAYERS.keys())}",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/figures/",
        help="Directory to save output figures",
    )
    parser.add_argument(
        "--annotate", action="store_true",
        help="Annotate 1D curve peaks with 'Prior Override' / 'Late Activation'",
    )

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_paper_style()

    # ------------------------------------------------------------------
    # Load and compute both curve and heatmap for every series
    # ------------------------------------------------------------------
    curve_series   = []   # (label, mean_curve, std_curve)
    heatmap_series = []   # (label, mean_heatmap_2d)

    for (label, path) in args.results:
        print(f"[{label}]")
        samples = load_pathology_filtered(path)
        if not samples:
            print(f"  WARNING: no failed samples in {path} — skipping.")
            continue

        mean_curve, std_curve = compute_flip_rate_curve(samples)
        mean_heatmap          = compute_mean_flip_heatmap(samples)

        curve_series.append((label, mean_curve, std_curve))
        heatmap_series.append((label, mean_heatmap))

    if not curve_series:
        print("ERROR: No valid series to plot.  Check that your JSON files "
              "contain samples with baseline_correct == false.")
        return

    # ------------------------------------------------------------------
    # Figure 1 (Main text) — 1D diagnostic curve
    # ------------------------------------------------------------------
    plot_pathology_diagnostic_curve(
        curve_series,
        band_start=args.band_start,
        band_end=args.band_end,
        output_dir=out_dir,
        model_key=args.model_key,
        annotate=args.annotate,
    )

    # ------------------------------------------------------------------
    # Figure 2 (Appendix) — 2D heatmap panels
    # ------------------------------------------------------------------
    plot_pathology_heatmaps(
        heatmap_series,
        band_start=args.band_start,
        band_end=args.band_end,
        output_dir=out_dir,
        model_key=args.model_key,
    )


if __name__ == "__main__":
    main()
