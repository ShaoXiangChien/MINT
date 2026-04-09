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

Each ``--results`` entry is ``LABEL:MODEL_KEY:PATH`` so that every series
carries its own layer-count information (different models have different
numbers of decoder layers).

Usage::

    python experiments/08_pathology_diagnosis/plot_pathology.py \\
        --results \\
            "NaturalBench-Qwen:qwen25:results/08_pathology/qwen25_naturalbench.json" \\
            "MINDCUBE-Qwen:qwen25:results/08_pathology/qwen25_mindcube.json" \\
            "NaturalBench-LLaVA:llava_onevision:results/08_pathology/llava_onevision_naturalbench.json" \\
            "MINDCUBE-LLaVA:llava_onevision:results/08_pathology/llava_onevision_mindcube.json" \\
            "NaturalBench-InternVL:internvl25:results/08_pathology/internvl25_naturalbench.json" \\
            "MINDCUBE-InternVL:internvl25:results/08_pathology/internvl25_mindcube.json" \\
        --band_start 9 --band_end 18 \\
        --annotate \\
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

def parse_entry(token: str):
    """Parse a ``LABEL:MODEL_KEY:PATH`` token.

    Returns (label: str, model_key: str, path: Path).
    Raises ArgumentTypeError if the format is wrong or model_key is unknown.
    """
    parts = token.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Expected LABEL:MODEL_KEY:PATH, got: {token!r}\n"
            f"  Valid model keys: {list(MODEL_LAYERS.keys())}"
        )
    label, model_key, path_str = parts
    if model_key not in MODEL_LAYERS:
        raise argparse.ArgumentTypeError(
            f"Unknown model key {model_key!r} in entry {token!r}.\n"
            f"  Valid keys: {list(MODEL_LAYERS.keys())}"
        )
    return label, model_key, Path(path_str)


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
    print(f"  {len(filtered)} failed / {len(data)} total  ←  {path.name}")
    return filtered


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def compute_flip_rate_curve(samples):
    """Flip rate per target layer, averaged over source layers and samples.

    For each sample's ``patching_sweep`` matrix we first take the column mean
    (``axis=0``) to get one flip-rate value per target layer, representing
    "average effectiveness of patching INTO target layer t from any source".
    We then average (and compute std) across samples.

    Returns
    -------
    mean_curve, std_curve : np.ndarray, shape (n_target_layers,)
        Values in [0, 1].  Returns two empty arrays if no valid samples.
    """
    col_means = []
    for s in samples:
        sweep = s.get("patching_sweep")
        if sweep is None:
            continue
        mat = np.array(sweep, dtype=float)
        if mat.ndim != 2 or mat.size == 0:
            continue
        # axis=0 → average over source layers, one value per target layer
        col_means.append(np.nanmean(mat, axis=0))

    if not col_means:
        return np.array([]), np.array([])

    max_len = max(len(c) for c in col_means)
    padded  = np.full((len(col_means), max_len), np.nan)
    for i, c in enumerate(col_means):
        padded[i, :len(c)] = c

    return np.nanmean(padded, axis=0), np.nanstd(padded, axis=0)


def compute_mean_flip_heatmap(samples):
    """Average the 2D ``patching_sweep`` matrices across failed samples.

    Returns np.ndarray shape (n_src, n_tgt), or None if no valid data.
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
                                    output_dir, annotate):
    """1D Flip Rate curves with Fusion Band overlay.

    Parameters
    ----------
    series   : list of (label, model_key, mean_curve, std_curve)
    band_start, band_end : int, actual layer numbers
    output_dir : Path
    annotate : bool
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Fusion band (drawn first, behind everything)
    ax.axvspan(band_start, band_end, color="#FFFDE7", alpha=0.5,
               label="Normal Fusion Band", zorder=0)

    palette = sns.color_palette("colorblind", n_colors=max(len(series), 1))
    x_max = 0

    for (label, model_key, mean_curve, std_curve), color in zip(series, palette):
        if len(mean_curve) == 0:
            print(f"  WARNING: empty curve for '{label}', skipping.")
            continue

        step = MODEL_LAYERS[model_key]["step"]
        x    = np.arange(len(mean_curve)) * step
        x_max = max(x_max, int(x[-1]))

        ax.plot(x, mean_curve, color=color, linewidth=2, label=label, zorder=2)
        ax.fill_between(
            x,
            np.clip(mean_curve - std_curve, 0.0, 1.0),
            np.clip(mean_curve + std_curve, 0.0, 1.0),
            color=color, alpha=0.15, zorder=1,
        )

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

    # X-axis: span the largest model's range; ticks every 2 layers
    if x_max == 0:
        x_max = 28
    ax.set_xlim(0, x_max)
    tick_step = 2
    ticks = list(range(0, x_max + tick_step, tick_step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks], fontsize=9)

    ax.set_ylim(0, 1)
    ax.set_xlabel("Target Layer (injection point)", fontsize=14)
    ax.set_ylabel("Flip Rate (avg. over source layers)", fontsize=14)
    ax.set_title("Pathology Diagnosis: Where Does Patching Cure Failures?",
                 fontsize=16, fontweight="bold")
    ax.legend(fontsize=11, loc="upper right", framealpha=0.85)
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
                             output_dir):
    """2D Flip Rate heatmaps — one panel per series.

    Parameters
    ----------
    heatmap_series : list of (label, model_key, mean_heatmap_2d)
    band_start, band_end : int, actual layer numbers
    output_dir : Path
    """
    n = len(heatmap_series)
    if n == 0:
        print("WARNING: No heatmap data to plot.")
        return

    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.5 * ncols, 4.5 * nrows),
        squeeze=False,
    )

    im_ref = None

    for idx, (label, model_key, heatmap) in enumerate(heatmap_series):
        row, col = divmod(idx, ncols)
        ax   = axes[row][col]
        step = MODEL_LAYERS[model_key]["step"]

        if heatmap is None:
            ax.set_visible(False)
            continue

        n_src, n_tgt = heatmap.shape

        im = ax.imshow(heatmap, vmin=0, vmax=1, cmap="viridis",
                       aspect="auto", origin="lower")
        im_ref = im

        # Horizontal dashed midline — visual guide showing that the 1D curve
        # collapses ALL source rows into a column mean
        ax.axhline(n_src / 2, color="white", linestyle="--",
                   linewidth=1.0, alpha=0.6)

        # Fusion Band: vertical dotted lines on target-layer (x) axis
        ax.axvline(band_start / step, color="#FFD700", linestyle=":",
                   linewidth=1.8, alpha=0.9, label="Band start")
        ax.axvline(band_end   / step, color="#FF8C00", linestyle=":",
                   linewidth=1.8, alpha=0.9, label="Band end")

        # Tick labels in actual layer numbers
        stride    = max(1, n_tgt // 8)
        tgt_ticks = list(range(0, n_tgt, stride))
        src_ticks = list(range(0, n_src, stride))
        ax.set_xticks(tgt_ticks)
        ax.set_xticklabels([str(t * step) for t in tgt_ticks],
                           fontsize=8, rotation=45)
        ax.set_yticks(src_ticks)
        ax.set_yticklabels([str(t * step) for t in src_ticks], fontsize=8)

        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.set_xlabel("Target Layer", fontsize=11)
        ax.set_ylabel("Source Layer", fontsize=11)

        if idx == 0:
            ax.legend(fontsize=8, loc="upper left", framealpha=0.7)

    # Hide unused panels
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    # Shared colorbar
    if im_ref is not None:
        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.70])
        cb = fig.colorbar(im_ref, cax=cbar_ax)
        cb.set_label("Flip Rate (Incorrect → Correct)", fontsize=11)
        cb.ax.tick_params(labelsize=9)

    fig.suptitle(
        "Appendix — Pathology Flip Rate Heatmaps (Source Layer × Target Layer)\n"
        "Dotted lines = Normal Fusion Band  |  1D curve = column mean (avg. over source layers)",
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
        type=parse_entry,
        metavar="LABEL:MODEL_KEY:PATH",
        help=(
            "One entry per series in LABEL:MODEL_KEY:PATH format.\n"
            f"Valid model keys: {list(MODEL_LAYERS.keys())}\n"
            'e.g. "NaturalBench-Qwen:qwen25:results/08_pathology/qwen25_naturalbench.json"'
        ),
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

    curve_series   = []   # (label, model_key, mean_curve, std_curve)
    heatmap_series = []   # (label, model_key, mean_heatmap_2d)

    for (label, model_key, path) in args.results:
        print(f"[{label}]  model={model_key}")
        samples = load_pathology_filtered(path)
        if not samples:
            print(f"  WARNING: no failed samples in {path} — skipping.")
            continue

        mean_curve, std_curve = compute_flip_rate_curve(samples)
        mean_heatmap          = compute_mean_flip_heatmap(samples)

        curve_series.append((label, model_key, mean_curve, std_curve))
        heatmap_series.append((label, model_key, mean_heatmap))

    if not curve_series:
        print("ERROR: No valid series to plot.")
        return

    plot_pathology_diagnostic_curve(
        curve_series,
        band_start=args.band_start,
        band_end=args.band_end,
        output_dir=out_dir,
        annotate=args.annotate,
    )

    plot_pathology_heatmaps(
        heatmap_series,
        band_start=args.band_start,
        band_end=args.band_end,
        output_dir=out_dir,
    )


if __name__ == "__main__":
    main()
