"""Visualization for Pathology Diagnosis Experiment (Section 5.2).

Plots the Flip Rate (fraction of patched samples that switch from incorrect to
correct) as a function of decoder layer.  Only samples where the model was
originally *wrong* (baseline_correct == false) are included.

The Normal Fusion Band from Section 5.1 is overlaid as a shaded region.
A peak that falls *left* of the band indicates Prior Override (language prior
suppresses the visual signal early).  A peak that falls *right* of the band
indicates Late Activation / Spatial Blindness (visual signal not yet fused).

Usage::

    python experiments/08_pathology_diagnosis/plot_pathology.py \\
        --results "NaturalBench-LLaVA:results/pathology_llava_naturalbench.json" \\
                  "MINDCUBE-LLaVA:results/pathology_llava_mindcube.json" \\
                  "NaturalBench-Qwen:results/pathology_qwen_naturalbench.json" \\
        --band_start 9 --band_end 18 \\
        --model_key qwen --annotate \\
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
    """Parse a 'LABEL:PATH' token into (label, Path).

    Used as argparse type= so errors are reported cleanly.
    """
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
    """Load pathology JSON and return only samples where the model failed.

    Samples missing the 'baseline_correct' key are treated as correct and
    excluded (conservative).
    """
    with open(path) as f:
        data = json.load(f)

    missing_key = sum(1 for s in data if "baseline_correct" not in s)
    if missing_key:
        print(f"  WARNING: {missing_key} sample(s) missing 'baseline_correct' key — excluded.")

    filtered = [s for s in data if not s.get("baseline_correct", True)]
    print(f"  Loaded {len(filtered)} failed samples (of {len(data)} total) from {path}")
    return filtered


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def compute_flip_rate_curve(samples):
    """Compute mean and std of the diagonal flip rate across samples.

    Parameters
    ----------
    samples : list of sample dicts, each with 'patching_sweep' (2-D list)

    Returns
    -------
    mean_curve : np.ndarray, shape (num_diag_layers,)
    std_curve  : np.ndarray, shape (num_diag_layers,)
        Both are in [0, 1].  Returns (empty, empty) arrays if no valid samples.
    """
    diagonals = []
    for sample in samples:
        sweep = sample.get("patching_sweep")
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
    padded = np.full((len(diagonals), max_len), np.nan)
    for i, d in enumerate(diagonals):
        padded[i, :len(d)] = d

    mean_curve = np.nanmean(padded, axis=0)
    std_curve  = np.nanstd(padded, axis=0)
    return mean_curve, std_curve


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_pathology(series, band_start, band_end, output_dir,
                   model_key, annotate):
    """Generate the pathology diagnosis figure.

    Parameters
    ----------
    series      : list of (label, mean_curve, std_curve)
    band_start  : int, actual layer number for Normal Fusion Band left edge
    band_end    : int, actual layer number for Normal Fusion Band right edge
    output_dir  : Path
    model_key   : str, key into MODEL_LAYERS (determines step and tick spacing)
    annotate    : bool, whether to add Prior Override / Late Activation arrows
    """
    step = MODEL_LAYERS.get(model_key, {"step": 1})["step"]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Fusion band shading (drawn first so it's behind the curves)
    ax.axvspan(band_start, band_end, color="#FFFDE7", alpha=0.5,
               label="Normal Fusion Band", zorder=0)

    palette = sns.color_palette("colorblind", n_colors=max(len(series), 1))

    num_layers_max = 0
    for (label, mean_curve, std_curve), color in zip(series, palette):
        if len(mean_curve) == 0:
            print(f"  WARNING: No data for series '{label}', skipping.")
            continue

        x = np.arange(len(mean_curve)) * step
        num_layers_max = max(num_layers_max, int(x[-1]) + step)

        ax.plot(x, mean_curve, color=color, linewidth=2, label=label, zorder=2)

        # Shaded ±1 std band, clipped to [0, 1]
        lower = np.clip(mean_curve - std_curve, 0.0, 1.0)
        upper = np.clip(mean_curve + std_curve, 0.0, 1.0)
        ax.fill_between(x, lower, upper, color=color, alpha=0.15, zorder=1)

        # Optional pathology-type annotations
        if annotate and len(mean_curve) > 0:
            peak_idx  = int(np.nanargmax(mean_curve))
            peak_layer = peak_idx * step
            peak_val   = float(mean_curve[peak_idx])

            if peak_layer < band_start:
                ax.annotate(
                    "Prior Override",
                    xy=(peak_layer, peak_val),
                    xytext=(peak_layer - step, peak_val + 0.09),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
                    fontsize=10, ha="center", color="black",
                )
            elif peak_layer > band_end:
                ax.annotate(
                    "Late Activation",
                    xy=(peak_layer, peak_val),
                    xytext=(peak_layer + step, peak_val + 0.09),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
                    fontsize=10, ha="center", color="black",
                )

    # Axes
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

    fig.savefig(output_dir / "pathology_diagnosis.pdf")
    fig.savefig(output_dir / "pathology_diagnosis.png")
    plt.close(fig)
    print(f"Saved pathology_diagnosis.pdf/.png → {output_dir}")


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
             '"NaturalBench-Qwen:results/pathology_qwen_nb.json"',
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
        "--model_key", type=str, default="qwen",
        choices=list(MODEL_LAYERS.keys()),
        help=f"Model key for layer spacing and tick generation. "
             f"Choices: {list(MODEL_LAYERS.keys())}",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/figures/",
        help="Directory to save output figures",
    )
    parser.add_argument(
        "--annotate", action="store_true",
        help="Annotate peaks with 'Prior Override' or 'Late Activation' arrows",
    )

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_paper_style()

    series = []
    for (label, path) in args.results:
        print(f"[{label}]")
        samples = load_pathology_filtered(path)
        if not samples:
            print(f"  WARNING: No failed samples found in {path} — skipping series.")
            continue
        mean_curve, std_curve = compute_flip_rate_curve(samples)
        series.append((label, mean_curve, std_curve))

    if not series:
        print("ERROR: No valid series to plot. Check that your JSON files contain "
              "samples with baseline_correct == false.")
        return

    plot_pathology(
        series,
        band_start=args.band_start,
        band_end=args.band_end,
        output_dir=out_dir,
        model_key=args.model_key,
        annotate=args.annotate,
    )


if __name__ == "__main__":
    main()
