import json
import argparse
from collections import defaultdict
from typing import Dict, Tuple, List, Optional, Iterable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------------------------
# Types
# ---------------------------
DataKey = Tuple[int]  # (sample_id,)
BaselineCorrectMap = Dict[DataKey, bool]

# ---------------------------
# IO
# ---------------------------
def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

# ---------------------------
# Utilities
# ---------------------------

def _normalize_yn_text(s: str) -> Optional[int]:
    if not isinstance(s, str):
        return None
    t = s.strip().lower()
    t = t.replace(".", "").replace("!", "").replace("?", "").replace("\"", "'").strip()
    if t.startswith("yes"):
        return 1
    if t.startswith("no"):
        return 0
    return None


def coerce_prediction(pred) -> Optional[int]:
    # prediction might be int 0/1, string yes/no, or embedded under different keys
    if isinstance(pred, (int, float)):
        v = int(pred)
        if v in (0, 1):
            return v
        return 1 if v > 0 else 0
    if isinstance(pred, str):
        return _normalize_yn_text(pred)
    return None

# ---------------------------
# Schema helpers for surf-exp
# ---------------------------
# Patched (grid) JSON: list of dicts with keys: sample_id, filename, category, results -> 2D list with cells possibly objects/dicts or numbers
# Baseline JSON: list of dicts with keys: sample_id, filename, category, prediction (0/1 or text), response (string)


def extract_layer_axes_from_grid(items: List[dict]) -> Tuple[List[int], List[int]]:
    ls_set, lt_set = set(), set()
    for item in items:
        results = item.get("results")
        if results is None:
            continue
        try:
            arr = np.array(results, dtype=object)
        except Exception:
            continue
        if arr.ndim != 2:
            continue
        rows, cols = arr.shape
        for i in range(rows):
            ls_set.add(i)
        for j in range(cols):
            lt_set.add(j)
    return sorted(ls_set), sorted(lt_set)


def iter_grid_cells(items: List[dict]) -> Iterable[Tuple[DataKey, int, int, Optional[int]]]:
    for item in items:
        sid = item.get("sample_id")
        if sid is None:
            continue
        key: DataKey = (int(sid),)
        results = item.get("results")
        if results is None:
            continue
        try:
            arr = np.array(results, dtype=object)
        except Exception:
            continue
        if arr.ndim != 2:
            continue
        rows, cols = arr.shape
        for i in range(rows):
            for j in range(cols):
                cell = arr[i, j]
                pred_val: Optional[int] = None
                if isinstance(cell, dict):
                    # possible keys: prediction, result
                    pred_val = coerce_prediction(cell.get("prediction", cell.get("result")))
                else:
                    pred_val = coerce_prediction(cell)
                yield key, i, j, pred_val


def load_baseline_correct_map_from_baseline_list(baseline_list: List[dict]) -> BaselineCorrectMap:
    # Ground truth is 0 (no). Correct iff prediction == 0.
    m: BaselineCorrectMap = {}
    for row in baseline_list:
        sid = row.get("sample_id")
        if sid is None:
            continue
        pred = row.get("prediction")
        if pred is None:
            pred = _normalize_yn_text(row.get("response", ""))
        pred_val = coerce_prediction(pred)
        if pred_val is None:
            # cannot decide; mark as incorrect to not bias flip-rate denominators unexpectedly
            m[(int(sid),)] = False
        else:
            m[(int(sid),)] = (pred_val == 0)
    return m

# ---------------------------
# Correct-Rate matrix
# ---------------------------

def compute_correct_rate_matrix(items: List[dict]) -> Tuple[pd.DataFrame, Dict[Tuple[int, int], List[bool]]]:
    ls_axis, lt_axis = extract_layer_axes_from_grid(items)
    bucket: Dict[Tuple[int, int], List[bool]] = defaultdict(list)
    for _key, ls, lt, pred in iter_grid_cells(items):
        if pred is None:
            continue
        corr = (pred == 0)  # ground-truth is 0
        bucket[(ls, lt)].append(bool(corr))

    mat = np.full((len(ls_axis), len(lt_axis)), np.nan, dtype=float)
    for i, ls in enumerate(ls_axis):
        for j, lt in enumerate(lt_axis):
            vals = bucket.get((ls, lt), [])
            mat[i, j] = float(np.mean(vals)) if vals else np.nan

    df = pd.DataFrame(mat, index=ls_axis, columns=lt_axis)
    df.index.name = "layer_source"
    df.columns.name = "layer_target"
    return df, bucket

# ---------------------------
# Flip-Rate matrix
# ---------------------------

def compute_flip_rate_matrix(
    items: List[dict],
    baseline_correct: BaselineCorrectMap,
) -> Tuple[pd.DataFrame, Dict[Tuple[int, int], Tuple[int, int]]]:
    ls_axis, lt_axis = extract_layer_axes_from_grid(items)
    num_flip = defaultdict(int)
    den_bad = defaultdict(int)

    for key, ls, lt, pred in iter_grid_cells(items):
        base = baseline_correct.get(key, None)
        if base is None:
            continue
        if base is False:                # eligible: baseline was wrong
            den_bad[(ls, lt)] += 1
            if pred is not None and (pred == 0):  # patched became correct
                num_flip[(ls, lt)] += 1

    mat = np.full((len(ls_axis), len(lt_axis)), np.nan, dtype=float)
    for i, ls in enumerate(ls_axis):
        for j, lt in enumerate(lt_axis):
            n = num_flip.get((ls, lt), 0)
            d = den_bad.get((ls, lt), 0)
            mat[i, j] = (n / d) if d > 0 else np.nan

    df = pd.DataFrame(mat, index=ls_axis, columns=lt_axis)
    df.index.name = "layer_source"
    df.columns.name = "layer_target"

    counts = { (ls, lt): (num_flip.get((ls, lt), 0), den_bad.get((ls, lt), 0))
               for ls in ls_axis for lt in lt_axis }
    return df, counts

# ---------------------------
# Failure-Depth variants
# ---------------------------

def compute_failure_depth_any(flip_rate_df: pd.DataFrame, threshold: float = 0.5) -> Optional[int]:
    """Earliest target layer where ANY source reaches threshold."""
    for lt in flip_rate_df.columns:
        col = flip_rate_df[lt].values
        if np.nanmax(col) >= threshold:
            return int(lt)
    return None


def compute_failure_depth_median(flip_rate_df: pd.DataFrame, threshold: float = 0.5) -> Optional[int]:
    """Earliest target layer where MEDIAN across sources reaches threshold."""
    for lt in flip_rate_df.columns:
        col = flip_rate_df[lt].values
        med = np.nanmedian(col)
        if not np.isnan(med) and med >= threshold:
            return int(lt)
    return None


def compute_failure_depth_prop(flip_rate_df: pd.DataFrame, threshold: float = 0.5, prop: float = 0.5) -> Optional[int]:
    """Earliest target layer where >= prop fraction of sources reach threshold."""
    for lt in flip_rate_df.columns:
        col = flip_rate_df[lt].values
        valid = ~np.isnan(col)
        if valid.sum() == 0:
            continue
        frac = (col[valid] >= threshold).mean()
        if frac >= prop:
            return int(lt)
    return None

# ---------------------------
# Plot helpers
# ---------------------------

def plot_heatmap(df: pd.DataFrame, title: str, outpath_png: str):
    plt.figure()
    data = df.values.copy()
    plt.imshow(data, aspect='auto', origin='lower')
    plt.colorbar()
    plt.xticks(ticks=np.arange(df.shape[1]), labels=df.columns, rotation=45, ha='right')
    plt.yticks(ticks=np.arange(df.shape[0]), labels=df.index)
    plt.xlabel("layer_target")
    plt.ylabel("layer_source")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath_png, dpi=300)
    plt.close()


def plot_aggregates(target_layers: List[int], med: np.ndarray, mean: np.ndarray, prop_vals: np.ndarray, out_png: str):
    plt.figure()
    plt.plot(target_layers, med, label="median across sources")
    plt.plot(target_layers, mean, label="mean across sources")
    plt.plot(target_layers, prop_vals, label="prop≥threshold across sources")
    plt.xlabel("layer_target")
    plt.ylabel("value")
    plt.title("Flip-Rate aggregates vs target layer")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

# ---------------------------
# Matching helpers
# ---------------------------

def filter_items_by_sample_ids(items: List[dict], valid_ids: set) -> List[dict]:
    out = []
    for it in items:
        sid = it.get("sample_id")
        if sid is None:
            continue
        if int(sid) in valid_ids:
            out.append(it)
    return out

# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patched", required=True, help="Path to patched results JSON (grid over (ls, lt)).")
    ap.add_argument("--baseline", required=True, help="Path to baseline results JSON list.")
    ap.add_argument("--outdir", required=True, help="Output directory for CSV/PNGs.")
    ap.add_argument("--threshold", type=float, default=0.5, help="Threshold for Failure-Depth.")
    ap.add_argument("--prop", type=float, default=0.5, help="Proportion for prop-based Failure-Depth.")
    ap.add_argument("--prefix", type=str, default="", help="Optional filename prefix, e.g., model name.")
    args = ap.parse_args()

    ensure_outdir(args.outdir)

    # Load data
    patched = load_json(args.patched)
    baseline_list = load_json(args.baseline)

    # Build baseline correctness map (GT=0)
    baseline_map = load_baseline_correct_map_from_baseline_list(baseline_list)

    # Match by sample_id intersection to avoid mismatches
    baseline_ids = set(int(k[0]) for k in baseline_map.keys())
    patched_ids = set(int(it.get("sample_id")) for it in patched if isinstance(it, dict) and it.get("sample_id") is not None)
    common_ids = baseline_ids.intersection(patched_ids)
    if not common_ids:
        print("[ERROR] No overlapping sample_id between baseline and patched.")
        return
    if len(common_ids) < len(patched_ids) or len(common_ids) < len(baseline_ids):
        print(f"[INFO] Filtering to {len(common_ids)} common sample_ids (baseline={len(baseline_ids)}, patched={len(patched_ids)})")

    # Filter patched to only common sample_ids
    patched = filter_items_by_sample_ids(patched, common_ids)

    # Correct-Rate
    correct_df, _ = compute_correct_rate_matrix(patched)
    correct_csv = os.path.join(args.outdir, f"{args.prefix}correct_rate_matrix.csv")
    correct_png = os.path.join(args.outdir, f"{args.prefix}correct_rate_heatmap.png")
    correct_df.to_csv(correct_csv)
    plot_heatmap(correct_df, "Correct-Rate (mean correctness, GT=0)", correct_png)

    # Flip-Rate
    flip_df, flip_counts = compute_flip_rate_matrix(patched, baseline_map)
    flip_csv = os.path.join(args.outdir, f"{args.prefix}flip_rate_matrix.csv")
    flip_png = os.path.join(args.outdir, f"{args.prefix}flip_rate_heatmap.png")
    flip_df.to_csv(flip_csv)
    plot_heatmap(flip_df, "Flip-Rate (baseline wrong → patched correct, GT=0)", flip_png)

    # Failure-Depth (three variants)
    fd_any = compute_failure_depth_any(flip_df, threshold=args.threshold)
    fd_med = compute_failure_depth_median(flip_df, threshold=args.threshold)
    fd_prop = compute_failure_depth_prop(flip_df, threshold=args.threshold, prop=args.prop)

    # Aggregates across source layers vs target layer (for plotting & CSV)
    tlayers = list(flip_df.columns)
    med = np.array([np.nanmedian(flip_df[lt].values) for lt in tlayers], dtype=float)
    mean = np.array([np.nanmean(flip_df[lt].values) for lt in tlayers], dtype=float)
    prop_vals = np.array([
        ( (flip_df[lt].values[~np.isnan(flip_df[lt].values)] >= args.threshold).mean()
          if (~np.isnan(flip_df[lt].values)).sum() > 0 else np.nan )
        for lt in tlayers
    ], dtype=float)

    agg_df = pd.DataFrame({
        "layer_target": tlayers,
        "median_flip_rate": med,
        "mean_flip_rate": mean,
        f"prop_ge_{args.threshold}": prop_vals
    })
    agg_csv = os.path.join(args.outdir, f"{args.prefix}flip_rate_aggregates.csv")
    agg_png = os.path.join(args.outdir, f"{args.prefix}flip_rate_aggregates.png")
    agg_df.to_csv(agg_csv, index=False)
    plot_aggregates(tlayers, med, mean, prop_vals, agg_png)

    # Minimal summary
    print(f"[OK] Saved: {correct_csv}, {correct_png}")
    print(f"[OK] Saved: {flip_csv}, {flip_png}")
    print(f"[OK] Saved: {agg_csv}, {agg_png}")
    print(f"[Info] Failure-Depth (ANY-source ≥ {args.threshold}): {fd_any}")
    print(f"[Info] Failure-Depth (MEDIAN across sources ≥ {args.threshold}): {fd_med}")
    print(f"[Info] Failure-Depth (PROP ≥ {args.threshold} with prop={args.prop}): {fd_prop}")

if __name__ == "__main__":
    main()
