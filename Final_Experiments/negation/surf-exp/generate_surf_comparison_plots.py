import os
import json
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Global matplotlib settings for readability
plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'figure.titlesize': 24
})

# Debug logging (enabled via --debug)
DEBUG = False


def _log(msg: str):
    if DEBUG:
        print(msg)

MODEL_NAME_MAP = {
    'qwen': 'qwen2-vl-7b-instruct',
    'deepseek': 'deepseek-vl2-small',
    'llava': 'llava-v1.5-7b'
}

ResultItem = Dict[str, object]
SAMPLED_LAYER_LABELS: Dict[str, List[int]] = {
    'qwen': list(range(0, 28, 3)),
    'deepseek': list(range(0, 10, 2)),
    'llava': list(range(0, 32, 3)),
}


def _fit_labels(base: List[int], n: int) -> List[int]:
    """Return the first n labels from base if available; otherwise fallback to 0..n-1."""
    if len(base) >= n:
        return base[:n]
    return list(range(n))


def get_layer_labels_for_model(model_key: str, n: int) -> List[int]:
    base = SAMPLED_LAYER_LABELS.get(model_key, list(range(n)))
    return _fit_labels(base, n)



def safe_json_load(path: str):
    """Load JSON robustly. Returns None if unable to parse."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to parse JSON at {path}: {e}")
        return None


def coerce_matrix(results_field) -> np.ndarray:
    """Coerce the 'results' field to a numeric matrix of 0/1 values.
    Returns empty array if invalid.
    """
    try:
        arr = np.array(results_field, dtype=float)
        # Ensure 2D
        if arr.ndim != 2:
            return np.array([])
        # Some pipelines may include non-binary; clip to [0,1]
        arr = np.clip(arr, 0, 1)
        return arr
    except Exception:
        return np.array([])


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
    if isinstance(pred, (int, float)):
        v = int(pred)
        if v in (0, 1):
            return v
        return 1 if v > 0 else 0
    if isinstance(pred, str):
        return _normalize_yn_text(pred)
    return None


def load_model_results(base_dir: str, model_key: str) -> List[ResultItem]:
    """Load prediction results for a model from surf-exp.
    Expects files:
      - {base_dir}/{model_key}/results/results_{model_key}.json
      - LLaVA variant might be named results_llava.json
    Returns list of dicts with keys: sample_id, filename, category, results (2D matrix)
    """
    results_dir = os.path.join(base_dir, model_key, 'results')
    candidate_files = [
        os.path.join(results_dir, f'results_{model_key}.json'),
    ]

    data = None
    for p in candidate_files:
        _log(f"Loading {model_key} results from {p}")
        if os.path.exists(p):
            data = safe_json_load(p)
            _log(f"Loaded {model_key} results from {p}")
            if data is not None:
                print(f"[OK] Loaded {model_key} results from {p}")
                break
    if data is None:
        print(f"[WARN] No valid results JSON found for {model_key} in {results_dir}")
        return []

    # Expect a list of samples
    if not isinstance(data, list):
        print(f"[WARN] Unexpected JSON type for {model_key}: {type(data)}. Skipping.")
        return []

    cleaned: List[ResultItem] = []
    dropped = 0
    for item in data:
        if not isinstance(item, dict):
            dropped += 1
            continue
        if 'results' not in item:
            dropped += 1
            continue
        mat = coerce_matrix(item['results'])
        if mat.size == 0:
            dropped += 1
            continue
        cleaned.append({
            'sample_id': item.get('sample_id'),
            'filename': item.get('filename'),
            'category': item.get('category'),
            'results': mat
        })
    if dropped:
        print(f"[INFO] Dropped {dropped} invalid entries for {model_key}")
    print(f"[OK] {model_key}: kept {len(cleaned)} samples")
    return cleaned


def load_model_baseline(base_dir: str, model_key: str) -> List[Dict[str, object]]:
    """Load baseline outputs for a model. Expected files:
      - {base_dir}/{model_key}/results/{model_key}_results_baseline.json
      - or llava_results_baseline.json for LLaVA
    Each entry should include sample_id and a prediction/response.
    """
    results_dir = os.path.join(base_dir, model_key, 'results')
    candidate_files = [
        os.path.join(results_dir, f'{model_key}_results_baseline.json'),
        os.path.join(results_dir, f'{model_key}_baseline.json'),
    ]
    if model_key == 'llava':
        candidate_files.insert(0, os.path.join(results_dir, 'llava_results_baseline.json'))
        candidate_files.append(os.path.join(base_dir, model_key, 'llava_results_baseline.json'))
    data = None
    for p in candidate_files:
        if os.path.exists(p):
            data = safe_json_load(p)
            if data is not None:
                print(f"[OK] Loaded {model_key} baseline from {p}")
                break
    if data is None or not isinstance(data, list):
        print(f"[WARN] No valid baseline JSON for {model_key} in {results_dir}")
        return []
    return data


essential_dims_cache: Dict[str, Tuple[int, int]] = {}


def determine_most_frequent_shape_for_items(items: List[ResultItem]) -> Tuple[int, int]:
    """Determine the most frequent (rows, cols) shape within a single model's items.
    Ties are broken by larger area.
    """
    from collections import Counter
    shape_counter = Counter()
    for it in items:
        mat = it['results']
        shape_counter[(mat.shape[0], mat.shape[1])] += 1
    if not shape_counter:
        return (0, 0)
    most_common = shape_counter.most_common()
    best_shape, best_count = None, -1
    best_area = -1
    for shape, cnt in most_common:
        area = shape[0] * shape[1]
        if cnt > best_count or (cnt == best_count and area > best_area):
            best_shape, best_count, best_area = shape, cnt, area
    return best_shape


def _build_baseline_map(baseline_list: List[dict]) -> Dict[int, bool]:
    """sid -> is_correct (GT=0)."""
    m: Dict[int, bool] = {}
    for row in baseline_list:
        sid = row.get('sample_id')
        if sid is None:
            continue
        pred = row.get('prediction')
        if pred is None:
            pred = _normalize_yn_text(row.get('response', ''))
        pred_val = coerce_prediction(pred)
        m[int(sid)] = (pred_val == 0) if pred_val is not None else False
    return m


def compute_baseline_intersection_stats(items: List[ResultItem], baseline_list: List[dict]) -> Tuple[int, int, int, int]:
    """Return (baseline_total, baseline_wrong_total, inter_size, inter_wrong)."""
    bm = _build_baseline_map(baseline_list)
    baseline_total = len(baseline_list)
    baseline_wrong_total = sum(1 for v in bm.values() if v is False)
    patched_ids = set(int(it.get('sample_id')) for it in items if it.get('sample_id') is not None)
    inter = patched_ids.intersection(bm.keys())
    inter_wrong = sum(1 for sid in inter if bm.get(sid) is False)
    return baseline_total, baseline_wrong_total, len(inter), inter_wrong


def aggregate_model_matrix(items: List[ResultItem], target_shape: Tuple[int, int]) -> pd.DataFrame:
    """Average across samples to get mean success rate per (source,target).
    Only samples matching target_shape are included.
    """
    if not items or target_shape == (0, 0):
        return pd.DataFrame()
    rows, cols = target_shape
    stack = []
    for it in items:
        mat = it['results']
        if mat.shape == target_shape:
            stack.append(mat)
    if not stack:
        return pd.DataFrame()
    arr = np.stack(stack, axis=0)  # [N, rows, cols]
    mean_mat = arr.mean(axis=0)
    df = pd.DataFrame(mean_mat,
                      index=[f"S{r}" for r in range(rows)],
                      columns=[f"T{c}" for c in range(cols)])
    return df


def compute_flip_rate_matrix(items: List[ResultItem], baseline_list: List[dict], target_shape: Tuple[int, int]) -> Tuple[pd.DataFrame, Dict[Tuple[int, int], Tuple[int, int]]]:
    """Compute flip-rate matrix for a model.
    Flip Rate at (ls, lt) = fraction of baseline-wrong samples that become correct after patching.
    Correctness is computed with ground-truth=0.
    Sample-level matching is enforced via sample_id intersection.
    """
    if not items or target_shape == (0, 0):
        return pd.DataFrame(), {}

    # Build baseline correctness map
    baseline_map: Dict[int, bool] = {}
    for row in baseline_list:
        sid = row.get('sample_id')
        if sid is None:
            continue
        pred = row.get('prediction')
        if pred is None:
            pred = _normalize_yn_text(row.get('response', ''))
        pred_val = coerce_prediction(pred)
        baseline_map[int(sid)] = (pred_val == 0) if pred_val is not None else False
    baseline_total = len(baseline_list)
    baseline_wrong_total = sum(1 for v in baseline_map.values() if v is False)
    _log(f"Baseline stats: total={baseline_total}, mapped={len(baseline_map)}, baseline_wrong={baseline_wrong_total}")

    # Keep only items with sample_id in baseline
    baseline_ids = set(baseline_map.keys())
    usable_items: List[ResultItem] = []
    for it in items:
        sid = it.get('sample_id')
        if sid is None:
            continue
        if int(sid) in baseline_ids:
            usable_items.append(it)
    _log(f"Patched items: total={len(items)}, usable (in baseline)={len(usable_items)}")

    rows, cols = target_shape
    num_flip: Dict[Tuple[int, int], int] = {}
    den_bad: Dict[Tuple[int, int], int] = {}
    for it in usable_items:
        mat = it['results']
        if not isinstance(mat, np.ndarray) or mat.shape != target_shape:
            continue
        sid = int(it.get('sample_id'))
        base_corr = baseline_map.get(sid, None)
        if base_corr is None or base_corr is True:
            continue
        for i in range(rows):
            for j in range(cols):
                pred = mat[i, j]
                try:
                    pred_int = int(pred)
                except Exception:
                    continue
                is_corr = (pred_int == 0)
                den_bad[(i, j)] = den_bad.get((i, j), 0) + 1
                if is_corr:
                    num_flip[(i, j)] = num_flip.get((i, j), 0) + 1

    # Build matrix
    mat = np.full((rows, cols), np.nan, dtype=float)
    col_den = [0 for _ in range(cols)]
    for i in range(rows):
        for j in range(cols):
            n = num_flip.get((i, j), 0)
            d = den_bad.get((i, j), 0)
            mat[i, j] = (n / d) if d > 0 else np.nan
            col_den[j] += d
    _log(f"Denominator sums per target layer: {col_den}")
    if all(d == 0 for d in col_den):
        _log("All denominators are zero across the grid (no baseline-wrong samples) -> flip-rate all NaN")

    df = pd.DataFrame(mat,
                      index=[f"S{i}" for i in range(rows)],
                      columns=[f"T{j}" for j in range(cols)])
    counts = { (i, j): (num_flip.get((i, j), 0), den_bad.get((i, j), 0))
               for i in range(rows) for j in range(cols) }
    return df, counts


def compute_failure_depth_variants(flip_df: pd.DataFrame, threshold: float = 0.5, prop: float = 0.5) -> Dict[str, Optional[int]]:
    if flip_df.empty:
        return { 'fd_any': None, 'fd_median': None, 'fd_prop': None }
    fd_any: Optional[int] = None
    fd_median: Optional[int] = None
    fd_prop: Optional[int] = None
    for idx, lt in enumerate(flip_df.columns):
        col = flip_df[lt].values
        # any
        if fd_any is None and np.nanmax(col) >= threshold:
            fd_any = idx
        # median
        med = np.nanmedian(col)
        if fd_median is None and (not np.isnan(med)) and med >= threshold:
            fd_median = idx
        # prop
        valid = ~np.isnan(col)
        if valid.sum() > 0:
            frac = (col[valid] >= threshold).mean()
            if fd_prop is None and frac >= prop:
                fd_prop = idx
        else:
            _log(f"Target layer {idx}: all NaN column -> no eligible samples at this target")
    return { 'fd_any': fd_any, 'fd_median': fd_median, 'fd_prop': fd_prop }


def compute_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-target aggregates: mean and median across sources."""
    if df.empty:
        return pd.DataFrame()
    agg = pd.DataFrame({
        'layer_target': list(range(df.shape[1])),
        'median_success_rate': df.median(axis=0).values,
        'mean_success_rate': df.mean(axis=0).values,
    })
    return agg


def create_comparison_heatmaps(model_to_df: Dict[str, pd.DataFrame], title_prefix: str, output_path: str):
    models = list(model_to_df.keys())
    if not models:
        print(f"[WARN] No models to plot for {title_prefix}")
        return

    # Compute global vmin/vmax from available values
    all_vals = []
    for m in models:
        df = model_to_df[m]
        if not df.empty:
            all_vals.extend(df.values.flatten().tolist())
    if not all_vals:
        print(f"[WARN] No data values to plot for {title_prefix}")
        return
    vmin, vmax = float(np.min(all_vals)), float(np.max(all_vals))

    fig, axes = plt.subplots(1, len(models), figsize=(6.5*len(models), 7))
    if len(models) == 1:
        axes = [axes]

    for i, m in enumerate(models):
        ax = axes[i]
        df = model_to_df[m]
        if df.empty:
            ax.text(0.5, 0.5, f"No data for {m}", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(MODEL_NAME_MAP.get(m, m))
            continue
        im = ax.imshow(df.values, aspect='auto', origin='lower', vmin=vmin, vmax=vmax, cmap='viridis')
        # Use sampled layer labels for both axes based on model
        x_ticks = np.arange(df.shape[1])
        y_ticks = np.arange(df.shape[0])
        x_labels = get_layer_labels_for_model(m, df.shape[1])
        y_labels = get_layer_labels_for_model(m, df.shape[0])
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_xlabel("Layer Target")
        ax.set_ylabel("Layer Source")
        ax.set_title(MODEL_NAME_MAP.get(m, m))
        if i == len(models) - 1:
            plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(f"{title_prefix} Comparison Across Models", fontsize=16, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved comparison plot: {output_path}")


def create_aggregates_comparison(model_to_agg: Dict[str, pd.DataFrame], output_path: str):
    models = list(model_to_agg.keys())
    if not models:
        print("[WARN] No models to plot for aggregates")
        return

    fig, axes = plt.subplots(1, len(models), figsize=(6.5*len(models), 7))
    if len(models) == 1:
        axes = [axes]

    for i, m in enumerate(models):
        ax = axes[i]
        df = model_to_agg[m]
        if df.empty:
            ax.text(0.5, 0.5, f"No data for {m}", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(MODEL_NAME_MAP.get(m, m))
            continue
        ax.plot(df['layer_target'], df['median_success_rate'], label='Median across sources', marker='o', linewidth=2)
        ax.plot(df['layer_target'], df['mean_success_rate'], label='Mean across sources', marker='s', linewidth=2)
        ax.set_xlabel("Layer Target")
        ax.set_ylabel("Success Rate")
        ax.set_title(MODEL_NAME_MAP.get(m, m))
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        # Replace x tick labels with sampled layer labels for this model
        num_targets = len(df)
        xticks = list(range(num_targets))
        ax.set_xticks(xticks)
        ax.set_xticklabels(get_layer_labels_for_model(m, num_targets), rotation=45, ha='right')

    fig.suptitle("Success Rate Aggregates Comparison Across Models", fontsize=16, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved aggregates comparison plot: {output_path}")


def save_failure_depth_table(model_to_fd: Dict[str, Dict[str, Optional[int]]], output_csv: str):
    rows = []
    for m, fds in model_to_fd.items():
        rows.append({
            'model': MODEL_NAME_MAP.get(m, m),
            'fd_any': fds.get('fd_any'),
            'fd_median': fds.get('fd_median'),
            'fd_prop': fds.get('fd_prop'),
        })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[OK] Saved failure-depth table: {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Generate comparison plots for surf-exp (flip-rate & failure-depth)")
    parser.add_argument('--base_dir', type=str, default='/home/sc305/VLM/Final Experiments/negation/v-patchscope-demo-code/surf-exp', help='Base surf-exp directory containing model subdirs')
    parser.add_argument('--models', nargs='+', default=['qwen', 'deepseek', 'llava'], help='Model keys to include')
    parser.add_argument('--output_dir', type=str, default='/home/sc305/VLM/Final Experiments/negation/v-patchscope-demo-code/surf-exp/comparisons', help='Output directory for plots')
    parser.add_argument('--threshold', type=float, default=0.5, help='Failure-depth threshold')
    parser.add_argument('--prop', type=float, default=0.5, help='Proportion for prop-based failure-depth')
    parser.add_argument('--debug', action='store_true', help='Enable verbose debug logging')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Enable debug logging if requested
    global DEBUG
    DEBUG = bool(args.debug)

    # Load results and baselines for requested models
    model_to_items: Dict[str, List[ResultItem]] = {}
    model_to_baseline: Dict[str, List[dict]] = {}
    for m in args.models:
        items = load_model_results(args.base_dir, m)
        # If LLaVA raw file is malformed, skip it cleanly
        if not items:
            print(f"[WARN] No valid items for {m}; it will still appear with 'No data' in plots")
        model_to_items[m] = items
        model_to_baseline[m] = load_model_baseline(args.base_dir, m)

    # Aggregate per model at each model's most frequent native shape
    model_to_df: Dict[str, pd.DataFrame] = {}
    model_to_agg: Dict[str, pd.DataFrame] = {}
    model_to_fd: Dict[str, Dict[str, Optional[int]]] = {}
    model_to_diag: Dict[str, Dict[str, int]] = {}
    for m, items in model_to_items.items():
        native_shape = determine_most_frequent_shape_for_items(items)
        if native_shape == (0, 0):
            print(f"[WARN] {m}: could not determine a native matrix shape; skipping aggregation")
            model_to_df[m] = pd.DataFrame()
            model_to_agg[m] = pd.DataFrame()
            model_to_fd[m] = { 'fd_any': None, 'fd_median': None, 'fd_prop': None }
            model_to_diag[m] = { 'baseline_total': 0, 'baseline_wrong_total': 0, 'intersection_total': 0, 'intersection_wrong': 0 }
            continue
        print(f"[OK] Using {m} native matrix shape: {native_shape}")

        # Compute flip-rate matrix against the model's own baseline
        flip_df, _counts = compute_flip_rate_matrix(items, model_to_baseline.get(m, []), native_shape)
        model_to_df[m] = flip_df

        # Diagnostics for CSV
        btot, bwrong, itot, iwrong = compute_baseline_intersection_stats(items, model_to_baseline.get(m, []))
        model_to_diag[m] = {
            'baseline_total': btot,
            'baseline_wrong_total': bwrong,
            'intersection_total': itot,
            'intersection_wrong': iwrong,
        }
        _log(f"Diag[{m}]: baseline_total={btot}, baseline_wrong_total={bwrong}, intersection_total={itot}, intersection_wrong={iwrong}")
        # Aggregates across sources per target layer
        if not flip_df.empty:
            agg = pd.DataFrame({
                'layer_target': list(range(flip_df.shape[1])),
                'median_success_rate': flip_df.median(axis=0).values,
                'mean_success_rate': flip_df.mean(axis=0).values,
            })
        else:
            agg = pd.DataFrame()
        model_to_agg[m] = agg

        # Failure-depth variants
        model_to_fd[m] = compute_failure_depth_variants(flip_df, threshold=args.threshold, prop=args.prop)
        print(f"[OK] Aggregated {m}: flip matrix {flip_df.shape if not flip_df.empty else 'empty'}; FD={model_to_fd[m]}")

    # Heatmap comparison of mean success rate
    heatmap_path = os.path.join(args.output_dir, 'flip_rate_comparison.png')
    create_comparison_heatmaps(model_to_df, 'Flip Rate', heatmap_path)

    # Aggregates comparison
    aggregates_path = os.path.join(args.output_dir, 'aggregates_flip_rate_comparison.png')
    create_aggregates_comparison(model_to_agg, aggregates_path)

    # Failure-depth table (with diagnostics)
    fd_csv = os.path.join(args.output_dir, 'failure_depth_summary.csv')
    rows = []
    for m in args.models:
        fds = model_to_fd.get(m, { 'fd_any': None, 'fd_median': None, 'fd_prop': None })
        diag = model_to_diag.get(m, { 'baseline_total': 0, 'baseline_wrong_total': 0, 'intersection_total': 0, 'intersection_wrong': 0 })
        rows.append({
            'model': MODEL_NAME_MAP.get(m, m),
            'fd_any': fds.get('fd_any'),
            'fd_median': fds.get('fd_median'),
            'fd_prop': fds.get('fd_prop'),
            'baseline_total': diag['baseline_total'],
            'baseline_wrong_total': diag['baseline_wrong_total'],
            'intersection_total': diag['intersection_total'],
            'intersection_wrong': diag['intersection_wrong'],
        })
    diag_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(fd_csv), exist_ok=True)
    diag_df.to_csv(fd_csv, index=False)
    print(f"[OK] Saved failure-depth table: {fd_csv}")

    print(f"[COMPLETE] All comparison plots and tables saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
