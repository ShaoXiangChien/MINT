import os
import json
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 20
})

DEBUG = False

def _log(msg: str):
    if DEBUG:
        print(msg)

SAMPLED_LAYER_LABELS = list(range(0, 28, 3))


def safe_json_load(path: str):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to parse JSON at {path}: {e}")
        return None


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


def coerce_matrix(results_field) -> np.ndarray:
    try:
        arr = np.array(results_field, dtype=float)
        if arr.ndim != 2:
            return np.array([])
        return np.clip(arr, 0, 1)
    except Exception:
        return np.array([])


def load_qwen_results(base_dir: str) -> List[Dict[str, object]]:
    results_path = os.path.join(base_dir, 'qwen', 'results', 'results_qwen.json')
    data = safe_json_load(results_path)
    if not isinstance(data, list):
        print(f"[ERROR] Could not load Qwen results at {results_path}")
        return []
    cleaned = []
    for item in data:
        mat = coerce_matrix(item.get('results'))
        if mat.size == 0:
            continue
        cleaned.append({
            'sample_id': item.get('sample_id'),
            'results': mat
        })
    print(f"[OK] Qwen: kept {len(cleaned)} samples from {results_path}")
    return cleaned


def load_qwen_baseline(base_dir: str) -> List[dict]:
    path = os.path.join(base_dir, 'qwen', 'results', 'qwen_results_baseline.json')
    data = safe_json_load(path)
    if not isinstance(data, list):
        print(f"[ERROR] Could not load Qwen baseline at {path}")
        return []
    print(f"[OK] Loaded Qwen baseline from {path}")
    return data


def determine_shape(items: List[Dict[str, object]]) -> Tuple[int, int]:
    from collections import Counter
    c = Counter()
    for it in items:
        m = it['results']
        c[(m.shape[0], m.shape[1])] += 1
    if not c:
        return (0, 0)
    return c.most_common(1)[0][0]


def compute_flip_rate(items: List[Dict[str, object]], baseline_list: List[dict], shape: Tuple[int, int]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    if not items or shape == (0, 0):
        return pd.DataFrame(), {}
    bm: Dict[int, bool] = {}
    for r in baseline_list:
        sid = r.get('sample_id')
        if sid is None:
            continue
        pv = r.get('prediction')
        if pv is None:
            pv = _normalize_yn_text(r.get('response', ''))
        pv = coerce_prediction(pv)
        bm[int(sid)] = (pv == 0) if pv is not None else False
    btot = len(baseline_list)
    bwrong = sum(1 for v in bm.values() if v is False)

    rows, cols = shape
    num = np.zeros((rows, cols), dtype=int)
    den = np.zeros((rows, cols), dtype=int)

    for it in items:
        sid = it.get('sample_id')
        if sid is None or int(sid) not in bm:
            continue
        if bm[int(sid)] is True:
            continue
        mat = it['results']
        if mat.shape != shape:
            continue
        for i in range(rows):
            for j in range(cols):
                den[i, j] += 1
                try:
                    pred = int(mat[i, j])
                except Exception:
                    pred = 1
                if pred == 0:
                    num[i, j] += 1

    with np.errstate(invalid='ignore', divide='ignore'):
        flip = num / den
        flip[den == 0] = np.nan

    df = pd.DataFrame(flip, index=[f"S{i}" for i in range(rows)], columns=[f"T{j}" for j in range(cols)])
    diag = {
        'baseline_total': btot,
        'baseline_wrong_total': bwrong,
        'intersection_total': len(items),
        'intersection_wrong': sum(1 for it in items if bm.get(int(it.get('sample_id', -1)), True) is False),
    }
    return df, diag


def compute_failure_depths(flip_df: pd.DataFrame, threshold: float, prop: float) -> Dict[str, Optional[int]]:
    if flip_df.empty:
        return { 'fd_any': None, 'fd_median': None, 'fd_prop': None }
    fd_any = None
    fd_median = None
    fd_prop = None
    for idx, lt in enumerate(flip_df.columns):
        col = flip_df[lt].values
        if np.isnan(col).all():
            continue
        if fd_any is None and np.nanmax(col) >= threshold:
            fd_any = idx
        med = np.nanmedian(col)
        if fd_median is None and (not np.isnan(med)) and med >= threshold:
            fd_median = idx
        valid = ~np.isnan(col)
        if valid.any():
            frac = (col[valid] >= threshold).mean()
            if fd_prop is None and frac >= prop:
                fd_prop = idx
    return { 'fd_any': fd_any, 'fd_median': fd_median, 'fd_prop': fd_prop }


def plot_heatmap(df: pd.DataFrame, title: str, out_png: str, save_pdf: bool = False):
    plt.figure()
    im = plt.imshow(df.values, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(im)
    nrows, ncols = df.shape
    x_ticks = np.arange(ncols)
    y_ticks = np.arange(nrows)
    # Layer labels cropped to ncols/nrows
    x_labels = SAMPLED_LAYER_LABELS[:ncols] if len(SAMPLED_LAYER_LABELS) >= ncols else list(range(ncols))
    y_labels = SAMPLED_LAYER_LABELS[:nrows] if len(SAMPLED_LAYER_LABELS) >= nrows else list(range(nrows))
    plt.xticks(x_ticks, x_labels, rotation=45, ha='right')
    plt.yticks(y_ticks, y_labels)
    plt.xlabel("Layer Target")
    plt.ylabel("Layer Source")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    if save_pdf:
        out_pdf = os.path.splitext(out_png)[0] + '.pdf'
        plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()


def plot_aggregates(df: pd.DataFrame, out_png: str, save_pdf: bool = False):
    if df.empty:
        return
    x = list(range(df.shape[1]))
    med = [np.nanmedian(df.iloc[:, j].values) for j in x]
    mean = [np.nanmean(df.iloc[:, j].values) for j in x]
    plt.figure()
    plt.plot(x, med, label='Median across sources', marker='o', linewidth=2)
    plt.plot(x, mean, label='Mean across sources', marker='s', linewidth=2)
    x_labels = SAMPLED_LAYER_LABELS[:len(x)] if len(SAMPLED_LAYER_LABELS) >= len(x) else x
    plt.xticks(x, x_labels, rotation=45, ha='right')
    plt.xlabel("Layer Target")
    plt.ylabel("Flip Rate")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    if save_pdf:
        out_pdf = os.path.splitext(out_png)[0] + '.pdf'
        plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Qwen-only flip-rate & failure-depth plots')
    parser.add_argument('--base_dir', type=str, default='/home/sc305/VLM/Final Experiments/negation/v-patchscope-demo-code/surf-exp')
    parser.add_argument('--output_dir', type=str, default='/home/sc305/VLM/Final Experiments/negation/v-patchscope-demo-code/surf-exp/comparisons/qwen_only')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--prop', type=float, default=0.5)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--save_pdf', action='store_true', help='Also save figures as PDF alongside PNG')
    args = parser.parse_args()

    global DEBUG
    DEBUG = bool(args.debug)

    items = load_qwen_results(args.base_dir)
    baseline = load_qwen_baseline(args.base_dir)
    shape = determine_shape(items)
    if shape == (0, 0):
        print('[ERROR] Could not determine matrix shape for Qwen results')
        return
    flip_df, diag = compute_flip_rate(items, baseline, shape)
    fds = compute_failure_depths(flip_df, threshold=args.threshold, prop=args.prop)

    # Save heatmap and aggregates
    heatmap_png = os.path.join(args.output_dir, 'qwen_flip_rate_heatmap.png')
    plot_aggregates_png = os.path.join(args.output_dir, 'qwen_flip_rate_aggregates.png')
    plot_heatmap(flip_df, 'Flip Rate (Qwen)', heatmap_png, save_pdf=args.save_pdf)
    plot_aggregates(flip_df, plot_aggregates_png, save_pdf=args.save_pdf)

    # Save CSV summary
    summary_csv = os.path.join(args.output_dir, 'qwen_failure_depth_summary.csv')
    row = {
        'fd_any': fds['fd_any'],
        'fd_median': fds['fd_median'],
        'fd_prop': fds['fd_prop'],
        'baseline_total': diag.get('baseline_total', 0),
        'baseline_wrong_total': diag.get('baseline_wrong_total', 0),
        'intersection_total': diag.get('intersection_total', 0),
        'intersection_wrong': diag.get('intersection_wrong', 0),
    }
    pd.DataFrame([row]).to_csv(summary_csv, index=False)
    if args.save_pdf:
        heatmap_pdf = os.path.splitext(heatmap_png)[0] + '.pdf'
        aggregates_pdf = os.path.splitext(plot_aggregates_png)[0] + '.pdf'
        print(f"[OK] Saved: {heatmap_png}, {heatmap_pdf}, {plot_aggregates_png}, {aggregates_pdf}, {summary_csv}")
    else:
        print(f"[OK] Saved: {heatmap_png}, {plot_aggregates_png}, {summary_csv}")


if __name__ == '__main__':
    main()


