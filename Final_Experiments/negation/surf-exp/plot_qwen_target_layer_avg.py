#!/usr/bin/env python3
"""
Qwen-only: plot average flip rate per target layer (single line plot).

This script loads Qwen surf-exp results and baseline, computes the flip-rate
matrix as in generate_qwen_flip_plots.py, and produces a single line plot of
the mean (across source layers) flip rate vs target layer, with a shaded band
of ± std/2. Outputs are saved to the given output directory.
"""

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
    'figure.titlesize': 20,
    'figure.dpi': 100,
})


DEBUG = False


def _log(msg: str):
    if DEBUG:
        print(msg)


# Layer labels sampled every 3 up to 27, as used in the existing script
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
    t = t.replace('.', '').replace('!', '').replace('?', '').replace('"', "'").strip()
    if t.startswith('yes'):
        return 1
    if t.startswith('no'):
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
            'results': mat,
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


def compute_flip_rate(items: List[Dict[str, object]], baseline_list: List[dict], shape: Tuple[int, int]) -> pd.DataFrame:
    if not items or shape == (0, 0):
        return pd.DataFrame()
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

    rows, cols = shape
    num = np.zeros((rows, cols), dtype=int)
    den = np.zeros((rows, cols), dtype=int)

    for it in items:
        sid = it.get('sample_id')
        if sid is None or int(sid) not in bm:
            continue
        if bm[int(sid)] is True:  # baseline already correct; not eligible
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
    return df


def plot_target_avg_line(df: pd.DataFrame, out_png: str, save_pdf: bool = False):
    if df.empty:
        print('[ERROR] Empty flip-rate matrix; nothing to plot')
        return
    x = list(range(df.shape[1]))
    # Mean and std across sources for each target layer
    mean_vals = np.array([np.nanmean(df.iloc[:, j].values) for j in x], dtype=float)
    std_vals = np.array([np.nanstd(df.iloc[:, j].values) for j in x], dtype=float)

    plt.figure(figsize=(10, 5))
    plt.plot(x, mean_vals, label='Mean across sources', marker='o', linewidth=2.5)
    plt.fill_between(x, mean_vals - std_vals / 2.0, mean_vals + std_vals / 2.0, alpha=0.2)

    x_labels = SAMPLED_LAYER_LABELS[:len(x)] if len(SAMPLED_LAYER_LABELS) >= len(x) else x
    plt.xticks(x, x_labels, rotation=45, ha='right')
    plt.xlabel('Target Layer')
    plt.ylabel('Average Flip Rate')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    if save_pdf:
        out_pdf = os.path.splitext(out_png)[0] + '.pdf'
        plt.savefig(out_pdf, bbox_inches='tight')
        print(f"[OK] Saved: {out_png}, {out_pdf}")
    else:
        print(f"[OK] Saved: {out_png}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Qwen-only: average flip rate per target layer')
    parser.add_argument('--base_dir', type=str, default='/home/sc305/VLM/Final Experiments/negation/v-patchscope-demo-code/surf-exp')
    parser.add_argument('--output_dir', type=str, default='/home/sc305/VLM/Final Experiments/negation/v-patchscope-demo-code/surf-exp/comparisons/qwen_only')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--save_pdf', action='store_true', help='Also save a PDF alongside the PNG')
    args = parser.parse_args()

    global DEBUG
    DEBUG = bool(args.debug)

    items = load_qwen_results(args.base_dir)
    baseline = load_qwen_baseline(args.base_dir)
    shape = determine_shape(items)
    if shape == (0, 0):
        print('[ERROR] Could not determine matrix shape for Qwen results')
        return

    flip_df = compute_flip_rate(items, baseline, shape)

    out_png = os.path.join(args.output_dir, 'qwen_target_layer_avg_only_flip.png')
    plot_target_avg_line(flip_df, out_png, save_pdf=args.save_pdf)


if __name__ == '__main__':
    main()









