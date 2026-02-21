#!/usr/bin/env python3
"""
Bootstrap Confidence Intervals for Negation surf-exp (Flip-Rate)

This script computes nonparametric bootstrap CIs for flip-rate matrices derived
from surf-exp negation experiments. Flip-rate is defined over samples where the
baseline prediction is wrong (ground-truth is 'no' → 0) as the proportion of
patched predictions that become correct (==0).

Per model, we compute:
- Overall flip-rate (mean across all [source, target] cells)
- Flip-rate matrix with 95% CIs
- Target-layer mean flip-rates (avg across sources)
- Source-layer mean flip-rates (avg across targets)
- Significance masks (CI lower bound > 0)

Inputs (default under base_dir):
  Qwen:
    - patched:  qwen/results/results_qwen.json
    - baseline: qwen/results/qwen_results_baseline.json
  DeepSeek:
    - patched:  deepseek/results/results_deepseek.json
    - baseline: deepseek/results/deepseek_results_baseline.json
  LLaVA:
    - patched:  llava/results/results_llava.json
    - baseline: llava/results/llava_results_baseline.json

Outputs:
  - Per-model JSON under ./analysis_results/bootstrap_ci/{model_key}_bootstrap.json
  - Combined summary JSON at ./analysis_results/bootstrap_ci/bootstrap_summary.json

Usage:
  python3 bootstrap_flip_ci.py --base_dir . --iterations 10000 --confidence 0.95
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def load_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def coerce_prediction(pred: Any) -> Optional[int]:
    if isinstance(pred, (int, float)):
        v = int(pred)
        if v in (0, 1):
            return v
        return 1 if v > 0 else 0
    if isinstance(pred, str):
        t = pred.strip().lower()
        t = t.replace(".", "").replace("!", "").replace("?", "").replace('"', "'").strip()
        if t.startswith("yes"):
            return 1
        if t.startswith("no"):
            return 0
    return None


def extract_shape(items: List[dict]) -> Tuple[int, int]:
    shapes: Dict[Tuple[int, int], int] = {}
    for it in items:
        try:
            arr = np.array(it.get("results"), dtype=object)
        except Exception:
            continue
        if arr.ndim != 2:
            continue
        key = (arr.shape[0], arr.shape[1])
        shapes[key] = shapes.get(key, 0) + 1
    if not shapes:
        return (0, 0)
    return sorted(shapes.items(), key=lambda kv: kv[1], reverse=True)[0][0]


def build_baseline_correct_map(baseline_rows: List[dict]) -> Dict[int, bool]:
    # Ground truth is 0 (negation). Correct iff prediction == 0.
    m: Dict[int, bool] = {}
    for row in baseline_rows:
        sid = row.get("sample_id")
        if sid is None:
            continue
        pv = row.get("prediction")
        if pv is None:
            pv = row.get("response")
        pred = coerce_prediction(pv)
        m[int(sid)] = (pred == 0) if pred is not None else False
    return m


def collect_usable_samples(patched: List[dict], baseline_map: Dict[int, bool], shape: Tuple[int, int]) -> List[Tuple[int, np.ndarray]]:
    rows, cols = shape
    usable: List[Tuple[int, np.ndarray]] = []
    for it in patched:
        sid = it.get("sample_id")
        if sid is None:
            continue
        if int(sid) not in baseline_map:
            continue
        if baseline_map[int(sid)] is True:
            # Baseline already correct; not eligible for flip-rate
            continue
        try:
            arr = np.array(it.get("results"), dtype=object)
        except Exception:
            continue
        if arr.ndim != 2 or arr.shape != (rows, cols):
            continue
        # Convert to correctness under negation (0 is correct); treat unparsable as np.nan
        matrix = np.full((rows, cols), np.nan, dtype=float)
        for i in range(rows):
            for j in range(cols):
                pred = arr[i, j]
                val = None
                if isinstance(pred, dict):
                    val = coerce_prediction(pred.get("prediction", pred.get("result")))
                else:
                    val = coerce_prediction(pred)
                if val is None:
                    matrix[i, j] = np.nan
                else:
                    matrix[i, j] = 1.0 if (val == 0) else 0.0
        usable.append((int(sid), matrix))
    return usable


def bootstrap_flip_rates(
    matrices: List[np.ndarray],
    iterations: int,
    alpha: float,
) -> Dict[str, Any]:
    if not matrices:
        return {
            "overall": {"mean": np.nan, "ci": [np.nan, np.nan]},
            "matrix": None,
            "target_mean": None,
            "source_mean": None,
        }

    mats = np.stack(matrices, axis=0)  # [N, S, T] with NaNs allowed
    num_samples, num_sources, num_targets = mats.shape

    # Helper: nanmean along axis=0 for a resampled set
    def mean_over_samples(indices: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        sub = mats[indices]  # [n, S, T]
        m = np.nanmean(sub, axis=0)  # [S, T]
        overall = float(np.nanmean(m))
        target = np.nanmean(m, axis=0)  # [T]
        source = np.nanmean(m, axis=1)  # [S]
        return m, overall, target, source

    # Point estimates using all samples without resampling
    point_matrix, point_overall, point_target, point_source = mean_over_samples(np.arange(num_samples))

    # Bootstrap draws
    overall_draws = np.empty(iterations, dtype=np.float64)
    matrix_draws = np.empty((iterations, num_sources, num_targets), dtype=np.float64)
    target_draws = np.empty((iterations, num_targets), dtype=np.float64)
    source_draws = np.empty((iterations, num_sources), dtype=np.float64)

    rng = np.random.default_rng(42)
    for b in range(iterations):
        idx = rng.integers(0, num_samples, size=num_samples)
        m_b, overall_b, target_b, source_b = mean_over_samples(idx)
        matrix_draws[b] = m_b
        overall_draws[b] = overall_b
        target_draws[b] = target_b
        source_draws[b] = source_b

    def pct_ci(arr: np.ndarray) -> Tuple[float, float]:
        lo = float(np.nanpercentile(arr, 100 * (alpha / 2.0)))
        hi = float(np.nanpercentile(arr, 100 * (1.0 - alpha / 2.0)))
        return lo, hi

    overall_ci = pct_ci(overall_draws)
    matrix_ci_lower = np.nanpercentile(matrix_draws, 100 * (alpha / 2.0), axis=0)
    matrix_ci_upper = np.nanpercentile(matrix_draws, 100 * (1.0 - alpha / 2.0), axis=0)
    target_ci_lower = np.nanpercentile(target_draws, 100 * (alpha / 2.0), axis=0)
    target_ci_upper = np.nanpercentile(target_draws, 100 * (1.0 - alpha / 2.0), axis=0)
    source_ci_lower = np.nanpercentile(source_draws, 100 * (alpha / 2.0), axis=0)
    source_ci_upper = np.nanpercentile(source_draws, 100 * (1.0 - alpha / 2.0), axis=0)

    # Significance: CI lower > 0 indicates significant positive flip-rate
    sig_matrix = (matrix_ci_lower > 0).astype(int)
    sig_target = (target_ci_lower > 0).astype(int)
    sig_source = (source_ci_lower > 0).astype(int)
    sig_overall = overall_ci[0] > 0

    return {
        "overall": {"mean": float(point_overall), "ci": [overall_ci[0], overall_ci[1]], "significant": bool(sig_overall)},
        "matrix": {
            "mean": point_matrix.tolist(),
            "ci_lower": matrix_ci_lower.tolist(),
            "ci_upper": matrix_ci_upper.tolist(),
            "significant": sig_matrix.tolist(),
        },
        "target_mean": {
            "mean": point_target.tolist(),
            "ci_lower": target_ci_lower.tolist(),
            "ci_upper": target_ci_upper.tolist(),
            "significant": sig_target.tolist(),
        },
        "source_mean": {
            "mean": point_source.tolist(),
            "ci_lower": source_ci_lower.tolist(),
            "ci_upper": source_ci_upper.tolist(),
            "significant": sig_source.tolist(),
        },
    }


def process_model(
    model_key: str,
    patched_path: Path,
    baseline_path: Path,
    iterations: int,
    confidence: float,
) -> Dict[str, Any]:
    alpha = 1.0 - confidence

    if not patched_path.exists():
        print(f"  Missing patched file: {patched_path}")
        return {}
    if not baseline_path.exists():
        print(f"  Missing baseline file: {baseline_path}")
        return {}

    patched = load_json(patched_path)
    baseline_rows = load_json(baseline_path)

    baseline_map = build_baseline_correct_map(baseline_rows)
    shape = extract_shape(patched)
    if shape == (0, 0):
        print("  Could not determine matrix shape; skipping")
        return {}

    usable = collect_usable_samples(patched, baseline_map, shape)
    if not usable:
        print("  No usable samples (baseline-wrong intersection); skipping")
        return {}

    matrices = [m for _sid, m in usable]
    flip_boot = bootstrap_flip_rates(matrices, iterations=iterations, alpha=alpha)

    return {
        "model": model_key,
        "samples": len(usable),
        "iterations": iterations,
        "confidence": confidence,
        "flip": flip_boot,
        "shape": {"rows": shape[0], "cols": shape[1]},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Bootstrap CIs for surf-exp negation flip-rate")
    ap.add_argument("--base_dir", type=str, default="./")
    ap.add_argument("--out_dir", type=str, default="./analysis_results/bootstrap_ci/")
    ap.add_argument("--iterations", type=int, default=10000)
    ap.add_argument("--confidence", type=float, default=0.95)
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = {
        "QWEN2-VL": {
            "patched": base_dir / "qwen" / "results" / "results_qwen.json",
            "baseline": base_dir / "qwen" / "results" / "qwen_results_baseline.json",
        },
        "DeepSeek-VL2": {
            "patched": base_dir / "deepseek" / "results" / "results_deepseek.json",
            "baseline": base_dir / "deepseek" / "results" / "deepseek_results_baseline.json",
        },
        "LLaVA-1.5": {
            "patched": base_dir / "llava" / "results" / "results_llava.json",
            "baseline": base_dir / "llava" / "results" / "llava_results_baseline.json",
        },
    }

    print("Bootstrapping flip-rate CIs (surf-exp negation)...")
    print(f"  Iterations: {args.iterations} | Confidence: {args.confidence}")

    summary: Dict[str, Any] = {}
    for model_key, paths in models.items():
        print(f"\n{model_key}:")
        print(f"  Patched:  {paths['patched'].name}")
        print(f"  Baseline: {paths['baseline'].name}")
        result = process_model(
            model_key=model_key,
            patched_path=paths["patched"],
            baseline_path=paths["baseline"],
            iterations=args.iterations,
            confidence=args.confidence,
        )
        if not result:
            continue
        summary[model_key] = result
        out_file = out_dir / f"{model_key.replace(' ', '_')}_bootstrap.json"
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: {out_file}")

    summary_file = out_dir / "bootstrap_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nCombined summary saved to: {summary_file}")


if __name__ == "__main__":
    np.random.seed(42)
    main()


