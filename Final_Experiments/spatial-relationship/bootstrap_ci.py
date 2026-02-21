#!/usr/bin/env python3
"""
Bootstrap Confidence Intervals for Spatial-Relationship Experiments

This script computes nonparametric bootstrap confidence intervals for accuracy
metrics derived from spatial-relationship layer intervention results and their
baselines. It supports the three models when their result files are present in
./results/.

Experimental input format:
- Each entry contains "results": a 2D list (source x target). For spatial runs,
  each cell is a dict with keys like {"correct", "prediction", ...}. We extract
  the 0/1 correctness to form matrices.

Baseline input format:
- A flat list of dicts with key "correct" (0/1 or bool). We bootstrap baseline
  accuracy as a scalar and subtract it from experimental draws to form an effect.

Outputs:
- Per-model JSON under ./analysis_results/bootstrap_ci/{exp_file_stem}_bootstrap.json
- Combined summary JSON at ./analysis_results/bootstrap_ci/bootstrap_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np


def load_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def extract_cell_value(cell: Any) -> float:
    """Extract 0/1 correctness from a cell that may be an int/float or dict.

    Accepts:
      - int/float (already 0/1)
      - dict with key 'correct' (bool/int)
      - dict with key 'prediction' (fallback, coerced to int)
    """
    if isinstance(cell, (int, float)):
        return float(cell)
    if isinstance(cell, dict):
        if "correct" in cell:
            return 1.0 if bool(cell["correct"]) else 0.0
        if "prediction" in cell:
            try:
                return float(int(cell["prediction"]))
            except Exception:
                return 0.0
    # Unknown format -> treat as 0
    return 0.0


def to_numpy_matrices(samples: List[Dict[str, Any]]) -> np.ndarray:
    """Convert experimental entries to 3D numpy array [N, S, T] of correctness.

    Each sample['results'] is a 2D list (source x target) of cells. Cells may be
    0/1 or dicts; we extract 'correct'.
    """
    matrices: List[np.ndarray] = []
    for item in samples:
        raw = item.get("results", [])
        if not raw:
            continue
        mat_rows: List[List[float]] = []
        for row in raw:
            mat_rows.append([extract_cell_value(cell) for cell in row])
        mat = np.array(mat_rows, dtype=np.float64)
        matrices.append(mat)
    if not matrices:
        return np.empty((0, 0, 0), dtype=np.float64)
    shapes = {m.shape for m in matrices}
    if len(shapes) != 1:
        raise ValueError(f"Inconsistent result matrix shapes found: {shapes}")
    return np.stack(matrices, axis=0)


def compute_means(mats: np.ndarray) -> Dict[str, Any]:
    if mats.size == 0:
        return {"overall": np.nan, "matrix_mean": None, "target_mean": None, "source_mean": None}
    matrix_mean = mats.mean(axis=0)
    overall = matrix_mean.mean()
    target_mean = matrix_mean.mean(axis=0)
    source_mean = matrix_mean.mean(axis=1)
    return {
        "overall": float(overall),
        "matrix_mean": matrix_mean,
        "target_mean": target_mean,
        "source_mean": source_mean,
    }


def percentile_ci(samples: np.ndarray, alpha: float) -> Tuple[float, float]:
    lower = np.percentile(samples, 100 * (alpha / 2.0))
    upper = np.percentile(samples, 100 * (1.0 - alpha / 2.0))
    return float(lower), float(upper)


def bootstrap_experimental(mats: np.ndarray, iterations: int, alpha: float) -> Dict[str, Any]:
    if mats.size == 0:
        return {
            "overall": {"mean": np.nan, "ci": [np.nan, np.nan]},
            "matrix": None,
            "target_mean": None,
            "source_mean": None,
        }
    num_samples, num_sources, num_targets = mats.shape
    overall_draws = np.empty(iterations, dtype=np.float64)
    matrix_draws = np.empty((iterations, num_sources, num_targets), dtype=np.float64)
    target_draws = np.empty((iterations, num_targets), dtype=np.float64)
    source_draws = np.empty((iterations, num_sources), dtype=np.float64)
    for b in range(iterations):
        idx = np.random.randint(0, num_samples, size=num_samples)
        mats_b = mats[idx]
        m_mean = mats_b.mean(axis=0)
        matrix_draws[b] = m_mean
        overall_draws[b] = m_mean.mean()
        target_draws[b] = m_mean.mean(axis=0)
        source_draws[b] = m_mean.mean(axis=1)
    point = compute_means(mats)
    overall_ci = percentile_ci(overall_draws, alpha)
    matrix_ci_lower = np.percentile(matrix_draws, 100 * (alpha / 2.0), axis=0)
    matrix_ci_upper = np.percentile(matrix_draws, 100 * (1.0 - alpha / 2.0), axis=0)
    target_ci_lower = np.percentile(target_draws, 100 * (alpha / 2.0), axis=0)
    target_ci_upper = np.percentile(target_draws, 100 * (1.0 - alpha / 2.0), axis=0)
    source_ci_lower = np.percentile(source_draws, 100 * (alpha / 2.0), axis=0)
    source_ci_upper = np.percentile(source_draws, 100 * (1.0 - alpha / 2.0), axis=0)
    return {
        "overall": {"mean": float(point["overall"]), "ci": [overall_ci[0], overall_ci[1]]},
        "matrix": {"mean": point["matrix_mean"].tolist(), "ci_lower": matrix_ci_lower.tolist(), "ci_upper": matrix_ci_upper.tolist()},
        "target_mean": {"mean": point["target_mean"].tolist(), "ci_lower": target_ci_lower.tolist(), "ci_upper": target_ci_upper.tolist()},
        "source_mean": {"mean": point["source_mean"].tolist(), "ci_lower": source_ci_lower.tolist(), "ci_upper": source_ci_upper.tolist()},
        "_draws": {"overall": overall_draws, "matrix": matrix_draws, "target": target_draws, "source": source_draws},
    }


def bootstrap_baseline_correct(correct_list: List[Any], iterations: int, alpha: float) -> Dict[str, Any]:
    if not correct_list:
        return {"mean": np.nan, "ci": [np.nan, np.nan], "_draws": np.array([])}
    preds = np.asarray([1 if bool(x) else 0 for x in correct_list], dtype=np.float64)
    n = len(preds)
    draws = np.empty(iterations, dtype=np.float64)
    for b in range(iterations):
        idx = np.random.randint(0, n, size=n)
        draws[b] = preds[idx].mean()
    mean_val = float(preds.mean())
    ci = percentile_ci(draws, alpha)
    return {"mean": mean_val, "ci": [ci[0], ci[1]], "_draws": draws}


def compute_effect_bootstrap(exp_boot: Dict[str, Any], base_boot: Dict[str, Any], alpha: float) -> Dict[str, Any]:
    base_draws: np.ndarray = base_boot.get("_draws", np.array([]))
    exp_overall: np.ndarray = exp_boot["_draws"]["overall"]
    exp_matrix: np.ndarray = exp_boot["_draws"]["matrix"]
    exp_target: np.ndarray = exp_boot["_draws"]["target"]
    exp_source: np.ndarray = exp_boot["_draws"]["source"]
    iterations = exp_overall.shape[0]
    if base_draws.size == 0:
        baseline_point = float(base_boot.get("mean", 0.0))
        base_draws = np.full(iterations, baseline_point, dtype=np.float64)
    if base_draws.shape[0] != iterations:
        idx = np.random.randint(0, base_draws.shape[0], size=iterations)
        base_draws = base_draws[idx]
    effect_overall_draws = exp_overall - base_draws
    effect_matrix_draws = exp_matrix - base_draws[:, None, None]
    effect_target_draws = exp_target - base_draws[:, None]
    effect_source_draws = exp_source - base_draws[:, None]
    point_matrix_mean = np.array(exp_boot["matrix"]["mean"]) - base_boot["mean"]
    point_target_mean = np.array(exp_boot["target_mean"]["mean"]) - base_boot["mean"]
    point_source_mean = np.array(exp_boot["source_mean"]["mean"]) - base_boot["mean"]
    point_overall_mean = float(exp_boot["overall"]["mean"] - base_boot["mean"])  # type: ignore
    overall_ci = percentile_ci(effect_overall_draws, alpha)
    matrix_ci_lower = np.percentile(effect_matrix_draws, 100 * (alpha / 2.0), axis=0)
    matrix_ci_upper = np.percentile(effect_matrix_draws, 100 * (1.0 - alpha / 2.0), axis=0)
    target_ci_lower = np.percentile(effect_target_draws, 100 * (alpha / 2.0), axis=0)
    target_ci_upper = np.percentile(effect_target_draws, 100 * (1.0 - alpha / 2.0), axis=0)
    source_ci_lower = np.percentile(effect_source_draws, 100 * (alpha / 2.0), axis=0)
    source_ci_upper = np.percentile(effect_source_draws, 100 * (1.0 - alpha / 2.0), axis=0)
    matrix_sig = (matrix_ci_lower > 0) | (matrix_ci_upper < 0)
    target_sig = (target_ci_lower > 0) | (target_ci_upper < 0)
    source_sig = (source_ci_lower > 0) | (source_ci_upper < 0)
    overall_sig = (overall_ci[0] > 0) or (overall_ci[1] < 0)
    return {
        "overall": {"mean": point_overall_mean, "ci": [overall_ci[0], overall_ci[1]], "significant": bool(overall_sig)},
        "matrix": {"mean": point_matrix_mean.tolist(), "ci_lower": matrix_ci_lower.tolist(), "ci_upper": matrix_ci_upper.tolist(), "significant": matrix_sig.astype(int).tolist()},
        "target_mean": {"mean": point_target_mean.tolist(), "ci_lower": target_ci_lower.tolist(), "ci_upper": target_ci_upper.tolist(), "significant": target_sig.astype(int).tolist()},
        "source_mean": {"mean": point_source_mean.tolist(), "ci_lower": source_ci_lower.tolist(), "ci_upper": source_ci_upper.tolist(), "significant": source_sig.astype(int).tolist()},
    }


def process_model(
    model_key: str,
    exp_file: Path,
    baseline_file: Path,
    iterations: int,
    confidence: float,
) -> Dict[str, Any]:
    alpha = 1.0 - confidence
    exp_samples: List[Dict[str, Any]] = []
    if exp_file.exists():
        exp_samples = load_json(exp_file)
    else:
        print(f"  Warning: Experimental file not found: {exp_file}")
    baseline_correct: List[int] = []
    if baseline_file.exists():
        baseline_data = load_json(baseline_file)
        try:
            baseline_correct = [int(1 if bool(item.get("correct", False)) else 0) for item in baseline_data]
        except Exception as e:
            raise ValueError(f"Baseline file format error: {baseline_file}: {e}")
    else:
        print(f"  Warning: Baseline file not found: {baseline_file}")
    mats = to_numpy_matrices(exp_samples)
    if mats.size == 0:
        print(f"  Warning: No experimental matrices found for {model_key}")
    exp_boot = bootstrap_experimental(mats, iterations=iterations, alpha=alpha)
    base_boot = bootstrap_baseline_correct(baseline_correct, iterations=iterations, alpha=alpha)
    effect_boot = compute_effect_bootstrap(exp_boot, base_boot, alpha=alpha)
    return {
        "model": model_key,
        "samples": len(exp_samples),
        "iterations": iterations,
        "confidence": confidence,
        "experimental": {
            "overall": exp_boot["overall"],
            "matrix": exp_boot["matrix"],
            "target_mean": exp_boot["target_mean"],
            "source_mean": exp_boot["source_mean"],
        },
        "baseline": {"overall": {"mean": base_boot["mean"], "ci": base_boot["ci"]}},
        "effect": effect_boot,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap CIs for spatial-relationship experiments")
    parser.add_argument("--results_dir", type=str, default="./results/", help="Directory with results JSONs")
    parser.add_argument("--out_dir", type=str, default="./analysis_results/bootstrap_ci/", help="Directory to save outputs")
    parser.add_argument("--iterations", type=int, default=10000, help="Number of bootstrap iterations")
    parser.add_argument("--confidence", type=float, default=0.95, help="Confidence level (e.g., 0.95)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = {
        "DeepSeek-VL2": {"exp": "ds_results.json", "baseline": "ds_baseline_results.json"},
        "LLaVA-1.5": {"exp": "llava_results.json", "baseline": "llava_baseline_results.json"},
        "QWEN2-VL": {"exp": "qwen_results.json", "baseline": "qwen_baseline_results.json"},
    }

    all_results: Dict[str, Any] = {}

    print("Bootstrapping confidence intervals (spatial-relationship)...")
    print(f"  Iterations: {args.iterations} | Confidence: {args.confidence}")

    for model_key, files in models.items():
        exp_path = results_dir / files["exp"]
        base_path = results_dir / files["baseline"]
        if not exp_path.exists():
            print(f"\n{model_key}: Experimental file missing ({exp_path.name}). Skipping.")
            continue
        print(f"\n{model_key}:")
        print(f"  Experimental: {exp_path.name}")
        print(f"  Baseline:     {base_path.name if base_path.exists() else 'MISSING'}")

        result = process_model(
            model_key=model_key,
            exp_file=exp_path,
            baseline_file=base_path,
            iterations=args.iterations,
            confidence=args.confidence,
        )
        all_results[model_key] = result

        out_file = out_dir / f"{files['exp'].replace('.json', '')}_bootstrap.json"
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: {out_file}")

    summary_file = out_dir / "bootstrap_summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined summary saved to: {summary_file}")


if __name__ == "__main__":
    np.random.seed(42)
    main()


