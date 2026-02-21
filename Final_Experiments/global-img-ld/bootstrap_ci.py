#!/usr/bin/env python3
"""
Bootstrap Confidence Intervals for VLM Layer Intervention Experiments

This script computes nonparametric bootstrap confidence intervals for metrics
derived from the experimental layer-to-layer intervention results and their
baselines. It supports all three models used in this project when their result
files are present in `./lg_results/`.

Metrics bootstrapped (per model):
- Overall experimental accuracy (averaged across all source/target pairs)
- Baseline accuracy (from corresponding baseline predictions)
- Overall intervention effect: experimental - baseline
- Per-layer-pair accuracy matrix
- Per-layer-pair intervention effect matrix
- Target-layer averages (across source) and Source-layer averages (across target)

Outputs (per model) are saved to `./analysis_results/bootstrap_ci/<model_key>_bootstrap.json`:
- Means, 95% CIs (percentile), and significance masks (CI excluding 0 for effects)

Usage:
    python bootstrap_ci.py --iterations 10000 --confidence 0.95

Notes:
- Bootstrap samples the experimental samples with replacement along the sample
  dimension. For baselines, we bootstrap the prediction list the same way.
- Effect distributions are computed by subtracting a baseline draw per
  iteration, accounting for baseline uncertainty.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np


def load_json(path: Path) -> Any:
    """Load JSON file and return parsed content. Handles one-line JSON as well."""
    with open(path, "r") as f:
        return json.load(f)


def to_numpy_matrices(samples: List[Dict[str, Any]]) -> np.ndarray:
    """Convert list of sample dicts with key 'results' into 3D numpy array [N, S, T]."""
    matrices: List[np.ndarray] = []
    for item in samples:
        # Each 'results' is a 2D list (source x target) of 0/1
        results = np.array(item["results"], dtype=np.float64)
        matrices.append(results)
    if not matrices:
        return np.empty((0, 0, 0), dtype=np.float64)
    # Ensure consistent shapes
    shapes = {m.shape for m in matrices}
    if len(shapes) != 1:
        raise ValueError(f"Inconsistent result matrix shapes found: {shapes}")
    return np.stack(matrices, axis=0)


def compute_means(mats: np.ndarray) -> Dict[str, Any]:
    """Compute point-estimate means for matrices [N, S, T]."""
    if mats.size == 0:
        return {
            "overall": np.nan,
            "matrix_mean": None,
            "target_mean": None,
            "source_mean": None,
        }
    # Average across samples -> [S, T]
    matrix_mean = mats.mean(axis=0)
    # Overall scalar mean across all entries
    overall = matrix_mean.mean()
    # Target mean (avg across source) -> length T
    target_mean = matrix_mean.mean(axis=0)
    # Source mean (avg across target) -> length S
    source_mean = matrix_mean.mean(axis=1)
    return {
        "overall": float(overall),
        "matrix_mean": matrix_mean,
        "target_mean": target_mean,
        "source_mean": source_mean,
    }


def percentile_ci(samples: np.ndarray, alpha: float) -> Tuple[float, float]:
    """Return lower/upper percentile bounds at 100*(alpha/2) and 100*(1-alpha/2)."""
    lower = np.percentile(samples, 100 * (alpha / 2.0))
    upper = np.percentile(samples, 100 * (1.0 - alpha / 2.0))
    return float(lower), float(upper)


def bootstrap_experimental(
    mats: np.ndarray,
    iterations: int,
    alpha: float,
) -> Dict[str, Any]:
    """
    Bootstrap experimental results by resampling the sample dimension with replacement.

    Returns dict of bootstrap distributions and CIs for:
    - overall
    - matrix (per [S, T])
    - target_mean (length T)
    - source_mean (length S)
    """
    if mats.size == 0:
        return {
            "overall": {"mean": np.nan, "ci": [np.nan, np.nan]},
            "matrix": None,
            "target_mean": None,
            "source_mean": None,
        }

    num_samples, num_sources, num_targets = mats.shape

    # Preallocate arrays for efficiency
    overall_draws = np.empty(iterations, dtype=np.float64)
    matrix_draws = np.empty((iterations, num_sources, num_targets), dtype=np.float64)
    target_draws = np.empty((iterations, num_targets), dtype=np.float64)
    source_draws = np.empty((iterations, num_sources), dtype=np.float64)

    for b in range(iterations):
        # Sample indices with replacement along sample axis
        idx = np.random.randint(0, num_samples, size=num_samples)
        mats_b = mats[idx]  # [N, S, T]
        m_mean = mats_b.mean(axis=0)  # [S, T]
        matrix_draws[b] = m_mean
        overall_draws[b] = m_mean.mean()
        target_draws[b] = m_mean.mean(axis=0)  # length T
        source_draws[b] = m_mean.mean(axis=1)  # length S

    # Compute point means and CIs
    point = compute_means(mats)
    overall_ci = percentile_ci(overall_draws, alpha)

    matrix_ci_lower = np.percentile(matrix_draws, 100 * (alpha / 2.0), axis=0)
    matrix_ci_upper = np.percentile(matrix_draws, 100 * (1.0 - alpha / 2.0), axis=0)

    target_ci_lower = np.percentile(target_draws, 100 * (alpha / 2.0), axis=0)
    target_ci_upper = np.percentile(target_draws, 100 * (1.0 - alpha / 2.0), axis=0)

    source_ci_lower = np.percentile(source_draws, 100 * (alpha / 2.0), axis=0)
    source_ci_upper = np.percentile(source_draws, 100 * (1.0 - alpha / 2.0), axis=0)

    return {
        "overall": {
            "mean": float(point["overall"]),
            "ci": [overall_ci[0], overall_ci[1]],
        },
        "matrix": {
            "mean": point["matrix_mean"].tolist(),
            "ci_lower": matrix_ci_lower.tolist(),
            "ci_upper": matrix_ci_upper.tolist(),
        },
        "target_mean": {
            "mean": point["target_mean"].tolist(),
            "ci_lower": target_ci_lower.tolist(),
            "ci_upper": target_ci_upper.tolist(),
        },
        "source_mean": {
            "mean": point["source_mean"].tolist(),
            "ci_lower": source_ci_lower.tolist(),
            "ci_upper": source_ci_upper.tolist(),
        },
        # Return draws for effect computation (we keep them aggregated to scalars)
        "_draws": {
            "overall": overall_draws,
            "matrix": matrix_draws,
            "target": target_draws,
            "source": source_draws,
        },
    }


def bootstrap_baseline(predictions: List[int], iterations: int, alpha: float) -> Dict[str, Any]:
    """Bootstrap baseline accuracy from 0/1 predictions."""
    if not predictions:
        return {"mean": np.nan, "ci": [np.nan, np.nan], "_draws": np.array([])}

    preds = np.asarray(predictions, dtype=np.float64)
    n = len(preds)
    draws = np.empty(iterations, dtype=np.float64)
    for b in range(iterations):
        idx = np.random.randint(0, n, size=n)
        draws[b] = preds[idx].mean()
    mean_val = float(preds.mean())
    ci = percentile_ci(draws, alpha)
    return {"mean": mean_val, "ci": [ci[0], ci[1]], "_draws": draws}


def compute_effect_bootstrap(
    exp_boot: Dict[str, Any],
    base_boot: Dict[str, Any],
    alpha: float,
) -> Dict[str, Any]:
    """
    Combine experimental and baseline bootstrap draws to form effect draws.
    Effect = Experimental - Baseline. We subtract a baseline draw from each
    experimental draw to incorporate baseline uncertainty.
    """
    # Draw arrays
    base_draws: np.ndarray = base_boot.get("_draws", np.array([]))
    exp_overall: np.ndarray = exp_boot["_draws"]["overall"]
    exp_matrix: np.ndarray = exp_boot["_draws"]["matrix"]  # [B, S, T]
    exp_target: np.ndarray = exp_boot["_draws"]["target"]  # [B, T]
    exp_source: np.ndarray = exp_boot["_draws"]["source"]  # [B, S]

    iterations = exp_overall.shape[0]
    if base_draws.size == 0:
        # Use point baseline mean as fixed if no baseline draws available
        baseline_point = float(base_boot.get("mean", 0.0))
        base_draws = np.full(iterations, baseline_point, dtype=np.float64)

    # If baseline draws count differs, resample to match experimental iterations
    if base_draws.shape[0] != iterations:
        # Simple approach: sample with replacement from baseline draws
        idx = np.random.randint(0, base_draws.shape[0], size=iterations)
        base_draws = base_draws[idx]

    # Compute effect draws
    effect_overall_draws = exp_overall - base_draws
    effect_matrix_draws = exp_matrix - base_draws[:, None, None]
    effect_target_draws = exp_target - base_draws[:, None]
    effect_source_draws = exp_source - base_draws[:, None]

    # Point estimates from means
    point_matrix_mean = np.array(exp_boot["matrix"]["mean"]) - base_boot["mean"]
    point_target_mean = np.array(exp_boot["target_mean"]["mean"]) - base_boot["mean"]
    point_source_mean = np.array(exp_boot["source_mean"]["mean"]) - base_boot["mean"]
    point_overall_mean = float(exp_boot["overall"]["mean"] - base_boot["mean"])  # type: ignore

    # CIs
    overall_ci = percentile_ci(effect_overall_draws, alpha)
    matrix_ci_lower = np.percentile(effect_matrix_draws, 100 * (alpha / 2.0), axis=0)
    matrix_ci_upper = np.percentile(effect_matrix_draws, 100 * (1.0 - alpha / 2.0), axis=0)
    target_ci_lower = np.percentile(effect_target_draws, 100 * (alpha / 2.0), axis=0)
    target_ci_upper = np.percentile(effect_target_draws, 100 * (1.0 - alpha / 2.0), axis=0)
    source_ci_lower = np.percentile(effect_source_draws, 100 * (alpha / 2.0), axis=0)
    source_ci_upper = np.percentile(effect_source_draws, 100 * (1.0 - alpha / 2.0), axis=0)

    # Significance masks (CI excludes 0)
    matrix_sig = (matrix_ci_lower > 0) | (matrix_ci_upper < 0)
    target_sig = (target_ci_lower > 0) | (target_ci_upper < 0)
    source_sig = (source_ci_lower > 0) | (source_ci_upper < 0)
    overall_sig = (overall_ci[0] > 0) or (overall_ci[1] < 0)

    return {
        "overall": {
            "mean": point_overall_mean,
            "ci": [overall_ci[0], overall_ci[1]],
            "significant": bool(overall_sig),
        },
        "matrix": {
            "mean": point_matrix_mean.tolist(),
            "ci_lower": matrix_ci_lower.tolist(),
            "ci_upper": matrix_ci_upper.tolist(),
            "significant": matrix_sig.astype(int).tolist(),
        },
        "target_mean": {
            "mean": point_target_mean.tolist(),
            "ci_lower": target_ci_lower.tolist(),
            "ci_upper": target_ci_upper.tolist(),
            "significant": target_sig.astype(int).tolist(),
        },
        "source_mean": {
            "mean": point_source_mean.tolist(),
            "ci_lower": source_ci_lower.tolist(),
            "ci_upper": source_ci_upper.tolist(),
            "significant": source_sig.astype(int).tolist(),
        },
    }


def process_model(
    model_key: str,
    exp_file: Path,
    baseline_file: Path,
    iterations: int,
    confidence: float,
) -> Dict[str, Any]:
    """Load data, run bootstraps, and return result dict for one model."""
    alpha = 1.0 - confidence

    # Load experimental samples
    exp_samples: List[Dict[str, Any]] = []
    if exp_file.exists():
        exp_samples = load_json(exp_file)
    else:
        print(f"  Warning: Experimental file not found: {exp_file}")

    # Load baseline predictions
    baseline_predictions: List[int] = []
    if baseline_file.exists():
        baseline_data = load_json(baseline_file)
        try:
            baseline_predictions = [int(item["prediction"]) for item in baseline_data]
        except Exception as e:
            raise ValueError(f"Baseline file format error: {baseline_file}: {e}")
    else:
        print(f"  Warning: Baseline file not found: {baseline_file}")

    # Convert experimental matrices
    mats = to_numpy_matrices(exp_samples)
    if mats.size == 0:
        print(f"  Warning: No experimental matrices found for {model_key}")

    # Bootstrap experimental
    exp_boot = bootstrap_experimental(mats, iterations=iterations, alpha=alpha)
    # Bootstrap baseline
    base_boot = bootstrap_baseline(baseline_predictions, iterations=iterations, alpha=alpha)
    # Effects
    effect_boot = compute_effect_bootstrap(exp_boot, base_boot, alpha=alpha)

    # Assemble result
    result: Dict[str, Any] = {
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
        "baseline": {
            "overall": {"mean": base_boot["mean"], "ci": base_boot["ci"]},
        },
        "effect": effect_boot,
    }

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap CIs for VLM experiments")
    parser.add_argument("--results_dir", type=str, default="./lg_results/", help="Directory with results JSONs")
    parser.add_argument("--out_dir", type=str, default="./analysis_results/bootstrap_ci/", help="Directory to save outputs")
    parser.add_argument("--iterations", type=int, default=10000, help="Number of bootstrap iterations")
    parser.add_argument("--confidence", type=float, default=0.95, help="Confidence level (e.g., 0.95)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Model files mapping (align with other analysis scripts)
    models = {
        "DeepSeek-VL2": {"exp": "ds-results.json", "baseline": "ds-baseline-results.json"},
        "LLaVA-1.5": {"exp": "llava-results.json", "baseline": "llava-baseline-results.json"},
        "QWEN2-VL": {"exp": "qwen_results.json", "baseline": "qwen_baseline_results.json"},
        "Qwen2.5-VL": {"exp": "qwen2.5_results.json", "baseline": "qwen2.5_baseline_results.json"},
    }

    all_results: Dict[str, Any] = {}

    print("Bootstrapping confidence intervals...")
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

        # Save per-model JSON
        out_file = out_dir / f"{files['exp'].replace('.json', '')}_bootstrap.json"
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: {out_file}")

    # Save combined summary
    summary_file = out_dir / "bootstrap_summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined summary saved to: {summary_file}")


if __name__ == "__main__":
    # Set a deterministic RNG seed for reproducibility of one run
    np.random.seed(42)
    main()


