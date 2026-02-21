#!/usr/bin/env python3
"""Bootstrap Confidence Intervals for MINT Patching Experiments.

Computes nonparametric bootstrap CIs for the (source_layer x target_layer)
override-accuracy matrices produced by any MINT experiment.  Supports an
optional baseline comparison (file-based or fixed value).

This is the **unified** version that replaces the per-experiment copies
formerly found in ``mm-patching-ld/``, ``text-patching-ld/``,
``global-img-ld/``, ``spatial-relationship/``, and ``negation/surf-exp/``.

Usage::

    python -m evaluation.bootstrap_ci \
        --results_dir experiments/04_global_image_fusion/results/ \
        --out_dir experiments/04_global_image_fusion/analysis/ \
        --confidence 0.95
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def to_numpy_matrices(samples: List[Dict[str, Any]]) -> np.ndarray:
    """Stack per-sample result matrices into shape ``(N, src_layers, tgt_layers)``."""
    matrices = []
    for item in samples:
        matrices.append(np.array(item["results"], dtype=np.float64))
    if not matrices:
        return np.empty((0, 0, 0), dtype=np.float64)
    shapes = {m.shape for m in matrices}
    if len(shapes) != 1:
        raise ValueError(f"Inconsistent result matrix shapes: {shapes}")
    return np.stack(matrices, axis=0)


def percentile_ci(samples: np.ndarray, alpha: float) -> Tuple[float, float]:
    return (
        float(np.percentile(samples, 100 * (alpha / 2.0))),
        float(np.percentile(samples, 100 * (1.0 - alpha / 2.0))),
    )


def bootstrap_experimental(mats: np.ndarray, iterations: int, alpha: float) -> Dict[str, Any]:
    """Bootstrap the experimental override-accuracy matrix."""
    if mats.size == 0:
        nan = float("nan")
        return {"overall": {"mean": nan, "ci": [nan, nan]},
                "matrix": None, "target_mean": None, "source_mean": None, "_draws": {}}

    n, ns, nt = mats.shape
    overall_draws = np.empty(iterations)
    matrix_draws = np.empty((iterations, ns, nt))
    target_draws = np.empty((iterations, nt))
    source_draws = np.empty((iterations, ns))

    for b in range(iterations):
        idx = np.random.randint(0, n, size=n)
        m = mats[idx].mean(axis=0)
        matrix_draws[b] = m
        overall_draws[b] = m.mean()
        target_draws[b] = m.mean(axis=0)
        source_draws[b] = m.mean(axis=1)

    point_matrix = mats.mean(axis=0)
    return {
        "overall": {"mean": float(point_matrix.mean()),
                     "ci": list(percentile_ci(overall_draws, alpha))},
        "matrix": {"mean": point_matrix.tolist(),
                    "ci_lower": np.percentile(matrix_draws, 100 * alpha / 2, axis=0).tolist(),
                    "ci_upper": np.percentile(matrix_draws, 100 * (1 - alpha / 2), axis=0).tolist()},
        "target_mean": {"mean": point_matrix.mean(axis=0).tolist(),
                        "ci_lower": np.percentile(target_draws, 100 * alpha / 2, axis=0).tolist(),
                        "ci_upper": np.percentile(target_draws, 100 * (1 - alpha / 2), axis=0).tolist()},
        "source_mean": {"mean": point_matrix.mean(axis=1).tolist(),
                        "ci_lower": np.percentile(source_draws, 100 * alpha / 2, axis=0).tolist(),
                        "ci_upper": np.percentile(source_draws, 100 * (1 - alpha / 2), axis=0).tolist()},
        "_draws": {"overall": overall_draws, "matrix": matrix_draws,
                   "target": target_draws, "source": source_draws},
    }


def bootstrap_baseline(predictions: List[int], iterations: int, alpha: float) -> Dict[str, Any]:
    if not predictions:
        return {"mean": float("nan"), "ci": [float("nan"), float("nan")], "_draws": np.array([])}
    preds = np.asarray(predictions, dtype=np.float64)
    draws = np.array([preds[np.random.randint(0, len(preds), size=len(preds))].mean()
                      for _ in range(iterations)])
    return {"mean": float(preds.mean()), "ci": list(percentile_ci(draws, alpha)), "_draws": draws}


def compute_effect(exp_boot, base_boot, alpha):
    """Compute bootstrap effect (experimental - baseline) with significance."""
    exp_draws = exp_boot["_draws"]["overall"]
    base_draws = base_boot.get("_draws", np.array([]))
    iters = len(exp_draws)
    if base_draws.size == 0:
        base_draws = np.full(iters, base_boot["mean"])
    elif len(base_draws) != iters:
        base_draws = base_draws[np.random.randint(0, len(base_draws), size=iters)]

    effect = exp_draws - base_draws
    ci = percentile_ci(effect, alpha)
    sig = (ci[0] > 0) or (ci[1] < 0)
    return {
        "overall": {"mean": float(exp_boot["overall"]["mean"] - base_boot["mean"]),
                     "ci": list(ci), "significant": bool(sig)},
    }


def process_model(model_key, exp_file, baseline_file, iterations, confidence,
                   baseline_strategy, baseline_fixed):
    alpha = 1.0 - confidence
    exp_samples = load_json(exp_file) if exp_file.exists() else []
    mats = to_numpy_matrices(exp_samples)
    exp_boot = bootstrap_experimental(mats, iterations, alpha)

    if baseline_strategy == "file" and baseline_file and baseline_file.exists():
        data = load_json(baseline_file)
        preds = [int(item.get("prediction", 0)) for item in data]
        base_boot = bootstrap_baseline(preds, iterations, alpha)
    elif baseline_strategy == "fixed":
        base_boot = {"mean": baseline_fixed,
                     "ci": [baseline_fixed, baseline_fixed], "_draws": np.array([])}
    else:
        base_boot = {"mean": 0.0, "ci": [0.0, 0.0], "_draws": np.array([])}

    effect = compute_effect(exp_boot, base_boot, alpha) if baseline_strategy != "none" else None

    return {
        "model": model_key, "samples": len(exp_samples),
        "iterations": iterations, "confidence": confidence,
        "experimental": {k: v for k, v in exp_boot.items() if k != "_draws"},
        "baseline": {"overall": {"mean": base_boot["mean"], "ci": base_boot["ci"]},
                      "strategy": baseline_strategy},
        "effect": effect,
    }


def main():
    parser = argparse.ArgumentParser(description="Bootstrap CIs for MINT experiments")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--baseline_strategy", choices=["file", "fixed", "none"], default="fixed")
    parser.add_argument("--baseline_fixed", type=float, default=0.0)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Auto-discover result JSON files
    result_files = sorted(results_dir.glob("*.json"))
    all_results = {}

    for rf in result_files:
        model_key = rf.stem.replace("_results", "").replace("-results", "")
        baseline_file = results_dir / rf.name.replace("results", "baseline_results")
        print(f"Processing {model_key} ({rf.name})...")
        result = process_model(
            model_key, rf, baseline_file, args.iterations, args.confidence,
            args.baseline_strategy, args.baseline_fixed,
        )
        all_results[model_key] = result
        with open(out_dir / f"{model_key}_bootstrap.json", "w") as f:
            json.dump(result, f, indent=2)

    with open(out_dir / "bootstrap_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Summary saved to {out_dir / 'bootstrap_summary.json'}")


if __name__ == "__main__":
    np.random.seed(42)
    main()
