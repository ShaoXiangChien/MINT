#!/usr/bin/env bash
# Run bootstrap analysis and generate figures for all experiments.
set -e

RESULTS="results"
ANALYSIS="analysis"
FIGURES="figures"

mkdir -p "$ANALYSIS" "$FIGURES"

echo "=========================================="
echo "  MINT -- Running All Analyses"
echo "=========================================="

echo ""
echo "[1/3] Bootstrap Confidence Intervals"
python -m evaluation.bootstrap_ci \
    --results_dir "$RESULTS" \
    --out_dir "$ANALYSIS/bootstrap" \
    --iterations 10000 \
    --confidence 0.95

echo ""
echo "[2/3] Bootstrap Reports (tables + per-model plots)"
python -m evaluation.report_bootstrap \
    --summary "$ANALYSIS/bootstrap/bootstrap_summary.json" \
    --out_dir "$ANALYSIS/reports"

echo ""
echo "[3/3] Paper Figures"
python -m evaluation.generate_paper_figures \
    --experiment_dirs "$ANALYSIS/bootstrap" \
    --out_dir "$FIGURES"

echo ""
echo "Done. Tables in $ANALYSIS/reports/, Figures in $FIGURES/"
