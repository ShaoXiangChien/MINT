Here’s exactly which scripts produce each appendix artifact and how to run them.

- Combined figure (`appendix_combined.(png|pdf)`)
  - Script: `appendix_figure.py`
  - Input: `analysis_results/bootstrap_ci/bootstrap_summary.json`
  - Output dir: `analysis_results/bootstrap_ci/report/`

- Overall effect bars (`overall_effect_bars.png`) and overall metrics table (`overall_metrics.md` and `.csv`)
  - Script: `report_bootstrap.py`
  - Input: `analysis_results/bootstrap_ci/bootstrap_summary.json`
  - Output dir: `analysis_results/bootstrap_ci/report/`

- Prerequisite (creates the bootstrap summary used by both scripts)
  - Script: `bootstrap_ci.py`
  - Outputs: per-model bootstrap JSONs + `analysis_results/bootstrap_ci/bootstrap_summary.json`

Run these commands from `global-img-ld`:
```bash
# 1) Compute bootstrap summary (prereq for all reporting)
python3 bootstrap_ci.py --iterations 10000 --confidence 0.95

# 2) Generate tables + figures (overall bars, overall_metrics.md/csv, per-model lines/heatmaps)
python3 report_bootstrap.py \
  --summary ./analysis_results/bootstrap_ci/bootstrap_summary.json \
  --out_dir ./analysis_results/bootstrap_ci/report/

# 3) Generate combined appendix figure with subplots
python3 appendix_figure.py \
  --summary ./analysis_results/bootstrap_ci/bootstrap_summary.json \
  --out_dir ./analysis_results/bootstrap_ci/report/
```

Artifacts created in `analysis_results/bootstrap_ci/report/`:
- `appendix_combined.(png|pdf)`
- `overall_effect_bars.png`
- `overall_metrics.md`, `overall_metrics.csv`
- Plus: `{Model}_target_effect_line.png`, `{Model}_effect_heatmap_with_sig.png`, `{Model}_significance_heatmap.png`
