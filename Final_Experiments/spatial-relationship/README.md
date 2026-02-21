# Spatial Relationship Analysis

This directory contains scripts for analyzing spatial patching results across different vision-language models.

## Files

- `analyze_spatial_patching.py` - Original analysis script for individual models
- `generate_comparison_plots.py` - Script to create comparison plots across models  
- `run_all_analyses.py` - Batch script to run all analyses

## Quick Start

### Option 1: Run everything at once
```bash
python run_all_analyses.py --threshold 0.5 --prop 0.5
```

### Option 2: Individual model analysis
```bash
# DS model
python analyze_spatial_patching.py --patched results/ds_results.json --baseline results/ds_baseline_results.json --outdir results/spatial-relationship/ds --threshold 0.5 --prop 0.5 --prefix ds_

# LLaVA model  
python analyze_spatial_patching.py --patched results/llava_results.json --baseline results/llava_baseline_results.json --outdir results/spatial-relationship/llava --threshold 0.5 --prop 0.5 --prefix llava_

# Qwen model
python analyze_spatial_patching.py --patched results/qwen_results.json --baseline results/qwen_baseline_results.json --outdir results/spatial-relationship/qwen --threshold 0.5 --prop 0.5 --prefix qwen_
```

### Option 3: Just generate comparison plots (if individual analyses already done)
```bash
python generate_comparison_plots.py
```

## Output

### Individual Model Results
Each model gets its own directory under `results/spatial-relationship/{model}/`:
- `{model}_correct_rate_matrix.csv` - Correct rate data
- `{model}_correct_rate_heatmap.png` - Correct rate heatmap
- `{model}_flip_rate_matrix.csv` - Flip rate data  
- `{model}_flip_rate_heatmap.png` - Flip rate heatmap
- `{model}_flip_rate_aggregates.csv` - Aggregate statistics
- `{model}_flip_rate_aggregates.png` - Aggregate plots

### Comparison Results
Comparison plots are saved to `results/comparisons/`:
- `correct_rate_comparison.png` - Side-by-side correct rate heatmaps
- `flip_rate_comparison.png` - Side-by-side flip rate heatmaps  
- `aggregates_comparison.png` - Side-by-side aggregate trend plots

## Understanding the Plots

### Correct Rate Heatmap
Shows the mean correctness across layer source (y-axis) and layer target (x-axis) combinations. Higher values (brighter colors) indicate better performance.

### Flip Rate Heatmap  
Shows the rate at which patching flips incorrect baseline predictions to correct ones. This measures the "recovery" capability of patching at different layer combinations.

### Aggregates Plot
Shows three trend lines across target layers:
- **Median**: Median flip rate across all source layers
- **Mean**: Mean flip rate across all source layers  
- **Prop≥threshold**: Proportion of source layers achieving the threshold flip rate

## Customization

### Changing Models
Edit the `models` list in `generate_comparison_plots.py` or use the `--models` argument:
```bash
python generate_comparison_plots.py --models ds llava qwen
```

### Changing Paths
Use the `--base_dir` and `--output_dir` arguments:
```bash
python generate_comparison_plots.py --base_dir /path/to/results --output_dir /path/to/output
```

### Threshold and Proportion Parameters
Both scripts accept `--threshold` and `--prop` parameters to customize the failure depth analysis.
