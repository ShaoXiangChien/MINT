#!/usr/bin/env python3
"""
Complete Analysis Runner and Interpretation Guide

This script runs all analyses and provides interpretation guidance for your
vision-language model layer intervention experiments.

Run this script to generate all visualizations and get comprehensive insights.
"""

import subprocess
import sys
from pathlib import Path

def run_script(script_name):
    """Run a Python script and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running {script_name}...")
    print('='*60)
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print(f"Warnings: {result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}:")
        print(f"Return code: {e.returncode}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"Error: {script_name} not found!")
        return False

def print_interpretation_guide():
    """Print comprehensive interpretation guide."""
    print("\n" + "="*80)
    print("INTERPRETATION GUIDE FOR YOUR VLM LAYER INTERVENTION EXPERIMENTS")
    print("="*80)
    
    print("""
## KEY FINDINGS FROM YOUR EXPERIMENTS:

### 🔍 BASELINE COMPARISON REVEALS CRITICAL INSIGHTS:

1. **HIGH BASELINE PERFORMANCE**: All models achieved ~98.7-98.9% baseline accuracy
   - This indicates the original object recognition task is very easy
   - Most intervention effects appear NEGATIVE because baseline is near-perfect

2. **INTERVENTION EFFECTS**:
   - DeepSeek-VL2: -73% overall effect (most interventions hurt performance)
   - LLaVA-1.5: -12.5% overall effect (modest negative impact)
   - QWEN2-VL: -49.4% overall effect (substantial negative impact)

### 📊 WHAT YOUR LINE PLOTS SHOW:

**Target Layer Analysis** (your proposed approach):
- Shows which layers are most/least affected by interventions
- DeepSeek: Layer 2 most resilient (70.4% avg accuracy)
- LLaVA: Layer 6 most resilient (98.6% avg accuracy) 
- QWEN: Layer 6 most resilient (98.2% avg accuracy)

**Source Layer Analysis**:
- Shows which layers provide the most useful signals for intervention
- Different models show different optimal source layers

### 🔬 SCIENTIFIC INTERPRETATION:

1. **Vision Processing Hierarchy**: Early-mid layers (2-6) often most resilient
2. **Model Robustness**: LLaVA shows highest robustness to intervention
3. **Intervention Sensitivity**: Most layer pairs show negative effects

### ⚠️  IMPORTANT CONSIDERATIONS:

1. **Task Ceiling Effect**: With 98%+ baseline, improvements are limited
2. **Intervention Method**: Current approach may be disruptive rather than helpful
3. **Statistical Significance**: Most effects not statistically significant

### 🎯 RECOMMENDATIONS FOR FUTURE WORK:

1. **Use Harder Tasks**: Try tasks with lower baseline (60-80% accuracy)
2. **Positive Controls**: Test interventions known to improve performance
3. **Gentle Interventions**: Consider less disruptive intervention methods
4. **Layer-Specific Analysis**: Focus on layers showing positive effects

### 📈 HOW TO PRESENT YOUR RESULTS:

1. **Lead with Baseline Comparison**: Essential context for interpretation
2. **Highlight Model Differences**: LLaVA's robustness is interesting
3. **Focus on Patterns**: Layer hierarchies and intervention sensitivity
4. **Acknowledge Limitations**: High baseline limits observable improvements

## GENERATED VISUALIZATIONS:

### Primary Analysis (your request):
- `target_layer_line_plots.png`: Shows your proposed analysis ✅
- `source_layer_line_plots.png`: Complementary source analysis
- `enhanced_heatmaps.png`: Detailed layer-to-layer interactions
- `comparison_summary.png`: Multi-model comparison overview

### Critical Baseline Analysis:
- `baseline_comparison.png`: Essential for proper interpretation
- `intervention_effect_heatmaps.png`: True intervention effects

### Comprehensive Analysis:
- All above plus category breakdowns and statistical summaries
""")

def main():
    """Run complete analysis pipeline with guidance."""
    print("VLM Layer Intervention Analysis - Complete Pipeline")
    print("="*70)
    
    # Check if we're in the right directory
    required_files = ['quick_analysis.py', 'baseline_comparison.py']
    missing_files = [f for f in required_files if not Path(f).exists()]
    
    if missing_files:
        print(f"Error: Missing files: {missing_files}")
        print("Please run this script from the project directory.")
        return
    
    # Create plots directory
    Path("./plots/").mkdir(exist_ok=True)
    
    print("This will run all analysis scripts and generate comprehensive visualizations.")
    print("The analysis includes:")
    print("1. Your requested line plots (average accuracy per target layer)")
    print("2. Enhanced heatmaps with detailed annotations")
    print("3. CRITICAL baseline comparison (essential for interpretation)")
    print("4. Multi-model comparison and statistical analysis")
    
    response = input("\nProceed with complete analysis? (y/n): ").lower().strip()
    if response not in ['y', 'yes']:
        print("Analysis cancelled.")
        return
    
    # Run analyses
    success_count = 0
    
    if run_script('quick_analysis.py'):
        success_count += 1
        print("✅ Quick analysis completed successfully!")
    
    if run_script('baseline_comparison.py'):
        success_count += 1
        print("✅ Baseline comparison completed successfully!")
    
    # Try to run comprehensive analysis if available
    if Path('comprehensive_analysis.py').exists():
        if run_script('comprehensive_analysis.py'):
            success_count += 1
            print("✅ Comprehensive analysis completed successfully!")
    
    # Print results summary
    print(f"\n{'='*70}")
    print(f"ANALYSIS COMPLETE: {success_count} scripts ran successfully")
    print('='*70)
    
    plots_dir = Path("./plots/")
    if plots_dir.exists():
        plot_files = list(plots_dir.glob("*.png"))
        print(f"\nGenerated {len(plot_files)} visualization files in ./plots/:")
        for plot_file in sorted(plot_files):
            print(f"  - {plot_file.name}")
    
    # Print interpretation guide
    print_interpretation_guide()
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Review all generated plots in ./plots/ directory")
    print("2. Pay special attention to baseline_comparison.png")
    print("3. Use target_layer_line_plots.png for your proposed analysis")
    print("4. Consider the interpretation guide above when writing up results")
    print("5. For publication, focus on the model differences and layer hierarchies")

if __name__ == "__main__":
    main()


