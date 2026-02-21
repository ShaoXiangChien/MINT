#!/usr/bin/env python3
"""
Batch script to run spatial patching analysis for all three models
and then generate comparison plots.

This script automates the process of:
1. Running analyze_spatial_patching.py for each model (ds, llava, qwen)
2. Generating comparison plots across all models

Usage:
    python run_all_analyses.py [--threshold 0.5] [--prop 0.5]
"""

import subprocess
import os
import argparse
import sys

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ SUCCESS")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ FAILED")
        print("Error:", e.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description="Run spatial patching analysis for all models")
    parser.add_argument("--threshold", type=float, default=0.5, 
                       help="Threshold for Failure-Depth (default: 0.5)")
    parser.add_argument("--prop", type=float, default=0.5,
                       help="Proportion for prop-based Failure-Depth (default: 0.5)")
    parser.add_argument("--skip-individual", action="store_true",
                       help="Skip individual model analyses (only generate comparisons)")
    
    args = parser.parse_args()
    
    # Base directory
    base_dir = "/home/sc305/VLM/Final Experiments/spatial-relationship"
    os.chdir(base_dir)
    
    # Model configurations
    models = [
        {
            "name": "ds",
            "patched": "results/ds_results.json",
            "baseline": "results/ds_baseline_results.json",
            "outdir": "results/spatial-relationship/ds",
            "prefix": "ds_"
        },
        {
            "name": "llava", 
            "patched": "results/llava_results.json",
            "baseline": "results/llava_baseline_results.json",
            "outdir": "results/spatial-relationship/llava",
            "prefix": "llava_"
        },
        {
            "name": "qwen",
            "patched": "results/qwen_results.json", 
            "baseline": "results/qwen_baseline_results.json",
            "outdir": "results/spatial-relationship/qwen",
            "prefix": "qwen_"
        }
    ]
    
    success_count = 0
    
    # Run individual model analyses
    if not args.skip_individual:
        print("🚀 Starting individual model analyses...")
        
        for model in models:
            cmd = [
                "python", "analyze_spatial_patching.py",
                "--patched", model["patched"],
                "--baseline", model["baseline"], 
                "--outdir", model["outdir"],
                "--threshold", str(args.threshold),
                "--prop", str(args.prop),
                "--prefix", model["prefix"]
            ]
            
            description = f"Analyzing {model['name'].upper()} model"
            if run_command(cmd, description):
                success_count += 1
        
        print(f"\n📊 Individual analyses completed: {success_count}/{len(models)} successful")
    else:
        print("⏭️  Skipping individual model analyses...")
        success_count = len(models)  # Assume all succeeded for comparison generation
    
    # Generate comparison plots
    if success_count > 0:
        print(f"\n🎨 Generating comparison plots...")
        
        cmd = ["python", "generate_comparison_plots.py"]
        description = "Generating comparison plots across all models"
        
        if run_command(cmd, description):
            print("\n🎉 ALL ANALYSES COMPLETED SUCCESSFULLY!")
            print("\nGenerated files:")
            print("📁 Individual results: results/spatial-relationship/{ds,llava,qwen}/")
            print("📊 Comparison plots: results/comparisons/")
            print("   - correct_rate_comparison.png")
            print("   - flip_rate_comparison.png") 
            print("   - aggregates_comparison.png")
        else:
            print("\n⚠️  Individual analyses completed but comparison generation failed.")
    else:
        print("\n❌ No successful individual analyses - skipping comparison generation.")
        sys.exit(1)

if __name__ == "__main__":
    main()
