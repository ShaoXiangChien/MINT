import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, List, Tuple
import argparse

# Set global font sizes for better readability
plt.rcParams.update({
    'font.size': 18,          # Base font size
    'axes.titlesize': 22,     # Subplot titles
    'axes.labelsize': 20,     # Axis labels
    'xtick.labelsize': 18,    # X-axis tick labels
    'ytick.labelsize': 18,    # Y-axis tick labels
    'legend.fontsize': 18,    # Legend text
    'figure.titlesize': 24    # Main figure title
})

model_names = {
    "ds": "deepseek-vl2-tiny",
    "llava": "llava-v1.5-7b",
    "qwen": "qwen2-vl-7b-instruct"
}

def load_model_data(base_dir: str, models: List[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Load CSV data for all models and all plot types.
    
    Args:
        base_dir: Base directory containing model subdirectories
        models: List of model names (prefixes)
    
    Returns:
        Dictionary with structure: {model: {plot_type: dataframe}}
    """
    data = {}
    
    for model in models:
        model_dir = os.path.join(base_dir, model)
        data[model] = {}
        
        # Load correct rate matrix
        correct_csv = os.path.join(model_dir, f"{model}_correct_rate_matrix.csv")
        if os.path.exists(correct_csv):
            data[model]['correct_rate'] = pd.read_csv(correct_csv, index_col=0)
        
        # Load flip rate matrix  
        flip_csv = os.path.join(model_dir, f"{model}_flip_rate_matrix.csv")
        if os.path.exists(flip_csv):
            data[model]['flip_rate'] = pd.read_csv(flip_csv, index_col=0)
            
        # Load aggregates data
        agg_csv = os.path.join(model_dir, f"{model}_flip_rate_aggregates.csv")
        if os.path.exists(agg_csv):
            data[model]['aggregates'] = pd.read_csv(agg_csv)
    
    return data

def create_comparison_heatmaps(data: Dict[str, Dict[str, pd.DataFrame]], 
                              plot_type: str, 
                              title_prefix: str,
                              output_path: str,
                              vmin: float = None,
                              vmax: float = None,
                              save_pdf: bool = False):
    """
    Create a comparison plot with three subplots (one for each model).
    
    Args:
        data: Dictionary containing model data
        plot_type: Type of plot ('correct_rate' or 'flip_rate')
        title_prefix: Prefix for the main title
        output_path: Path to save the figure
        vmin, vmax: Color scale limits (optional)
    """
    models = list(data.keys())
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    # Determine global color scale if not provided
    if vmin is None or vmax is None:
        all_values = []
        for model in models:
            if plot_type in data[model]:
                values = data[model][plot_type].values
                all_values.extend(values[~np.isnan(values)])
        
        if all_values:
            if vmin is None:
                vmin = min(all_values)
            if vmax is None:
                vmax = max(all_values)
        else:
            vmin, vmax = 0, 1
    
    for i, model in enumerate(models):
        ax = axes[i]
        
        if plot_type in data[model]:
            df = data[model][plot_type]
            
            # Create heatmap
            im = ax.imshow(df.values, aspect='auto', origin='lower', 
                          vmin=vmin, vmax=vmax, cmap='viridis')
            
            # Set ticks and labels
            ax.set_xticks(np.arange(df.shape[1]))
            ax.set_xticklabels(df.columns, rotation=45, ha='right')
            ax.set_yticks(np.arange(df.shape[0]))
            ax.set_yticklabels(df.index)
            
            # Labels and title
            ax.set_xlabel("Layer Target")
            ax.set_ylabel("Layer Source")
            ax.set_title(f"{model_names[model]}")
            
            # Add colorbar to the rightmost subplot
            if i == len(models) - 1:
                plt.colorbar(im, ax=ax, shrink=0.8)
        else:
            ax.text(0.5, 0.5, f"No data for {model}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{model_names[model]}")
    
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    if save_pdf:
        pdf_path = os.path.splitext(output_path)[0] + '.pdf'
        plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    if save_pdf:
        pdf_path = os.path.splitext(output_path)[0] + '.pdf'
        print(f"[OK] Saved comparison plot: {output_path} and {pdf_path}")
    else:
        print(f"[OK] Saved comparison plot: {output_path}")

def create_aggregates_comparison(data: Dict[str, Dict[str, pd.DataFrame]], 
                               output_path: str,
                               save_pdf: bool = False):
    """
    Create a comparison plot for flip rate aggregates across models.
    
    Args:
        data: Dictionary containing model data
        output_path: Path to save the figure
    """
    models = list(data.keys())
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    for i, model in enumerate(models):
        ax = axes[i]
        
        if 'aggregates' in data[model]:
            df = data[model]['aggregates']
            
            # Plot the two aggregate measures
            ax.plot(df['layer_target'], df['median_flip_rate'], 
                   label="Median across sources", marker='o', linewidth=2)
            ax.plot(df['layer_target'], df['mean_flip_rate'], 
                   label="Mean across sources", marker='s', linewidth=2)
            
            # Formatting
            ax.set_xlabel("Layer Target")
            ax.set_ylabel("Flip Rate")
            ax.set_title(f"{model_names[model]}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            
        else:
            ax.text(0.5, 0.5, f"No data for {model}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{model_names[model]}")
    
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    if save_pdf:
        pdf_path = os.path.splitext(output_path)[0] + '.pdf'
        plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    if save_pdf:
        pdf_path = os.path.splitext(output_path)[0] + '.pdf'
        print(f"[OK] Saved aggregates comparison plot: {output_path} and {pdf_path}")
    else:
        print(f"[OK] Saved aggregates comparison plot: {output_path}")

def main():
    """
    Main function to generate all comparison plots.
    """
    parser = argparse.ArgumentParser(description="Generate comparison plots for spatial patching analysis")
    parser.add_argument("--base_dir", type=str, 
                       default="/home/sc305/VLM/Final Experiments/spatial-relationship/results/spatial-relationship",
                       help="Base directory containing model subdirectories")
    parser.add_argument("--models", nargs="+", default=["ds", "llava", "qwen"],
                       help="List of model names (prefixes)")
    parser.add_argument("--output_dir", type=str,
                       default="/home/sc305/VLM/Final Experiments/spatial-relationship/results/comparisons",
                       help="Output directory for comparison plots")
    parser.add_argument("--save_pdf", action="store_true", help="Also save figures as PDF alongside PNG")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load all model data
    print("Loading data for models:", args.models)
    data = load_model_data(args.base_dir, args.models)
    
    # Verify data was loaded
    for model in args.models:
        if model in data:
            print(f"[OK] Loaded data for {model}: {list(data[model].keys())}")
        else:
            print(f"[WARNING] No data found for {model}")
    
    # Generate comparison plots
    print("\nGenerating comparison plots...")
    
    # 1. Correct Rate Heatmap Comparison
    correct_output = os.path.join(args.output_dir, "correct_rate_comparison.png")
    create_comparison_heatmaps(data, 'correct_rate', 'Correct Rate', correct_output, save_pdf=args.save_pdf)
    
    # 2. Flip Rate Heatmap Comparison  
    flip_output = os.path.join(args.output_dir, "flip_rate_comparison.png")
    create_comparison_heatmaps(data, 'flip_rate', 'Flip Rate', flip_output, save_pdf=args.save_pdf)
    
    # 3. Aggregates Comparison
    agg_output = os.path.join(args.output_dir, "aggregates_comparison.png")
    create_aggregates_comparison(data, agg_output, save_pdf=args.save_pdf)
    
    print(f"\n[COMPLETE] All comparison plots saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
