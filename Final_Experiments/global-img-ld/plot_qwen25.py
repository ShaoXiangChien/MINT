#!/usr/bin/env python3
"""
Quick Plot Script for Qwen2.5-VL Results

This script generates all key visualizations specifically for Qwen2.5-VL,
including comparisons with other models when available.

Usage: python plot_qwen25.py
Outputs: plots in ./plots/ directory
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up matplotlib
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['figure.dpi'] = 100

def load_qwen25_data():
    """Load Qwen2.5-VL experimental and baseline data."""
    results_dir = Path("./lg_results/")
    
    # Load experimental results
    exp_path = results_dir / "qwen2.5_results.json"
    baseline_path = results_dir / "qwen2.5_baseline_results.json"
    
    data = {}
    
    if exp_path.exists():
        print(f"Loading Qwen2.5-VL experimental data...")
        with open(exp_path, 'r') as f:
            exp_data = json.load(f)
        
        if exp_data:
            matrices = [np.array(sample['results']) for sample in exp_data]
            avg_matrix = np.mean(matrices, axis=0)
            data['experimental'] = {
                'avg_matrix': avg_matrix,
                'all_matrices': matrices,
                'samples': len(matrices),
                'layers': list(range(0, 28, 3))
            }
            print(f"  Loaded {len(matrices)} experimental samples")
        else:
            print("  Warning: No experimental data found")
    else:
        print(f"  Error: {exp_path} not found!")
        return None
    
    if baseline_path.exists():
        print(f"Loading Qwen2.5-VL baseline data...")
        with open(baseline_path, 'r') as f:
            baseline_data = json.load(f)
        
        predictions = [sample['prediction'] for sample in baseline_data]
        data['baseline'] = {
            'accuracy': np.mean(predictions),
            'predictions': predictions,
            'samples': len(predictions)
        }
        print(f"  Loaded {len(predictions)} baseline samples")
        print(f"  Baseline accuracy: {data['baseline']['accuracy']:.3f}")
    else:
        print(f"  Warning: {baseline_path} not found")
    
    return data

def plot_qwen25_heatmap(data, save_path):
    """Generate heatmap for Qwen2.5-VL."""
    print("\nGenerating Qwen2.5-VL heatmap...")
    
    avg_matrix = data['experimental']['avg_matrix']
    layers = data['experimental']['layers']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Experimental accuracy heatmap
    im1 = ax1.imshow(avg_matrix, vmin=0, vmax=1, cmap='Blues')
    ax1.set_title('Qwen2.5-VL: Experimental Accuracy')
    ax1.set_xlabel('Target Layer')
    ax1.set_ylabel('Source Layer')
    ax1.set_xticks(range(len(layers)))
    ax1.set_xticklabels(layers, rotation=45)
    ax1.set_yticks(range(len(layers)))
    ax1.set_yticklabels(layers[::-1])
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # Add text annotations
    for i in range(len(layers)):
        for j in range(len(layers)):
            text = ax1.text(j, i, f'{avg_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    # Plot 2: Intervention effect (if baseline available)
    if 'baseline' in data:
        baseline_acc = data['baseline']['accuracy']
        effect_matrix = avg_matrix - baseline_acc
        
        vmax = max(abs(effect_matrix.min()), abs(effect_matrix.max()))
        im2 = ax2.imshow(effect_matrix, vmin=-vmax, vmax=vmax, cmap='RdBu_r')
        ax2.set_title(f'Qwen2.5-VL: Intervention Effect\n(vs Baseline: {baseline_acc:.3f})')
        ax2.set_xlabel('Target Layer')
        ax2.set_ylabel('Source Layer')
        ax2.set_xticks(range(len(layers)))
        ax2.set_xticklabels(layers, rotation=45)
        ax2.set_yticks(range(len(layers)))
        ax2.set_yticklabels(layers[::-1])
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        # Add text annotations
        for i in range(len(layers)):
            for j in range(len(layers)):
                value = effect_matrix[i, j]
                color = "white" if abs(value) > vmax * 0.5 else "black"
                text = ax2.text(j, i, f'{value:.2f}',
                              ha="center", va="center", color=color, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path / "qwen25_heatmap.png", dpi=300, bbox_inches='tight')
    plt.savefig(save_path / "qwen25_heatmap.pdf", bbox_inches='tight')
    print(f"  Saved: {save_path / 'qwen25_heatmap.png'}")
    plt.close()

def plot_qwen25_line_plots(data, save_path):
    """Generate line plots for target and source layer analysis."""
    print("Generating Qwen2.5-VL line plots...")
    
    avg_matrix = data['experimental']['avg_matrix']
    layers = data['experimental']['layers']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Average accuracy per target layer
    target_avg = np.mean(avg_matrix, axis=0)
    target_std = np.std(avg_matrix, axis=0)
    
    ax1.plot(layers, target_avg, marker='o', linewidth=2.5, markersize=8, 
             label='Qwen2.5-VL', color='blue')
    ax1.fill_between(layers, target_avg - target_std/2, target_avg + target_std/2, 
                     alpha=0.2, color='blue')
    
    if 'baseline' in data:
        baseline_acc = data['baseline']['accuracy']
        ax1.axhline(y=baseline_acc, color='red', linestyle='--', 
                   label=f'Baseline ({baseline_acc:.3f})', linewidth=2)
    
    ax1.set_xlabel('Target Layer')
    ax1.set_ylabel('Average Accuracy')
    ax1.set_title('Average Accuracy per Target Layer\n(Averaged across all source layers)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Plot 2: Best source performance per target layer
    best_source = np.max(avg_matrix, axis=0)
    ax2.plot(layers, best_source, marker='s', linewidth=2.5, markersize=8, 
             label='Best Source', color='green')
    
    if 'baseline' in data:
        ax2.axhline(y=baseline_acc, color='red', linestyle='--', 
                   label=f'Baseline ({baseline_acc:.3f})', linewidth=2)
    
    ax2.set_xlabel('Target Layer')
    ax2.set_ylabel('Best Accuracy')
    ax2.set_title('Best Source Performance per Target Layer')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Plot 3: Average effectiveness per source layer
    source_avg = np.mean(avg_matrix, axis=1)
    source_std = np.std(avg_matrix, axis=1)
    
    ax3.plot(layers, source_avg, marker='o', linewidth=2.5, markersize=8, 
             label='Qwen2.5-VL', color='blue')
    ax3.fill_between(layers, source_avg - source_std/2, source_avg + source_std/2, 
                     alpha=0.2, color='blue')
    
    if 'baseline' in data:
        ax3.axhline(y=baseline_acc, color='red', linestyle='--', 
                   label=f'Baseline ({baseline_acc:.3f})', linewidth=2)
    
    ax3.set_xlabel('Source Layer')
    ax3.set_ylabel('Average Effectiveness')
    ax3.set_title('Average Effectiveness per Source Layer\n(Averaged across all target layers)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # Plot 4: Intervention effect per target layer (if baseline available)
    if 'baseline' in data:
        baseline_acc = data['baseline']['accuracy']
        effect_target = target_avg - baseline_acc
        
        colors = ['green' if e > 0 else 'red' for e in effect_target]
        ax4.bar(layers, effect_target, color=colors, alpha=0.7, width=2)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax4.set_xlabel('Target Layer')
        ax4.set_ylabel('Intervention Effect')
        ax4.set_title('Intervention Effect per Target Layer\n(Experimental - Baseline)')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Baseline data not available', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_xlabel('Target Layer')
        ax4.set_ylabel('Intervention Effect')
        ax4.set_title('Intervention Effect per Target Layer')
    
    plt.tight_layout()
    plt.savefig(save_path / "qwen25_line_plots.png", dpi=300, bbox_inches='tight')
    plt.savefig(save_path / "qwen25_line_plots.pdf", bbox_inches='tight')
    print(f"  Saved: {save_path / 'qwen25_line_plots.png'}")
    plt.close()

def print_qwen25_stats(data):
    """Print summary statistics for Qwen2.5-VL."""
    print("\n" + "="*60)
    print("QWEN2.5-VL SUMMARY STATISTICS")
    print("="*60)
    
    avg_matrix = data['experimental']['avg_matrix']
    layers = data['experimental']['layers']
    
    print(f"\nExperimental Data:")
    print(f"  Samples: {data['experimental']['samples']}")
    print(f"  Layers tested: {layers}")
    print(f"  Overall accuracy: {np.mean(avg_matrix):.3f} ± {np.std(avg_matrix):.3f}")
    print(f"  Best performance: {np.max(avg_matrix):.3f}")
    print(f"  Worst performance: {np.min(avg_matrix):.3f}")
    
    # Best and worst layer pairs
    best_idx = np.unravel_index(np.argmax(avg_matrix), avg_matrix.shape)
    worst_idx = np.unravel_index(np.argmin(avg_matrix), avg_matrix.shape)
    
    best_source = layers[best_idx[0]]
    best_target = layers[best_idx[1]]
    worst_source = layers[worst_idx[0]]
    worst_target = layers[worst_idx[1]]
    
    print(f"  Best layer pair: {best_source}→{best_target} ({np.max(avg_matrix):.3f})")
    print(f"  Worst layer pair: {worst_source}→{worst_target} ({np.min(avg_matrix):.3f})")
    
    # Target layer statistics
    target_avgs = np.mean(avg_matrix, axis=0)
    best_target_idx = np.argmax(target_avgs)
    worst_target_idx = np.argmin(target_avgs)
    
    print(f"\nTarget Layer Analysis:")
    print(f"  Best target layer: {layers[best_target_idx]} (avg: {target_avgs[best_target_idx]:.3f})")
    print(f"  Worst target layer: {layers[worst_target_idx]} (avg: {target_avgs[worst_target_idx]:.3f})")
    
    # Source layer statistics
    source_avgs = np.mean(avg_matrix, axis=1)
    best_source_idx = np.argmax(source_avgs)
    worst_source_idx = np.argmin(source_avgs)
    
    print(f"\nSource Layer Analysis:")
    print(f"  Best source layer: {layers[best_source_idx]} (avg: {source_avgs[best_source_idx]:.3f})")
    print(f"  Worst source layer: {layers[worst_source_idx]} (avg: {source_avgs[worst_source_idx]:.3f})")
    
    if 'baseline' in data:
        baseline_acc = data['baseline']['accuracy']
        print(f"\nBaseline Comparison:")
        print(f"  Baseline accuracy: {baseline_acc:.3f}")
        print(f"  Overall intervention effect: {np.mean(avg_matrix) - baseline_acc:.3f}")
        print(f"  Best intervention effect: {np.max(avg_matrix) - baseline_acc:.3f}")
        print(f"  Worst intervention effect: {np.min(avg_matrix) - baseline_acc:.3f}")
        
        effect_matrix = avg_matrix - baseline_acc
        positive = np.sum(effect_matrix > 0)
        negative = np.sum(effect_matrix < 0)
        zero = np.sum(effect_matrix == 0)
        total = effect_matrix.size
        
        print(f"  Positive effects: {positive}/{total} ({100*positive/total:.1f}%)")
        print(f"  Negative effects: {negative}/{total} ({100*negative/total:.1f}%)")
        print(f"  Zero effects: {zero}/{total} ({100*zero/total:.1f}%)")

def main():
    """Main function to generate Qwen2.5-VL plots."""
    print("Qwen2.5-VL Visualization Script")
    print("="*60)
    
    # Load data
    data = load_qwen25_data()
    
    if data is None or 'experimental' not in data:
        print("\nError: Could not load Qwen2.5-VL data!")
        print("Make sure qwen2.5_results.json exists in ./lg_results/")
        return
    
    # Create plots directory
    save_path = Path("./plots/")
    save_path.mkdir(exist_ok=True)
    
    # Generate visualizations
    plot_qwen25_heatmap(data, save_path)
    plot_qwen25_line_plots(data, save_path)
    
    # Print statistics
    print_qwen25_stats(data)
    
    print(f"\n{'='*60}")
    print("QWEN2.5-VL VISUALIZATION COMPLETE!")
    print(f"{'='*60}")
    print(f"Plots saved to: {save_path.absolute()}")
    print("\nGenerated files:")
    print("  - qwen25_heatmap.png / .pdf")
    print("  - qwen25_line_plots.png / .pdf")
    print("\nTo generate plots comparing with other models, run:")
    print("  python quick_analysis.py")
    print("  python baseline_comparison.py")
    print("  python comprehensive_analysis.py")

if __name__ == "__main__":
    main()




