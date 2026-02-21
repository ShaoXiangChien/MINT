#!/usr/bin/env python3
"""
Baseline Comparison Analysis

This script compares experimental results with baseline performance to understand
the true intervention effects. This is crucial for interpreting your results!

Key analyses:
1. Experimental vs Baseline accuracy comparison
2. Intervention effect calculation (experimental - baseline)
3. Statistical significance testing
4. Layer-specific intervention effects
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up matplotlib
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.dpi'] = 100

def load_baseline_data():
    """Load baseline data for all models."""
    results_dir = Path("./lg_results/")
    
    models = {
        'DeepSeek-VL2': 'ds-baseline-results.json',
        'LLaVA-1.5': 'llava-baseline-results.json', 
        'QWEN2-VL': 'qwen_baseline_results.json',
        'Qwen2.5-VL': 'qwen2.5_baseline_results.json'
    }
    
    baseline_data = {}
    
    for model_name, filename in models.items():
        file_path = results_dir / filename
        if file_path.exists():
            print(f"Loading {model_name} baseline...")
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Calculate baseline accuracy
            predictions = [sample['prediction'] for sample in data]
            accuracy = np.mean(predictions)
            
            baseline_data[model_name] = {
                'accuracy': accuracy,
                'samples': len(predictions),
                'predictions': predictions
            }
            print(f"  Baseline accuracy: {accuracy:.3f} ({len(predictions)} samples)")
        else:
            print(f"  Warning: {file_path} not found")
    
    return baseline_data

def load_experimental_data():
    """Load experimental data for all models."""
    results_dir = Path("./lg_results/")
    
    models = {
        'DeepSeek-VL2': {'file': 'ds-results.json', 'layers': list(range(0, 12, 2))},
        'LLaVA-1.5': {'file': 'llava-results.json', 'layers': list(range(0, 32, 3))},
        'QWEN2-VL': {'file': 'qwen_results.json', 'layers': list(range(0, 28, 3))},
        'Qwen2.5-VL': {'file': 'qwen2.5_results.json', 'layers': list(range(0, 28, 3))}
    }
    
    experimental_data = {}
    
    for model_name, info in models.items():
        file_path = results_dir / info['file']
        if file_path.exists():
            print(f"Loading {model_name} experimental...")
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if data:
                matrices = [np.array(sample['results']) for sample in data]
                avg_matrix = np.mean(matrices, axis=0)
                
                experimental_data[model_name] = {
                    'avg_matrix': avg_matrix,
                    'layers': info['layers'],
                    'samples': len(matrices),
                    'all_matrices': matrices
                }
                print(f"  Loaded {len(matrices)} experimental samples")
        else:
            print(f"  Warning: {file_path} not found")
    
    return experimental_data

def calculate_intervention_effects(experimental_data, baseline_data):
    """Calculate intervention effects (experimental - baseline)."""
    effects = {}
    
    for model_name in experimental_data.keys():
        if model_name in baseline_data:
            exp_matrix = experimental_data[model_name]['avg_matrix']
            baseline_acc = baseline_data[model_name]['accuracy']
            
            # Calculate intervention effect for each layer pair
            effect_matrix = exp_matrix - baseline_acc
            
            effects[model_name] = {
                'effect_matrix': effect_matrix,
                'layers': experimental_data[model_name]['layers'],
                'baseline_accuracy': baseline_acc,
                'exp_overall': np.mean(exp_matrix),
                'effect_overall': np.mean(effect_matrix)
            }
    
    return effects

def generate_baseline_comparison_plots(experimental_data, baseline_data, effects, save_path="./plots/"):
    """Generate comprehensive baseline comparison plots."""
    print("\nGenerating baseline comparison plots...")
    
    Path(save_path).mkdir(exist_ok=True)
    
    # Plot 1: Overall accuracy comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Bar chart: Experimental vs Baseline
    models = list(experimental_data.keys())
    exp_scores = [experimental_data[m]['avg_matrix'].mean() for m in models if m in baseline_data]
    baseline_scores = [baseline_data[m]['accuracy'] for m in models if m in baseline_data]
    valid_models = [m for m in models if m in baseline_data]
    
    x = np.arange(len(valid_models))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, exp_scores, width, label='Experimental (avg)', alpha=0.8, color='skyblue')
    bars2 = ax1.bar(x + width/2, baseline_scores, width, label='Baseline', alpha=0.8, color='lightcoral')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Experimental vs Baseline Performance')
    ax1.set_xticks(x)
    ax1.set_xticklabels(valid_models)
    ax1.legend()
    ax1.set_ylim(0, max(max(exp_scores), max(baseline_scores)) * 1.2)
    ax1.grid(True, alpha=0.3)
    
    # 2. Intervention effect magnitudes
    effect_scores = [effects[m]['effect_overall'] for m in valid_models if m in effects]
    bars3 = ax2.bar(valid_models, effect_scores, alpha=0.8, 
                    color=['green' if x > 0 else 'red' for x in effect_scores])
    
    for i, (model, score) in enumerate(zip(valid_models, effect_scores)):
        ax2.text(i, score + (0.01 if score > 0 else -0.01), f'{score:.3f}',
                ha='center', va='bottom' if score > 0 else 'top')
    
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Intervention Effect')
    ax2.set_title('Overall Intervention Effect\n(Experimental - Baseline)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Target layer intervention effects
    for model_name in valid_models:
        if model_name in effects:
            layers = effects[model_name]['layers']
            effect_matrix = effects[model_name]['effect_matrix']
            target_effects = np.mean(effect_matrix, axis=0)  # Average across source layers
            
            ax3.plot(layers, target_effects, marker='o', label=model_name, linewidth=2, markersize=6)
    
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Target Layer')
    ax3.set_ylabel('Intervention Effect')
    ax3.set_title('Intervention Effect per Target Layer')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Source layer intervention effects
    for model_name in valid_models:
        if model_name in effects:
            layers = effects[model_name]['layers']
            effect_matrix = effects[model_name]['effect_matrix']
            source_effects = np.mean(effect_matrix, axis=1)  # Average across target layers
            
            ax4.plot(layers, source_effects, marker='s', label=model_name, linewidth=2, markersize=6)
    
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Source Layer')
    ax4.set_ylabel('Intervention Effect')
    ax4.set_title('Intervention Effect per Source Layer')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(save_path) / "baseline_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

def generate_intervention_heatmaps(effects, save_path="./plots/"):
    """Generate intervention effect heatmaps."""
    print("Generating intervention effect heatmaps...")
    
    Path(save_path).mkdir(exist_ok=True)
    
    n_models = len(effects)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    for i, (model_name, effect_data) in enumerate(effects.items()):
        layers = effect_data['layers']
        effect_matrix = effect_data['effect_matrix']
        
        # Determine color scale range
        vmax = max(abs(effect_matrix.min()), abs(effect_matrix.max()))
        vmin = -vmax
        
        # Create heatmap
        im = axes[i].imshow(effect_matrix, vmin=vmin, vmax=vmax, cmap='RdBu_r')
        
        # Set labels and title
        axes[i].set_title(f'{model_name}\nIntervention Effect\n(Exp - Baseline: {effect_data["baseline_accuracy"]:.3f})')
        axes[i].set_xlabel('Target Layer')
        axes[i].set_ylabel('Source Layer')
        
        # Set layer labels
        axes[i].set_xticks(range(len(layers)))
        axes[i].set_xticklabels(layers, rotation=45)
        axes[i].set_yticks(range(len(layers)))
        axes[i].set_yticklabels(layers[::-1])  # Reverse for intuitive display
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        cbar.set_label('Effect Size')
        
        # Add text annotations
        for row in range(len(layers)):
            for col in range(len(layers)):
                value = effect_matrix[row, col]
                text_color = "white" if abs(value) > vmax * 0.5 else "black"
                text = axes[i].text(col, row, f'{value:.2f}',
                                  ha="center", va="center", 
                                  color=text_color, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(Path(save_path) / "intervention_effect_heatmaps.png", dpi=300, bbox_inches='tight')
    plt.show()

def calculate_statistical_significance(experimental_data, baseline_data):
    """Calculate statistical significance of intervention effects."""
    print("\nCalculating statistical significance...")
    
    significance_results = {}
    
    for model_name in experimental_data.keys():
        if model_name not in baseline_data:
            continue
            
        print(f"\n{model_name}:")
        
        exp_matrices = experimental_data[model_name]['all_matrices']
        baseline_acc = baseline_data[model_name]['accuracy']
        layers = experimental_data[model_name]['layers']
        
        # Calculate p-values for each layer pair using t-test approximation
        n_layers = len(layers)
        p_values = np.zeros((n_layers, n_layers))
        effect_sizes = np.zeros((n_layers, n_layers))
        
        for i in range(n_layers):
            for j in range(n_layers):
                # Get all samples for this layer pair
                layer_pair_results = [matrix[i, j] for matrix in exp_matrices]
                
                # Calculate effect size and approximate p-value
                mean_exp = np.mean(layer_pair_results)
                std_exp = np.std(layer_pair_results)
                n_samples = len(layer_pair_results)
                
                effect_size = mean_exp - baseline_acc
                effect_sizes[i, j] = effect_size
                
                # Approximate t-test (comparing sample mean to baseline)
                if std_exp > 0:
                    t_stat = (mean_exp - baseline_acc) / (std_exp / np.sqrt(n_samples))
                    # Approximate p-value (two-tailed)
                    p_value = 2 * (1 - abs(t_stat) / (abs(t_stat) + np.sqrt(n_samples - 1)))
                    p_values[i, j] = max(0, min(1, p_value))  # Clamp to [0,1]
                else:
                    p_values[i, j] = 1.0
        
        # Count significant effects
        significant_positive = np.sum((effect_sizes > 0) & (p_values < 0.05))
        significant_negative = np.sum((effect_sizes < 0) & (p_values < 0.05))
        total_pairs = n_layers * n_layers
        
        print(f"  Significant positive effects: {significant_positive}/{total_pairs}")
        print(f"  Significant negative effects: {significant_negative}/{total_pairs}")
        print(f"  Largest positive effect: {np.max(effect_sizes):.3f}")
        print(f"  Largest negative effect: {np.min(effect_sizes):.3f}")
        
        significance_results[model_name] = {
            'effect_sizes': effect_sizes,
            'p_values': p_values,
            'significant_positive': significant_positive,
            'significant_negative': significant_negative,
            'layers': layers
        }
    
    return significance_results

def print_detailed_analysis(experimental_data, baseline_data, effects):
    """Print detailed analysis results."""
    print("\n" + "="*80)
    print("DETAILED BASELINE COMPARISON ANALYSIS")
    print("="*80)
    
    for model_name in experimental_data.keys():
        if model_name not in baseline_data:
            continue
            
        print(f"\n{model_name}:")
        print("-" * 40)
        
        exp_data = experimental_data[model_name]
        baseline_acc = baseline_data[model_name]['accuracy']
        effect_data = effects[model_name]
        
        print(f"Baseline accuracy: {baseline_acc:.3f}")
        print(f"Experimental avg: {effect_data['exp_overall']:.3f}")
        print(f"Overall effect: {effect_data['effect_overall']:.3f}")
        print(f"Effect magnitude: {abs(effect_data['effect_overall']):.3f}")
        
        # Best and worst interventions
        effect_matrix = effect_data['effect_matrix']
        layers = effect_data['layers']
        
        best_idx = np.unravel_index(np.argmax(effect_matrix), effect_matrix.shape)
        worst_idx = np.unravel_index(np.argmin(effect_matrix), effect_matrix.shape)
        
        best_effect = effect_matrix[best_idx]
        worst_effect = effect_matrix[worst_idx]
        
        print(f"Best intervention: Layer {layers[best_idx[0]]}→{layers[best_idx[1]]} (+{best_effect:.3f})")
        print(f"Worst intervention: Layer {layers[worst_idx[0]]}→{layers[worst_idx[1]]} ({worst_effect:.3f})")
        
        # Positive vs negative effects
        positive_effects = np.sum(effect_matrix > 0)
        negative_effects = np.sum(effect_matrix < 0)
        zero_effects = np.sum(effect_matrix == 0)
        total = effect_matrix.size
        
        print(f"Positive effects: {positive_effects}/{total} ({100*positive_effects/total:.1f}%)")
        print(f"Negative effects: {negative_effects}/{total} ({100*negative_effects/total:.1f}%)")
        print(f"Zero effects: {zero_effects}/{total} ({100*zero_effects/total:.1f}%)")

def main():
    """Run the baseline comparison analysis."""
    print("Baseline Comparison Analysis for VLM Layer Intervention")
    print("=" * 70)
    
    # Load data
    print("Loading data...")
    baseline_data = load_baseline_data()
    experimental_data = load_experimental_data()
    
    if not baseline_data or not experimental_data:
        print("Error: Could not load required data files!")
        return
    
    # Calculate intervention effects
    print("\nCalculating intervention effects...")
    effects = calculate_intervention_effects(experimental_data, baseline_data)
    
    # Generate visualizations
    save_path = "./plots/"
    Path(save_path).mkdir(exist_ok=True)
    
    generate_baseline_comparison_plots(experimental_data, baseline_data, effects, save_path)
    generate_intervention_heatmaps(effects, save_path)
    
    # Statistical analysis
    significance = calculate_statistical_significance(experimental_data, baseline_data)
    
    # Print detailed results
    print_detailed_analysis(experimental_data, baseline_data, effects)
    
    print(f"\nBaseline comparison analysis complete!")
    print(f"Results saved to: {Path(save_path).absolute()}")
    print("\nGenerated files:")
    print("- baseline_comparison.png")
    print("- intervention_effect_heatmaps.png")

if __name__ == "__main__":
    main()


