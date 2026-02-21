#!/usr/bin/env python3
"""
Comprehensive Analysis Script for Vision-Language Model Layer Intervention Experiments

This script analyzes experimental results from layer-to-layer intervention experiments
on vision-language models (DeepSeek-VL2, LLaVA-1.5, QWEN2-VL).

Features:
- Load and preprocess experimental and baseline results
- Generate enhanced heatmaps (experimental, baseline, difference)
- Create line plots for target/source layer analysis
- Compare intervention effects across models
- Category-specific performance breakdown
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set up matplotlib for better visualization
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['figure.dpi'] = 100

class VLMAnalyzer:
    """Comprehensive analyzer for vision-language model intervention experiments."""
    
    def __init__(self, results_dir="./lg_results/"):
        """
        Initialize the analyzer with results directory.
        
        Args:
            results_dir (str): Path to directory containing experiment results
        """
        self.results_dir = Path(results_dir)
        self.models = {
            'DeepSeek-VL2': {'exp': 'ds-results.json', 'baseline': 'ds-baseline-results.json', 'layers': list(range(0, 12, 2))},
            'LLaVA-1.5': {'exp': 'llava-results.json', 'baseline': 'llava-baseline-results.json', 'layers': list(range(0, 32, 3))},
            'QWEN2-VL': {'exp': 'qwen_results.json', 'baseline': 'qwen_baseline_results.json', 'layers': list(range(0, 28, 3))},
            'Qwen2.5-VL': {'exp': 'qwen2.5_results.json', 'baseline': 'qwen2.5_baseline_results.json', 'layers': list(range(0, 28, 3))}
        }
        self.data = {}
        self.processed_data = {}
        
    def load_data(self):
        """Load experimental and baseline data for all models."""
        print("Loading experimental data...")
        
        for model_name, files in self.models.items():
            print(f"  Loading {model_name}...")
            self.data[model_name] = {}
            
            # Load experimental results
            exp_path = self.results_dir / files['exp']
            if exp_path.exists():
                with open(exp_path, 'r') as f:
                    self.data[model_name]['experimental'] = json.load(f)
                print(f"    Experimental: {len(self.data[model_name]['experimental'])} samples")
            else:
                print(f"    Warning: {exp_path} not found")
                self.data[model_name]['experimental'] = []
            
            # Load baseline results
            baseline_path = self.results_dir / files['baseline']
            if baseline_path.exists():
                with open(baseline_path, 'r') as f:
                    self.data[model_name]['baseline'] = json.load(f)
                print(f"    Baseline: {len(self.data[model_name]['baseline'])} samples")
            else:
                print(f"    Warning: {baseline_path} not found")
                self.data[model_name]['baseline'] = []
    
    def preprocess_data(self):
        """Preprocess data to calculate average matrices and statistics."""
        print("\nPreprocessing data...")
        
        for model_name in self.models.keys():
            if model_name not in self.data:
                continue
                
            print(f"  Processing {model_name}...")
            self.processed_data[model_name] = {}
            
            # Process experimental data
            if self.data[model_name]['experimental']:
                exp_matrices = [np.array(sample['results']) for sample in self.data[model_name]['experimental']]
                self.processed_data[model_name]['exp_avg'] = np.mean(exp_matrices, axis=0)
                self.processed_data[model_name]['exp_std'] = np.std(exp_matrices, axis=0)
                self.processed_data[model_name]['exp_samples'] = len(exp_matrices)
                
                # Calculate baseline accuracy for experimental samples
                baseline_acc = self._calculate_baseline_accuracy(model_name, 'experimental')
                self.processed_data[model_name]['exp_baseline_acc'] = baseline_acc
            
            # Process baseline data (for comparison)
            if self.data[model_name]['baseline']:
                baseline_predictions = [sample['prediction'] for sample in self.data[model_name]['baseline']]
                self.processed_data[model_name]['baseline_acc'] = np.mean(baseline_predictions)
                self.processed_data[model_name]['baseline_samples'] = len(baseline_predictions)
            
            # Calculate intervention effect (experimental - baseline)
            if 'exp_avg' in self.processed_data[model_name]:
                baseline_val = self.processed_data[model_name].get('exp_baseline_acc', 0)
                exp_avg = self.processed_data[model_name]['exp_avg']
                self.processed_data[model_name]['intervention_effect'] = exp_avg - baseline_val
    
    def _calculate_baseline_accuracy(self, model_name, data_type):
        """Calculate baseline accuracy for samples that have experimental results."""
        samples = self.data[model_name][data_type]
        if not samples:
            return 0.0
        
        # For experimental samples, baseline would be when no intervention is applied
        # We can estimate this as the overall average across all matrix positions
        matrices = [np.array(sample['results']) for sample in samples]
        return np.mean(matrices)
    
    def generate_enhanced_heatmaps(self, save_path="./analysis_results/"):
        """Generate enhanced heatmaps showing experimental, baseline, and difference."""
        print("\nGenerating enhanced heatmaps...")
        
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True)
        
        # Create figure with subplots for each model
        n_models = len([m for m in self.models.keys() if m in self.processed_data])
        fig, axes = plt.subplots(2, n_models, figsize=(6*n_models, 10))
        if n_models == 1:
            axes = axes.reshape(2, 1)
        
        for i, (model_name, model_info) in enumerate(self.models.items()):
            if model_name not in self.processed_data:
                continue
                
            layers = model_info['layers']
            processed = self.processed_data[model_name]
            
            # Top row: Experimental heatmap
            if 'exp_avg' in processed:
                im1 = axes[0, i].imshow(processed['exp_avg'], vmin=0, vmax=1, cmap='Blues')
                axes[0, i].set_title(f'{model_name}\nExperimental Results')
                
                # Set layer labels
                axes[0, i].set_xticks(range(len(layers)))
                axes[0, i].set_xticklabels(layers, rotation=45)
                axes[0, i].set_yticks(range(len(layers)))
                axes[0, i].set_yticklabels(layers[::-1])  # Reverse for intuitive display
                axes[0, i].set_xlabel('Target Layer')
                axes[0, i].set_ylabel('Source Layer')
                
                # Add colorbar
                plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)
                
                # Add text annotations
                for row in range(len(layers)):
                    for col in range(len(layers)):
                        text = axes[0, i].text(col, row, f'{processed["exp_avg"][row, col]:.2f}',
                                             ha="center", va="center", color="black", fontsize=8)
            
            # Bottom row: Intervention effect (experimental - baseline)
            if 'intervention_effect' in processed:
                effect_matrix = processed['intervention_effect']
                vmax = max(abs(effect_matrix.min()), abs(effect_matrix.max()))
                im2 = axes[1, i].imshow(effect_matrix, vmin=-vmax, vmax=vmax, cmap='RdBu_r')
                axes[1, i].set_title(f'{model_name}\nIntervention Effect')
                
                # Set layer labels
                axes[1, i].set_xticks(range(len(layers)))
                axes[1, i].set_xticklabels(layers, rotation=45)
                axes[1, i].set_yticks(range(len(layers)))
                axes[1, i].set_yticklabels(layers[::-1])
                axes[1, i].set_xlabel('Target Layer')
                axes[1, i].set_ylabel('Source Layer')
                
                # Add colorbar
                plt.colorbar(im2, ax=axes[1, i], fraction=0.046, pad=0.04)
                
                # Add text annotations
                for row in range(len(layers)):
                    for col in range(len(layers)):
                        text = axes[1, i].text(col, row, f'{effect_matrix[row, col]:.2f}',
                                             ha="center", va="center", color="black", fontsize=8)
        
        plt.tight_layout()
        plt.savefig(save_path / "enhanced_heatmaps.png", dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_target_layer_analysis(self, save_path="./analysis_results/"):
        """Generate line plots for target layer performance analysis."""
        print("\nGenerating target layer analysis...")
        
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Average accuracy per target layer
        for model_name, model_info in self.models.items():
            if model_name not in self.processed_data or 'exp_avg' not in self.processed_data[model_name]:
                continue
                
            layers = model_info['layers']
            exp_avg = self.processed_data[model_name]['exp_avg']
            
            # Calculate average accuracy for each target layer (average across source layers)
            target_layer_avg = np.mean(exp_avg, axis=0)
            target_layer_std = np.std(exp_avg, axis=0)
            
            ax1.plot(layers, target_layer_avg, marker='o', label=model_name, linewidth=2, markersize=6)
            ax1.fill_between(layers, 
                           target_layer_avg - target_layer_std/2, 
                           target_layer_avg + target_layer_std/2, 
                           alpha=0.2)
        
        ax1.set_xlabel('Target Layer')
        ax1.set_ylabel('Average Accuracy')
        ax1.set_title('Average Accuracy per Target Layer\n(Averaged across all source layers)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Plot 2: Intervention effect per target layer
        for model_name, model_info in self.models.items():
            if model_name not in self.processed_data or 'intervention_effect' not in self.processed_data[model_name]:
                continue
                
            layers = model_info['layers']
            effect_matrix = self.processed_data[model_name]['intervention_effect']
            
            # Calculate average intervention effect for each target layer
            target_effect_avg = np.mean(effect_matrix, axis=0)
            
            ax2.plot(layers, target_effect_avg, marker='s', label=model_name, linewidth=2, markersize=6)
        
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Target Layer')
        ax2.set_ylabel('Intervention Effect')
        ax2.set_title('Intervention Effect per Target Layer\n(Experimental - Baseline)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path / "target_layer_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_source_layer_analysis(self, save_path="./analysis_results/"):
        """Generate analysis of source layer effectiveness."""
        print("\nGenerating source layer analysis...")
        
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Average effectiveness per source layer
        for model_name, model_info in self.models.items():
            if model_name not in self.processed_data or 'exp_avg' not in self.processed_data[model_name]:
                continue
                
            layers = model_info['layers']
            exp_avg = self.processed_data[model_name]['exp_avg']
            
            # Calculate average effectiveness for each source layer (average across target layers)
            source_layer_avg = np.mean(exp_avg, axis=1)
            source_layer_std = np.std(exp_avg, axis=1)
            
            ax1.plot(layers, source_layer_avg, marker='o', label=model_name, linewidth=2, markersize=6)
            ax1.fill_between(layers, 
                           source_layer_avg - source_layer_std/2, 
                           source_layer_avg + source_layer_std/2, 
                           alpha=0.2)
        
        ax1.set_xlabel('Source Layer')
        ax1.set_ylabel('Average Accuracy')
        ax1.set_title('Average Effectiveness per Source Layer\n(Averaged across all target layers)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Plot 2: Source layer intervention effect
        for model_name, model_info in self.models.items():
            if model_name not in self.processed_data or 'intervention_effect' not in self.processed_data[model_name]:
                continue
                
            layers = model_info['layers']
            effect_matrix = self.processed_data[model_name]['intervention_effect']
            
            # Calculate average intervention effect for each source layer
            source_effect_avg = np.mean(effect_matrix, axis=1)
            
            ax2.plot(layers, source_effect_avg, marker='s', label=model_name, linewidth=2, markersize=6)
        
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Source Layer')
        ax2.set_ylabel('Intervention Effect')
        ax2.set_title('Source Layer Intervention Effect\n(Experimental - Baseline)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path / "source_layer_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_layer_distance_analysis(self, save_path="./analysis_results/"):
        """Analyze performance vs. layer distance (|source - target|)."""
        print("\nGenerating layer distance analysis...")
        
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        for model_name, model_info in self.models.items():
            if model_name not in self.processed_data or 'exp_avg' not in self.processed_data[model_name]:
                continue
                
            layers = model_info['layers']
            exp_avg = self.processed_data[model_name]['exp_avg']
            
            # Calculate performance vs distance
            distances = []
            accuracies = []
            effects = []
            
            for i, source_layer in enumerate(layers):
                for j, target_layer in enumerate(layers):
                    distance = abs(source_layer - target_layer)
                    accuracy = exp_avg[i, j]
                    
                    distances.append(distance)
                    accuracies.append(accuracy)
                    
                    if 'intervention_effect' in self.processed_data[model_name]:
                        effect = self.processed_data[model_name]['intervention_effect'][i, j]
                        effects.append(effect)
            
            # Group by distance and calculate averages
            distance_acc = defaultdict(list)
            distance_eff = defaultdict(list)
            
            for d, a, e in zip(distances, accuracies, effects):
                distance_acc[d].append(a)
                distance_eff[d].append(e)
            
            # Calculate averages
            sorted_distances = sorted(distance_acc.keys())
            avg_accuracies = [np.mean(distance_acc[d]) for d in sorted_distances]
            avg_effects = [np.mean(distance_eff[d]) for d in sorted_distances]
            
            ax1.plot(sorted_distances, avg_accuracies, marker='o', label=model_name, linewidth=2, markersize=6)
            if effects:  # Only plot if we have effect data
                ax2.plot(sorted_distances, avg_effects, marker='s', label=model_name, linewidth=2, markersize=6)
        
        ax1.set_xlabel('Layer Distance |Source - Target|')
        ax1.set_ylabel('Average Accuracy')
        ax1.set_title('Performance vs. Layer Distance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Layer Distance |Source - Target|')
        ax2.set_ylabel('Intervention Effect')
        ax2.set_title('Intervention Effect vs. Layer Distance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path / "layer_distance_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_category_analysis(self, save_path="./analysis_results/"):
        """Generate category-specific performance breakdown."""
        print("\nGenerating category analysis...")
        
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True)
        
        # Collect category performance data
        category_performance = defaultdict(lambda: defaultdict(list))
        
        for model_name in self.models.keys():
            if model_name not in self.data or not self.data[model_name]['experimental']:
                continue
                
            for sample in self.data[model_name]['experimental']:
                category = sample['category']
                results_matrix = np.array(sample['results'])
                avg_performance = np.mean(results_matrix)
                category_performance[category][model_name].append(avg_performance)
        
        # Calculate averages per category
        categories = list(category_performance.keys())
        models_with_data = [m for m in self.models.keys() if m in self.data]
        
        if categories:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            x = np.arange(len(categories))
            width = 0.25
            
            for i, model_name in enumerate(models_with_data):
                if model_name not in self.data:
                    continue
                    
                avg_scores = []
                err_scores = []
                
                for category in categories:
                    scores = category_performance[category][model_name]
                    if scores:
                        avg_scores.append(np.mean(scores))
                        err_scores.append(np.std(scores) / np.sqrt(len(scores)))  # Standard error
                    else:
                        avg_scores.append(0)
                        err_scores.append(0)
                
                ax.bar(x + i*width, avg_scores, width, label=model_name, 
                      yerr=err_scores, capsize=5, alpha=0.8)
            
            ax.set_xlabel('Object Category')
            ax.set_ylabel('Average Accuracy')
            ax.set_title('Performance by Object Category')
            ax.set_xticks(x + width)
            ax.set_xticklabels(categories, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            
            plt.tight_layout()
            plt.savefig(save_path / "category_analysis.png", dpi=300, bbox_inches='tight')
            plt.show()
    
    def generate_summary_statistics(self, save_path="./analysis_results/"):
        """Generate and save summary statistics."""
        print("\nGenerating summary statistics...")
        
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True)
        
        summary = {}
        
        for model_name, model_info in self.models.items():
            if model_name not in self.processed_data:
                continue
                
            processed = self.processed_data[model_name]
            summary[model_name] = {}
            
            if 'exp_avg' in processed:
                exp_avg = processed['exp_avg']
                summary[model_name]['overall_accuracy'] = float(np.mean(exp_avg))
                summary[model_name]['accuracy_std'] = float(np.std(exp_avg))
                summary[model_name]['best_source_target'] = {
                    'accuracy': float(np.max(exp_avg)),
                    'position': [int(pos) for pos in np.unravel_index(np.argmax(exp_avg), exp_avg.shape)]
                }
                summary[model_name]['worst_source_target'] = {
                    'accuracy': float(np.min(exp_avg)),
                    'position': [int(pos) for pos in np.unravel_index(np.argmin(exp_avg), exp_avg.shape)]
                }
            
            if 'baseline_acc' in processed:
                summary[model_name]['baseline_accuracy'] = float(processed['baseline_acc'])
            
            if 'intervention_effect' in processed:
                effect = processed['intervention_effect']
                summary[model_name]['intervention_effect'] = {
                    'mean': float(np.mean(effect)),
                    'max': float(np.max(effect)),
                    'min': float(np.min(effect)),
                    'positive_positions': int(np.sum(effect > 0)),
                    'negative_positions': int(np.sum(effect < 0))
                }
            
            summary[model_name]['sample_count'] = processed.get('exp_samples', 0)
        
        # Save summary to JSON
        with open(save_path / "summary_statistics.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print("\n" + "="*50)
        print("SUMMARY STATISTICS")
        print("="*50)
        
        for model_name, stats in summary.items():
            print(f"\n{model_name}:")
            print(f"  Samples: {stats.get('sample_count', 'N/A')}")
            if 'overall_accuracy' in stats:
                print(f"  Overall Accuracy: {stats['overall_accuracy']:.3f} ± {stats['accuracy_std']:.3f}")
            if 'baseline_accuracy' in stats:
                print(f"  Baseline Accuracy: {stats['baseline_accuracy']:.3f}")
            if 'intervention_effect' in stats:
                effect = stats['intervention_effect']
                print(f"  Intervention Effect: {effect['mean']:.3f} (range: {effect['min']:.3f} to {effect['max']:.3f})")
                print(f"  Positive Effects: {effect['positive_positions']} positions")
                print(f"  Negative Effects: {effect['negative_positions']} positions")
    
    def run_complete_analysis(self, save_path="./analysis_results/"):
        """Run the complete analysis pipeline."""
        print("Starting comprehensive VLM intervention analysis...")
        print("="*60)
        
        # Create output directory
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True)
        
        # Load and preprocess data
        self.load_data()
        self.preprocess_data()
        
        # Generate all visualizations
        self.generate_enhanced_heatmaps(save_path)
        self.generate_target_layer_analysis(save_path)
        self.generate_source_layer_analysis(save_path)
        self.generate_layer_distance_analysis(save_path)
        self.generate_category_analysis(save_path)
        self.generate_summary_statistics(save_path)
        
        print(f"\nAnalysis complete! Results saved to: {save_path.absolute()}")


def main():
    """Main function to run the analysis."""
    # Initialize analyzer
    analyzer = VLMAnalyzer(results_dir="./lg_results/")
    
    # Run complete analysis
    analyzer.run_complete_analysis(save_path="./analysis_results/")


if __name__ == "__main__":
    main()


