#!/usr/bin/env python3
"""
Target Layer Analysis Script for Text-Patching Layer Intervention Experiments

This script creates a line chart showing the average injection accuracy per target layer
for vision-language models (QWEN2-VL and DeepSeek-VL2) in text-patching experiments.

Based on the comprehensive_analysis.py structure but focused only on target layer analysis.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up matplotlib for better visualization
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['figure.dpi'] = 100

class TextPatchingAnalyzer:
    """Analyzer for text-patching vision-language model intervention experiments."""
    
    def __init__(self, results_dir="./results/"):
        """
        Initialize the analyzer with results directory.
        
        Args:
            results_dir (str): Path to directory containing experiment results
        """
        self.results_dir = Path(results_dir)
        
        # Define models and their configurations based on the data files
        self.models = {
            'QWEN2-VL': {
                'file': 'qwen_results.json',
                'layers': None  # Will be determined from data
            },
            'DeepSeek-VL2': {
                'file': 'ds-results.json', 
                'layers': None  # Will be determined from data
            }
        }
        self.data = {}
        self.processed_data = {}
        
    def load_data(self):
        """Load experimental data for all models."""
        print("Loading experimental data...")
        
        for model_name, model_info in self.models.items():
            print(f"  Loading {model_name}...")
            
            # Load experimental results
            data_path = self.results_dir / model_info['file']
            if data_path.exists():
                with open(data_path, 'r') as f:
                    self.data[model_name] = json.load(f)
                print(f"    Loaded: {len(self.data[model_name])} samples")
                
                # Determine layer configuration from first sample
                if self.data[model_name]:
                    first_sample = self.data[model_name][0]
                    matrix_size = len(first_sample['results'])
                    self.models[model_name]['layers'] = list(range(matrix_size))
                    print(f"    Matrix size: {matrix_size}x{matrix_size}")
            else:
                print(f"    Warning: {data_path} not found")
                self.data[model_name] = []
    
    def preprocess_data(self):
        """Preprocess data to calculate average matrices and statistics."""
        print("\nPreprocessing data...")
        
        for model_name in self.models.keys():
            if model_name not in self.data or not self.data[model_name]:
                continue
                
            print(f"  Processing {model_name}...")
            self.processed_data[model_name] = {}
            
            # Extract all result matrices
            matrices = []
            for sample in self.data[model_name]:
                matrix = np.array(sample['results'])
                matrices.append(matrix)
            
            if matrices:
                # Calculate average matrix across all samples
                avg_matrix = np.mean(matrices, axis=0)
                std_matrix = np.std(matrices, axis=0)
                
                self.processed_data[model_name]['avg_matrix'] = avg_matrix
                self.processed_data[model_name]['std_matrix'] = std_matrix
                self.processed_data[model_name]['sample_count'] = len(matrices)
                
                print(f"    Average accuracy: {np.mean(avg_matrix):.3f}")
                print(f"    Samples processed: {len(matrices)}")
    
    def generate_target_layer_line_chart(self, save_path="./"):
        """Generate line chart for target layer performance analysis."""
        print("\nGenerating target layer analysis line chart...")
        
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True)
        
        # Create the plot with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Plot data for each model
        for model_name, model_info in self.models.items():
            if model_name not in self.processed_data:
                continue
                
            layers = model_info['layers']
            if layers is None:
                continue
                
            avg_matrix = self.processed_data[model_name]['avg_matrix']
            std_matrix = self.processed_data[model_name]['std_matrix']
            
            # Plot 1: Average accuracy per target layer
            target_layer_avg = np.mean(avg_matrix, axis=0)
            target_layer_std = np.std(avg_matrix, axis=0)
            
            ax1.plot(layers, target_layer_avg, marker='o', label=model_name, 
                   linewidth=2.5, markersize=8)
            ax1.fill_between(layers, 
                           target_layer_avg - target_layer_std/2, 
                           target_layer_avg + target_layer_std/2, 
                           alpha=0.2)
            
            # Plot 2: Best performing source layer for each target
            best_source_performance = np.max(avg_matrix, axis=0)
            
            ax2.plot(layers, best_source_performance, marker='s', label=model_name, 
                   linewidth=2.5, markersize=8)
            
            # Print some statistics
            print(f"  {model_name}:")
            print(f"    Best target layer (avg): {layers[np.argmax(target_layer_avg)]} "
                  f"(accuracy: {np.max(target_layer_avg):.3f})")
            print(f"    Best target layer (max): {layers[np.argmax(best_source_performance)]} "
                  f"(accuracy: {np.max(best_source_performance):.3f})")
        
        # Customize Plot 1
        ax1.set_xlabel('Target Layer Index')
        ax1.set_ylabel('Average Injection Accuracy')
        ax1.set_title('Average Performance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Customize Plot 2
        ax2.set_xlabel('Target Layer Index')
        ax2.set_ylabel('Best Source Performance')
        ax2.set_title('Best Performance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Set integer ticks for x-axis
        if self.processed_data:
            # Get the maximum number of layers across all models
            max_layers = max(len(info['layers']) for info in self.models.values() 
                           if info['layers'] is not None)
            ax1.set_xticks(range(max_layers))
            ax2.set_xticks(range(max_layers))
        
        plt.tight_layout()
        
        # Save the plot
        output_file = save_path / "target_layer_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved plot to: {output_file}")
        
        # Show the plot
        plt.show()
        
        return fig, (ax1, ax2)
    
    def print_summary_statistics(self):
        """Print summary statistics for the analysis."""
        print("\n" + "="*50)
        print("SUMMARY STATISTICS")
        print("="*50)
        
        for model_name in self.models.keys():
            if model_name not in self.processed_data:
                continue
                
            processed = self.processed_data[model_name]
            avg_matrix = processed['avg_matrix']
            
            print(f"\n{model_name}:")
            print(f"  Samples: {processed['sample_count']}")
            print(f"  Matrix size: {avg_matrix.shape[0]}x{avg_matrix.shape[1]}")
            print(f"  Overall average accuracy: {np.mean(avg_matrix):.3f} ± {np.std(avg_matrix):.3f}")
            print(f"  Best accuracy: {np.max(avg_matrix):.3f}")
            print(f"  Worst accuracy: {np.min(avg_matrix):.3f}")
            
            # Target layer statistics
            target_layer_avg = np.mean(avg_matrix, axis=0)
            best_target = np.argmax(target_layer_avg)
            worst_target = np.argmin(target_layer_avg)
            
            print(f"  Best target layer: {best_target} (avg accuracy: {target_layer_avg[best_target]:.3f})")
            print(f"  Worst target layer: {worst_target} (avg accuracy: {target_layer_avg[worst_target]:.3f})")
    
    def run_analysis(self, save_path="./"):
        """Run the complete target layer analysis."""
        print("Starting text-patching target layer analysis...")
        print("="*60)
        
        # Load and preprocess data
        self.load_data()
        self.preprocess_data()
        
        # Generate the line chart
        self.generate_target_layer_line_chart(save_path)
        
        # Print summary statistics
        self.print_summary_statistics()
        
        print(f"\nAnalysis complete! Results saved to: {Path(save_path).absolute()}")


def main():
    """Main function to run the analysis."""
    # Initialize analyzer with results directory
    analyzer = TextPatchingAnalyzer(results_dir="./results/")
    
    # Run the analysis
    analyzer.run_analysis(save_path="./")


if __name__ == "__main__":
    main()
