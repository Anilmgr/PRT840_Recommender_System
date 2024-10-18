"""
Recommender System Visualization
==============================

This script generates comparative visualizations between two recommender system
implementations using various performance metrics. It creates bar charts comparing
the performance metrics of a basic recommender system versus a robust recommender
system under different attack scenarios.

Dependencies:
    - pandas
    - matplotlib.pyplot
"""

import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
from pathlib import Path


class RecommenderVisualizer:
    """Handles visualization of recommender system comparison results."""
    
    # Color schemes for different metrics
    COLOR_SCHEMES: Dict[str, tuple] = {
        'RMSE': ('#FF9999', '#FF3333'),
        'MAE': ('#99FF99', '#33FF33'),
        'Hit Rate@10': ('#9999FF', '#3333FF'),
        'ARHR@10': ('#F3F781', '#D7DF01'),
        'MAP@10': ('#FF99FF', '#FF33FF'),
        'NDCG@10': ('#99FFFF', '#33FFFF')
    }
    
    def __init__(self, basic_results_path: str, robust_results_path: str):
        """
        Initialize the visualizer with paths to result files.
        
        Args:
            basic_results_path: Path to basic recommender system results
            robust_results_path: Path to robust recommender system results
        """
        self.basic_results = self._load_results(basic_results_path)
        self.robust_results = self._load_results(robust_results_path)
        self.metrics = ['RMSE', 'MAE', 'Hit Rate@10', 'ARHR@10', 'MAP@10', 'NDCG@10']
    
    @staticmethod
    def _load_results(file_path: str) -> pd.DataFrame:
        """
        Load results from a CSV file.
        
        Args:
            file_path: Path to the results CSV file
            
        Returns:
            DataFrame containing the results
            
        Raises:
            FileNotFoundError: If the specified file doesn't exist
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Results file not found: {file_path}")
        
        return pd.read_csv(file_path)
    
    def plot_metric_comparison(self, metric: str) -> None:
        """
        Create a comparative bar plot for a specific metric.
        
        Args:
            metric: Name of the metric to plot
        """
        plt.figure(figsize=(12, 6))
        
        # Set up bar positions
        x = range(len(self.basic_results))
        width = 0.35
        
        # Create bars for both systems
        plt.bar(
            [i - width/2 for i in x],
            self.basic_results[metric],
            width,
            label='Basic Recommender',
            color=self.COLOR_SCHEMES[metric][0],
            alpha=0.8
        )
        
        plt.bar(
            [i + width/2 for i in x],
            self.robust_results[metric],
            width,
            label='Robust Recommender',
            color=self.COLOR_SCHEMES[metric][1],
            alpha=0.8
        )
        
        # Customize plot
        plt.xlabel('Model - Scenario')
        plt.ylabel(metric)
        plt.title(f'Comparison of {metric} between Recommender Systems')
        
        # Set x-axis labels
        plt.xticks(
            x,
            self.basic_results['Model'] + ' - ' + self.basic_results['Scenario'],
            rotation=45,
            ha='right'
        )
        
        plt.legend()
        plt.tight_layout()
        
        # Optional: Add grid for better readability
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        plt.show()
    
    def plot_all_metrics(self) -> None:
        """Plot comparisons for all available metrics."""
        for metric in self.metrics:
            print(f"Generating plot for {metric}...")
            self.plot_metric_comparison(metric)
    
    def generate_summary_statistics(self) -> pd.DataFrame:
        """
        Generate summary statistics comparing both systems.
        
        Returns:
            DataFrame containing summary statistics
        """
        summary_stats = []
        
        for metric in self.metrics:
            stats = {
                'Metric': metric,
                'Basic_Mean': self.basic_results[metric].mean(),
                'Basic_Std': self.basic_results[metric].std(),
                'Robust_Mean': self.robust_results[metric].mean(),
                'Robust_Std': self.robust_results[metric].std(),
                'Improvement': (
                    (self.robust_results[metric].mean() - self.basic_results[metric].mean())
                    / self.basic_results[metric].mean() * 100
                )
            }
            summary_stats.append(stats)
        
        return pd.DataFrame(summary_stats)


def main():
    """Main execution function."""
    try:
        # Initialize visualizer
        visualizer = RecommenderVisualizer(
            'recommender_system_results.csv',
            'robust_recommender_system_results.csv'
        )
        
        # Generate all plots
        visualizer.plot_all_metrics()
        
        # Generate and display summary statistics
        summary_stats = visualizer.generate_summary_statistics()
        print("\nSummary Statistics:")
        print(summary_stats.round(4).to_string(index=False))
        
        # Save summary statistics
        summary_stats.to_csv('recommender_comparison_summary.csv', index=False)
        print("\nSummary statistics saved to 'recommender_comparison_summary.csv'")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure both result files exist in the current directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()