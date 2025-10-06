#!/usr/bin/env python3
"""
Analyze Classification Results

This script analyzes the results from the QSAR classification pipeline and creates
visualizations and detailed statistics for the classification performance.

Date: 2025-01-30
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_esm_emb_classification/analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ClassificationResultsAnalyzer:
    """Analyzer for QSAR classification results"""
    
    def __init__(self, results_path: str = "/home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_esm_emb_classification/esm_classification_results.csv"):
        """
        Initialize the analyzer
        
        Args:
            results_path: Path to the classification results CSV file
        """
        self.results_path = results_path
        self.results_df = None
        self.successful_results = None
        
    def load_results(self):
        """Load classification results"""
        try:
            self.results_df = pd.read_csv(self.results_path)
            logger.info(f"Loaded {len(self.results_df)} results from {self.results_path}")
            
            # Filter successful results
            self.successful_results = self.results_df[self.results_df['status'] == 'success'].copy()
            logger.info(f"Found {len(self.successful_results)} successful models")
            
        except Exception as e:
            logger.error(f"Error loading results: {e}")
            raise
    
    def print_summary_statistics(self):
        """Print summary statistics for the classification results"""
        if self.successful_results is None or len(self.successful_results) == 0:
            logger.warning("No successful results to analyze")
            return
        
        logger.info("\n" + "="*60)
        logger.info("CLASSIFICATION RESULTS SUMMARY")
        logger.info("="*60)
        
        # Basic statistics
        logger.info(f"Total proteins processed: {len(self.results_df)}")
        logger.info(f"Successful models: {len(self.successful_results)}")
        logger.info(f"Failed models: {len(self.results_df) - len(self.successful_results)}")
        
        # Performance metrics
        metrics = ['avg_accuracy', 'avg_f1', 'avg_auc', 'avg_precision', 'avg_recall']
        
        logger.info("\nPerformance Metrics (averaged across all successful models):")
        logger.info("-" * 50)
        
        for metric in metrics:
            if metric in self.successful_results.columns:
                values = self.successful_results[metric].dropna()
                if len(values) > 0:
                    mean_val = values.mean()
                    std_val = values.std()
                    min_val = values.min()
                    max_val = values.max()
                    
                    logger.info(f"{metric.replace('avg_', '').upper():<15}: "
                              f"{mean_val:.3f} ± {std_val:.3f} (range: {min_val:.3f}-{max_val:.3f})")
        
        # Sample size statistics
        logger.info(f"\nSample Size Statistics:")
        logger.info("-" * 30)
        logger.info(f"Total samples: {self.successful_results['n_samples'].sum()}")
        logger.info(f"Average samples per protein: {self.successful_results['n_samples'].mean():.1f}")
        logger.info(f"Min samples: {self.successful_results['n_samples'].min()}")
        logger.info(f"Max samples: {self.successful_results['n_samples'].max()}")
        
        # Top performing proteins
        logger.info(f"\nTop 10 Proteins by F1 Score:")
        logger.info("-" * 40)
        top_proteins = self.successful_results.nlargest(10, 'avg_f1')[['protein', 'avg_f1', 'avg_accuracy', 'n_samples']]
        for _, row in top_proteins.iterrows():
            logger.info(f"{row['protein']:<20}: F1={row['avg_f1']:.3f}, "
                       f"Accuracy={row['avg_accuracy']:.3f}, Samples={row['n_samples']}")
    
    def create_performance_plots(self, output_dir: str = "/home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_esm_emb_classification/plots"):
        """Create performance visualization plots"""
        if self.successful_results is None or len(self.successful_results) == 0:
            logger.warning("No successful results to plot")
            return
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Performance metrics distribution
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('QSAR Classification Performance Metrics Distribution', fontsize=16)
        
        metrics = ['avg_accuracy', 'avg_f1', 'avg_auc', 'avg_precision', 'avg_recall']
        titles = ['Accuracy', 'F1 Score', 'AUC', 'Precision', 'Recall']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            row = i // 3
            col = i % 3
            
            if metric in self.successful_results.columns:
                values = self.successful_results[metric].dropna()
                if len(values) > 0:
                    axes[row, col].hist(values, bins=20, alpha=0.7, edgecolor='black')
                    axes[row, col].axvline(values.mean(), color='red', linestyle='--', 
                                         label=f'Mean: {values.mean():.3f}')
                    axes[row, col].set_title(title)
                    axes[row, col].set_xlabel(metric.replace('avg_', ''))
                    axes[row, col].set_ylabel('Frequency')
                    axes[row, col].legend()
                    axes[row, col].grid(True, alpha=0.3)
        
        # Remove empty subplot
        axes[1, 2].remove()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/performance_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Performance vs Sample Size
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Performance vs Sample Size', fontsize=16)
        
        metrics = ['avg_accuracy', 'avg_f1', 'avg_auc', 'avg_precision']
        titles = ['Accuracy', 'F1 Score', 'AUC', 'Precision']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            row = i // 2
            col = i % 2
            
            if metric in self.successful_results.columns:
                values = self.successful_results[metric].dropna()
                sample_sizes = self.successful_results.loc[values.index, 'n_samples']
                
                axes[row, col].scatter(sample_sizes, values, alpha=0.6, s=50)
                axes[row, col].set_xlabel('Number of Samples')
                axes[row, col].set_ylabel(title)
                axes[row, col].set_title(f'{title} vs Sample Size')
                axes[row, col].grid(True, alpha=0.3)
                
                # Add trend line
                if len(values) > 1:
                    z = np.polyfit(sample_sizes, values, 1)
                    p = np.poly1d(z)
                    axes[row, col].plot(sample_sizes, p(sample_sizes), "r--", alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/performance_vs_samples.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Top performing proteins
        top_n = min(15, len(self.successful_results))
        top_proteins = self.successful_results.nlargest(top_n, 'avg_f1')
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        y_pos = np.arange(len(top_proteins))
        bars = ax.barh(y_pos, top_proteins['avg_f1'], alpha=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_proteins['protein'])
        ax.set_xlabel('F1 Score')
        ax.set_title(f'Top {top_n} Proteins by F1 Score')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, f1_score) in enumerate(zip(bars, top_proteins['avg_f1'])):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{f1_score:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/top_proteins_f1.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Correlation matrix of metrics
        metric_cols = ['avg_accuracy', 'avg_f1', 'avg_auc', 'avg_precision', 'avg_recall', 'n_samples']
        available_cols = [col for col in metric_cols if col in self.successful_results.columns]
        
        if len(available_cols) > 1:
            corr_matrix = self.successful_results[available_cols].corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, ax=ax)
            ax.set_title('Correlation Matrix of Performance Metrics')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/correlation_matrix.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Plots saved to {output_dir}")
    
    def create_detailed_report(self, output_path: str = "/home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_esm_emb_classification/detailed_report.txt"):
        """Create a detailed text report of the results"""
        if self.successful_results is None or len(self.successful_results) == 0:
            logger.warning("No successful results to report")
            return
        
        with open(output_path, 'w') as f:
            f.write("QSAR CLASSIFICATION RESULTS - DETAILED REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Summary statistics
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total proteins processed: {len(self.results_df)}\n")
            f.write(f"Successful models: {len(self.successful_results)}\n")
            f.write(f"Failed models: {len(self.results_df) - len(self.successful_results)}\n")
            f.write(f"Success rate: {len(self.successful_results)/len(self.results_df)*100:.1f}%\n\n")
            
            # Performance metrics
            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 20 + "\n")
            
            metrics = ['avg_accuracy', 'avg_f1', 'avg_auc', 'avg_precision', 'avg_recall']
            for metric in metrics:
                if metric in self.successful_results.columns:
                    values = self.successful_results[metric].dropna()
                    if len(values) > 0:
                        f.write(f"{metric.replace('avg_', '').upper()}:\n")
                        f.write(f"  Mean: {values.mean():.3f}\n")
                        f.write(f"  Std:  {values.std():.3f}\n")
                        f.write(f"  Min:  {values.min():.3f}\n")
                        f.write(f"  Max:  {values.max():.3f}\n")
                        f.write(f"  Median: {values.median():.3f}\n\n")
            
            # Sample size statistics
            f.write("SAMPLE SIZE STATISTICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total samples: {self.successful_results['n_samples'].sum()}\n")
            f.write(f"Average samples per protein: {self.successful_results['n_samples'].mean():.1f}\n")
            f.write(f"Min samples: {self.successful_results['n_samples'].min()}\n")
            f.write(f"Max samples: {self.successful_results['n_samples'].max()}\n\n")
            
            # Top performing proteins
            f.write("TOP 10 PROTEINS BY F1 SCORE\n")
            f.write("-" * 30 + "\n")
            top_proteins = self.successful_results.nlargest(10, 'avg_f1')
            for i, (_, row) in enumerate(top_proteins.iterrows(), 1):
                f.write(f"{i:2d}. {row['protein']:<20}: F1={row['avg_f1']:.3f}, "
                       f"Accuracy={row['avg_accuracy']:.3f}, Samples={row['n_samples']}\n")
            
            f.write("\n")
            
            # Worst performing proteins
            f.write("BOTTOM 10 PROTEINS BY F1 SCORE\n")
            f.write("-" * 30 + "\n")
            bottom_proteins = self.successful_results.nsmallest(10, 'avg_f1')
            for i, (_, row) in enumerate(bottom_proteins.iterrows(), 1):
                f.write(f"{i:2d}. {row['protein']:<20}: F1={row['avg_f1']:.3f}, "
                       f"Accuracy={row['avg_accuracy']:.3f}, Samples={row['n_samples']}\n")
            
            f.write("\n")
            
            # Failed models
            failed_models = self.results_df[self.results_df['status'] != 'success']
            if len(failed_models) > 0:
                f.write("FAILED MODELS\n")
                f.write("-" * 15 + "\n")
                for _, row in failed_models.iterrows():
                    f.write(f"{row['protein']}: {row['status']}\n")
                    if 'error' in row and pd.notna(row['error']):
                        f.write(f"  Error: {row['error']}\n")
        
        logger.info(f"Detailed report saved to {output_path}")
    
    def analyze_results(self):
        """Run complete analysis of classification results"""
        logger.info("Starting analysis of classification results...")
        
        # Load results
        self.load_results()
        
        # Print summary statistics
        self.print_summary_statistics()
        
        # Create plots
        self.create_performance_plots()
        
        # Create detailed report
        self.create_detailed_report()
        
        logger.info("Analysis completed!")

def main():
    """Main function to run the analysis"""
    analyzer = ClassificationResultsAnalyzer()
    analyzer.analyze_results()

if __name__ == "__main__":
    main() 