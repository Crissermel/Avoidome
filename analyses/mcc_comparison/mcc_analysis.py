#!/usr/bin/env python3
"""
Matthews Correlation Coefficient (MCC) Analysis for QSAR Model Comparison

This script performs pairwise comparisons between regression and classification models
using MCC as the primary metric. It compares:
1. Morgan regression vs Morgan classification
2. ESM+Morgan regression vs ESM+Morgan classification

For regression models, predictions are binarized using a threshold (default: 7.0)
to enable MCC calculation and comparison with classification models.

Author: Generated for Avoidome QSAR modeling
Date: 2025-01-30
"""

import pandas as pd
import numpy as np
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.metrics import matthews_corrcoef, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/serramelendezcsm/RA/Avoidome/analyses/mcc_comparison/mcc_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MCCAnalysis:
    """
    MCC analysis for comparing regression and classification QSAR models
    """
    
    def __init__(self, 
                 models_dir: str = "/home/serramelendezcsm/RA/Avoidome/analyses/standardized_qsar_models",
                 output_dir: str = "/home/serramelendezcsm/RA/Avoidome/analyses/mcc_comparison",
                 threshold: float = 7.0):
        """
        Initialize MCC analysis
        
        Args:
            models_dir: Directory containing the standardized QSAR models
            output_dir: Directory to save analysis results
            threshold: Threshold for binarizing regression predictions
        """
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.threshold = threshold
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "results").mkdir(exist_ok=True)
        
        # Model directories
        self.morgan_regression_dir = self.models_dir / "morgan_regression"
        self.morgan_classification_dir = self.models_dir / "morgan_classification"
        self.esm_morgan_regression_dir = self.models_dir / "esm_morgan_regression"
        self.esm_morgan_classification_dir = self.models_dir / "esm_morgan_classification"
        
        # Results storage
        self.results = {
            'morgan_comparison': {},
            'esm_morgan_comparison': {},
            'summary': {}
        }
        
    def load_model_predictions(self, model_path: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Load model predictions and true values from a model directory
        
        Args:
            model_path: Path to the model directory
            
        Returns:
            Tuple of (true_values, predictions, metadata)
        """
        try:
            # Load predictions
            predictions_file = model_path / "predictions.csv"
            if not predictions_file.exists():
                logger.warning(f"Predictions file not found: {predictions_file}")
                return None, None, None
                
            pred_df = pd.read_csv(predictions_file)
            
            # Load results metadata
            results_file = model_path / "results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {}
            
            # Extract true values and predictions
            if 'true_values' in pred_df.columns and 'predictions' in pred_df.columns:
                true_values = pred_df['true_values'].values
                predictions = pred_df['predictions'].values
            else:
                logger.warning(f"Required columns not found in {predictions_file}")
                return None, None, None
                
            return true_values, predictions, metadata
            
        except Exception as e:
            logger.error(f"Error loading predictions from {model_path}: {e}")
            return None, None, None
    
    def binarize_predictions(self, true_values: np.ndarray, predictions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Binarize regression predictions and true values using threshold
        
        Args:
            true_values: True continuous values
            predictions: Predicted continuous values
            
        Returns:
            Tuple of (binarized_true, binarized_predictions)
        """
        binarized_true = (true_values >= self.threshold).astype(int)
        binarized_pred = (predictions >= self.threshold).astype(int)
        return binarized_true, binarized_pred
    
    def calculate_mcc_metrics(self, true_values: np.ndarray, predictions: np.ndarray) -> Dict:
        """
        Calculate MCC and related metrics
        
        Args:
            true_values: True binary values
            predictions: Predicted binary values
            
        Returns:
            Dictionary of calculated metrics
        """
        try:
            mcc = matthews_corrcoef(true_values, predictions)
            accuracy = accuracy_score(true_values, predictions)
            precision = precision_score(true_values, predictions, zero_division=0)
            recall = recall_score(true_values, predictions, zero_division=0)
            f1 = f1_score(true_values, predictions, zero_division=0)
            
            # Calculate confusion matrix
            cm = confusion_matrix(true_values, predictions)
            tn, fp, fn, tp = cm.ravel()
            
            # Calculate specificity
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            return {
                'mcc': mcc,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'specificity': specificity,
                'confusion_matrix': cm.tolist(),
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp)
            }
            
        except Exception as e:
            logger.error(f"Error calculating MCC metrics: {e}")
            return None
    
    def compare_models(self, protein_name: str, organism: str, 
                      regression_dir: Path, classification_dir: Path) -> Optional[Dict]:
        """
        Compare regression and classification models for a specific protein-organism combination
        
        Args:
            protein_name: Name of the protein
            organism: Organism (human, mouse, rat)
            regression_dir: Directory containing regression models
            classification_dir: Directory containing classification models
            
        Returns:
            Dictionary with comparison results or None if comparison failed
        """
        # Find protein directories - look for directories starting with protein_name
        protein_dirs = list(regression_dir.glob(f"{protein_name}_*"))
        if not protein_dirs:
            logger.warning(f"No regression model found for {protein_name} in {organism}")
            return None
            
        protein_dir = protein_dirs[0]
        regression_model_path = protein_dir
        
        # Find corresponding classification model
        classification_protein_dirs = list(classification_dir.glob(f"{protein_name}_*"))
        if not classification_protein_dirs:
            logger.warning(f"No classification model found for {protein_name} in {organism}")
            return None
            
        classification_protein_dir = classification_protein_dirs[0]
        classification_model_path = classification_protein_dir
        
        # Load regression model predictions
        reg_true, reg_pred, reg_metadata = self.load_model_predictions(regression_model_path)
        if reg_true is None:
            return None
            
        # Load classification model predictions
        class_true, class_pred, class_metadata = self.load_model_predictions(classification_model_path)
        if class_true is None:
            return None
        
        # Binarize regression predictions
        reg_true_bin, reg_pred_bin = self.binarize_predictions(reg_true, reg_pred)
        
        # Calculate metrics for both models
        reg_metrics = self.calculate_mcc_metrics(reg_true_bin, reg_pred_bin)
        class_metrics = self.calculate_mcc_metrics(class_true, class_pred)
        
        if reg_metrics is None or class_metrics is None:
            return None
        
        # Calculate difference metrics
        mcc_diff = class_metrics['mcc'] - reg_metrics['mcc']
        accuracy_diff = class_metrics['accuracy'] - reg_metrics['accuracy']
        f1_diff = class_metrics['f1'] - reg_metrics['f1']
        
        return {
            'protein_name': protein_name,
            'organism': organism,
            'n_samples': len(reg_true),
            'regression_metrics': reg_metrics,
            'classification_metrics': class_metrics,
            'mcc_difference': mcc_diff,
            'accuracy_difference': accuracy_diff,
            'f1_difference': f1_diff,
            'better_model': 'classification' if mcc_diff > 0 else 'regression',
            'mcc_improvement': abs(mcc_diff)
        }
    
    def run_morgan_comparison(self) -> Dict:
        """
        Run comparison between Morgan regression and classification models
        
        Returns:
            Dictionary with comparison results
        """
        logger.info("Starting Morgan regression vs classification comparison")
        
        results = []
        organisms = ['human', 'mouse', 'rat']
        
        for organism in organisms:
            logger.info(f"Processing {organism} models")
            
            regression_dir = self.morgan_regression_dir / organism
            classification_dir = self.morgan_classification_dir / organism
            
            if not regression_dir.exists() or not classification_dir.exists():
                logger.warning(f"Model directories not found for {organism}")
                continue
            
            # Get all protein directories
            protein_dirs = list(regression_dir.glob("*"))
            
            for protein_dir in protein_dirs:
                # Extract protein name from directory name (e.g., CYP1A2_P05177 -> CYP1A2)
                protein_name = protein_dir.name.split('_')[0]
                
                comparison = self.compare_models(
                    protein_name, organism, 
                    regression_dir, 
                    classification_dir
                )
                
                if comparison:
                    results.append(comparison)
                    logger.info(f"Completed comparison for {protein_name} ({organism})")
        
        # Calculate summary statistics
        if results:
            mcc_diffs = [r['mcc_difference'] for r in results]
            accuracy_diffs = [r['accuracy_difference'] for r in results]
            f1_diffs = [r['f1_difference'] for r in results]
            
            classification_wins = sum(1 for r in results if r['better_model'] == 'classification')
            regression_wins = sum(1 for r in results if r['better_model'] == 'regression')
            
            summary = {
                'total_comparisons': len(results),
                'classification_wins': classification_wins,
                'regression_wins': regression_wins,
                'classification_win_rate': classification_wins / len(results),
                'mean_mcc_difference': np.mean(mcc_diffs),
                'std_mcc_difference': np.std(mcc_diffs),
                'mean_accuracy_difference': np.mean(accuracy_diffs),
                'mean_f1_difference': np.mean(f1_diffs),
                'results': results
            }
        else:
            summary = {'total_comparisons': 0, 'results': []}
        
        self.results['morgan_comparison'] = summary
        logger.info(f"Morgan comparison completed: {summary['total_comparisons']} comparisons")
        
        return summary
    
    def run_esm_morgan_comparison(self) -> Dict:
        """
        Run comparison between ESM+Morgan regression and classification models
        
        Returns:
            Dictionary with comparison results
        """
        logger.info("Starting ESM+Morgan regression vs classification comparison")
        
        results = []
        organisms = ['human', 'mouse', 'rat']
        
        for organism in organisms:
            logger.info(f"Processing {organism} models")
            
            regression_dir = self.esm_morgan_regression_dir / organism
            classification_dir = self.esm_morgan_classification_dir / organism
            
            if not regression_dir.exists() or not classification_dir.exists():
                logger.warning(f"Model directories not found for {organism}")
                continue
            
            # Get all protein directories
            protein_dirs = list(regression_dir.glob("*"))
            
            for protein_dir in protein_dirs:
                # Extract protein name from directory name (e.g., CYP1A2_P05177 -> CYP1A2)
                protein_name = protein_dir.name.split('_')[0]
                
                comparison = self.compare_models(
                    protein_name, organism, 
                    regression_dir, 
                    classification_dir
                )
                
                if comparison:
                    results.append(comparison)
                    logger.info(f"Completed comparison for {protein_name} ({organism})")
        
        # Calculate summary statistics
        if results:
            mcc_diffs = [r['mcc_difference'] for r in results]
            accuracy_diffs = [r['accuracy_difference'] for r in results]
            f1_diffs = [r['f1_difference'] for r in results]
            
            classification_wins = sum(1 for r in results if r['better_model'] == 'classification')
            regression_wins = sum(1 for r in results if r['better_model'] == 'regression')
            
            summary = {
                'total_comparisons': len(results),
                'classification_wins': classification_wins,
                'regression_wins': regression_wins,
                'classification_win_rate': classification_wins / len(results),
                'mean_mcc_difference': np.mean(mcc_diffs),
                'std_mcc_difference': np.std(mcc_diffs),
                'mean_accuracy_difference': np.mean(accuracy_diffs),
                'mean_f1_difference': np.mean(f1_diffs),
                'results': results
            }
        else:
            summary = {'total_comparisons': 0, 'results': []}
        
        self.results['esm_morgan_comparison'] = summary
        logger.info(f"ESM+Morgan comparison completed: {summary['total_comparisons']} comparisons")
        
        return summary
    
    def create_visualizations(self):
        """
        Create visualizations for the MCC analysis
        """
        logger.info("Creating visualizations")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. MCC difference distributions
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Morgan comparison
        morgan_results = self.results['morgan_comparison'].get('results', [])
        if morgan_results:
            mcc_diffs = [r['mcc_difference'] for r in morgan_results]
            accuracy_diffs = [r['accuracy_difference'] for r in morgan_results]
            
            axes[0, 0].hist(mcc_diffs, bins=20, alpha=0.7, color='blue', edgecolor='black')
            axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.7)
            axes[0, 0].set_title('Morgan Models: MCC Difference Distribution')
            axes[0, 0].set_xlabel('MCC Difference (Classification - Regression)')
            axes[0, 0].set_ylabel('Frequency')
            
            axes[0, 1].hist(accuracy_diffs, bins=20, alpha=0.7, color='green', edgecolor='black')
            axes[0, 1].axvline(0, color='red', linestyle='--', alpha=0.7)
            axes[0, 1].set_title('Morgan Models: Accuracy Difference Distribution')
            axes[0, 1].set_xlabel('Accuracy Difference (Classification - Regression)')
            axes[0, 1].set_ylabel('Frequency')
        
        # ESM+Morgan comparison
        esm_results = self.results['esm_morgan_comparison'].get('results', [])
        if esm_results:
            mcc_diffs = [r['mcc_difference'] for r in esm_results]
            accuracy_diffs = [r['accuracy_difference'] for r in esm_results]
            
            axes[1, 0].hist(mcc_diffs, bins=20, alpha=0.7, color='orange', edgecolor='black')
            axes[1, 0].axvline(0, color='red', linestyle='--', alpha=0.7)
            axes[1, 0].set_title('ESM+Morgan Models: MCC Difference Distribution')
            axes[1, 0].set_xlabel('MCC Difference (Classification - Regression)')
            axes[1, 0].set_ylabel('Frequency')
            
            axes[1, 1].hist(accuracy_diffs, bins=20, alpha=0.7, color='purple', edgecolor='black')
            axes[1, 1].axvline(0, color='red', linestyle='--', alpha=0.7)
            axes[1, 1].set_title('ESM+Morgan Models: Accuracy Difference Distribution')
            axes[1, 1].set_xlabel('Accuracy Difference (Classification - Regression)')
            axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "mcc_difference_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Model performance comparison scatter plots
        if morgan_results and esm_results:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Morgan models
            morgan_reg_mcc = [r['regression_metrics']['mcc'] for r in morgan_results]
            morgan_class_mcc = [r['classification_metrics']['mcc'] for r in morgan_results]
            
            axes[0].scatter(morgan_reg_mcc, morgan_class_mcc, alpha=0.7, s=50)
            axes[0].plot([-1, 1], [-1, 1], 'r--', alpha=0.5)
            axes[0].set_xlabel('Regression MCC')
            axes[0].set_ylabel('Classification MCC')
            axes[0].set_title('Morgan Models: Regression vs Classification MCC')
            axes[0].grid(True, alpha=0.3)
            
            # ESM+Morgan models
            esm_reg_mcc = [r['regression_metrics']['mcc'] for r in esm_results]
            esm_class_mcc = [r['classification_metrics']['mcc'] for r in esm_results]
            
            axes[1].scatter(esm_reg_mcc, esm_class_mcc, alpha=0.7, s=50, color='orange')
            axes[1].plot([-1, 1], [-1, 1], 'r--', alpha=0.5)
            axes[1].set_xlabel('Regression MCC')
            axes[1].set_ylabel('Classification MCC')
            axes[1].set_title('ESM+Morgan Models: Regression vs Classification MCC')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "plots" / "mcc_scatter_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Win rate comparison
        if morgan_results or esm_results:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            categories = []
            classification_wins = []
            regression_wins = []
            
            if morgan_results:
                categories.append('Morgan')
                morgan_summary = self.results['morgan_comparison']
                classification_wins.append(morgan_summary['classification_wins'])
                regression_wins.append(morgan_summary['regression_wins'])
            
            if esm_results:
                categories.append('ESM+Morgan')
                esm_summary = self.results['esm_morgan_comparison']
                classification_wins.append(esm_summary['classification_wins'])
                regression_wins.append(esm_summary['regression_wins'])
            
            x = np.arange(len(categories))
            width = 0.35
            
            ax.bar(x - width/2, classification_wins, width, label='Classification Wins', alpha=0.8)
            ax.bar(x + width/2, regression_wins, width, label='Regression Wins', alpha=0.8)
            
            ax.set_xlabel('Model Type')
            ax.set_ylabel('Number of Wins')
            ax.set_title('Model Performance Comparison: Classification vs Regression')
            ax.set_xticks(x)
            ax.set_xticklabels(categories)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "plots" / "win_rate_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info("Visualizations created successfully")
    
    def save_results(self):
        """
        Save analysis results to files
        """
        logger.info("Saving results")
        
        # Save detailed results as JSON
        with open(self.output_dir / "results" / "mcc_analysis_results.json", 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Create summary CSV files
        for comparison_type in ['morgan_comparison', 'esm_morgan_comparison']:
            results = self.results[comparison_type].get('results', [])
            if results:
                df = pd.DataFrame(results)
                df.to_csv(self.output_dir / "results" / f"{comparison_type}_detailed.csv", index=False)
        
        # Create overall summary
        summary_data = []
        for comparison_type in ['morgan_comparison', 'esm_morgan_comparison']:
            summary = self.results[comparison_type]
            if summary.get('total_comparisons', 0) > 0:
                summary_data.append({
                    'model_type': comparison_type.replace('_comparison', ''),
                    'total_comparisons': summary['total_comparisons'],
                    'classification_wins': summary['classification_wins'],
                    'regression_wins': summary['regression_wins'],
                    'classification_win_rate': summary['classification_win_rate'],
                    'mean_mcc_difference': summary['mean_mcc_difference'],
                    'std_mcc_difference': summary['std_mcc_difference'],
                    'mean_accuracy_difference': summary['mean_accuracy_difference'],
                    'mean_f1_difference': summary['mean_f1_difference']
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(self.output_dir / "results" / "mcc_analysis_summary.csv", index=False)
        
        logger.info("Results saved successfully")
    
    def generate_report(self):
        """
        Generate a comprehensive analysis report
        """
        logger.info("Generating analysis report")
        
        report_path = self.output_dir / "MCC_Analysis_Report.md"
        
        with open(report_path, 'w') as f:
            f.write("# MCC Analysis Report: Regression vs Classification QSAR Models\n\n")
            f.write(f"**Generated on:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Analysis Directory:** {self.output_dir}\n\n")
            
            f.write("## Overview\n\n")
            f.write("This report presents a comprehensive comparison between regression and classification QSAR models using the Matthews Correlation Coefficient (MCC) as the primary metric. For regression models, predictions were binarized using a threshold of 7.0 to enable fair comparison with classification models.\n\n")
            
            f.write("## Analysis Results\n\n")
            
            # Morgan comparison results
            morgan_summary = self.results['morgan_comparison']
            if morgan_summary.get('total_comparisons', 0) > 0:
                f.write("### Morgan Models Comparison\n\n")
                f.write(f"- **Total Comparisons:** {morgan_summary['total_comparisons']}\n")
                f.write(f"- **Classification Wins:** {morgan_summary['classification_wins']} ({morgan_summary['classification_win_rate']:.1%})\n")
                f.write(f"- **Regression Wins:** {morgan_summary['regression_wins']} ({1-morgan_summary['classification_win_rate']:.1%})\n")
                f.write(f"- **Mean MCC Difference:** {morgan_summary['mean_mcc_difference']:.4f} ± {morgan_summary['std_mcc_difference']:.4f}\n")
                f.write(f"- **Mean Accuracy Difference:** {morgan_summary['mean_accuracy_difference']:.4f}\n")
                f.write(f"- **Mean F1 Difference:** {morgan_summary['mean_f1_difference']:.4f}\n\n")
            
            # ESM+Morgan comparison results
            esm_summary = self.results['esm_morgan_comparison']
            if esm_summary.get('total_comparisons', 0) > 0:
                f.write("### ESM+Morgan Models Comparison\n\n")
                f.write(f"- **Total Comparisons:** {esm_summary['total_comparisons']}\n")
                f.write(f"- **Classification Wins:** {esm_summary['classification_wins']} ({esm_summary['classification_win_rate']:.1%})\n")
                f.write(f"- **Regression Wins:** {esm_summary['regression_wins']} ({1-esm_summary['classification_win_rate']:.1%})\n")
                f.write(f"- **Mean MCC Difference:** {esm_summary['mean_mcc_difference']:.4f} ± {esm_summary['std_mcc_difference']:.4f}\n")
                f.write(f"- **Mean Accuracy Difference:** {esm_summary['mean_accuracy_difference']:.4f}\n")
                f.write(f"- **Mean F1 Difference:** {esm_summary['mean_f1_difference']:.4f}\n\n")
            
            # Overall conclusions
            f.write("## Conclusions\n\n")
            
            if morgan_summary.get('total_comparisons', 0) > 0 and esm_summary.get('total_comparisons', 0) > 0:
                f.write("### Key Findings\n\n")
                
                # Determine which approach is better for each model type
                morgan_better = "Classification" if morgan_summary['classification_win_rate'] > 0.5 else "Regression"
                esm_better = "Classification" if esm_summary['classification_win_rate'] > 0.5 else "Regression"
                
                f.write(f"1. **Morgan Models:** {morgan_better} models perform better ({morgan_summary['classification_win_rate']:.1%} win rate)\n")
                f.write(f"2. **ESM+Morgan Models:** {esm_better} models perform better ({esm_summary['classification_win_rate']:.1%} win rate)\n")
                
                # MCC improvement analysis
                if morgan_summary['mean_mcc_difference'] > 0:
                    f.write(f"3. **Morgan Models:** Classification shows average MCC improvement of {morgan_summary['mean_mcc_difference']:.4f}\n")
                else:
                    f.write(f"3. **Morgan Models:** Regression shows average MCC improvement of {abs(morgan_summary['mean_mcc_difference']):.4f}\n")
                
                if esm_summary['mean_mcc_difference'] > 0:
                    f.write(f"4. **ESM+Morgan Models:** Classification shows average MCC improvement of {esm_summary['mean_mcc_difference']:.4f}\n")
                else:
                    f.write(f"4. **ESM+Morgan Models:** Regression shows average MCC improvement of {abs(esm_summary['mean_mcc_difference']):.4f}\n")
            
            f.write("\n### Recommendations\n\n")
            f.write("Based on the MCC analysis:\n")
            f.write("- Use the model type that shows higher win rate for each specific approach\n")
            f.write("- Consider the magnitude of MCC differences when making final decisions\n")
            f.write("- Evaluate additional metrics (accuracy, F1, precision, recall) for comprehensive assessment\n")
            f.write("- Consider the specific protein targets and their characteristics when choosing between regression and classification\n\n")
            
            f.write("## Files Generated\n\n")
            f.write("- `mcc_analysis_results.json`: Complete analysis results\n")
            f.write("- `morgan_comparison_detailed.csv`: Detailed Morgan model comparisons\n")
            f.write("- `esm_morgan_comparison_detailed.csv`: Detailed ESM+Morgan model comparisons\n")
            f.write("- `mcc_analysis_summary.csv`: Summary statistics\n")
            f.write("- `plots/`: Visualization files\n")
            f.write("  - `mcc_difference_distributions.png`: Distribution of MCC differences\n")
            f.write("  - `mcc_scatter_comparison.png`: Scatter plots of regression vs classification MCC\n")
            f.write("  - `win_rate_comparison.png`: Win rate comparison between model types\n\n")
        
        logger.info(f"Analysis report generated: {report_path}")
    
    def run_complete_analysis(self):
        """
        Run the complete MCC analysis
        """
        logger.info("Starting complete MCC analysis")
        
        # Run comparisons
        self.run_morgan_comparison()
        self.run_esm_morgan_comparison()
        
        # Create visualizations
        self.create_visualizations()
        
        # Save results
        self.save_results()
        
        # Generate report
        self.generate_report()
        
        logger.info("Complete MCC analysis finished")
        
        return self.results


def main():
    """
    Main function to run the MCC analysis
    """
    # Initialize analysis
    analysis = MCCAnalysis()
    
    # Run complete analysis
    results = analysis.run_complete_analysis()
    
    # Print summary
    print("\n" + "="*60)
    print("MCC ANALYSIS SUMMARY")
    print("="*60)
    
    for comparison_type in ['morgan_comparison', 'esm_morgan_comparison']:
        summary = results[comparison_type]
        if summary.get('total_comparisons', 0) > 0:
            print(f"\n{comparison_type.upper().replace('_', ' ')}:")
            print(f"  Total Comparisons: {summary['total_comparisons']}")
            print(f"  Classification Wins: {summary['classification_wins']} ({summary['classification_win_rate']:.1%})")
            print(f"  Regression Wins: {summary['regression_wins']} ({1-summary['classification_win_rate']:.1%})")
            print(f"  Mean MCC Difference: {summary['mean_mcc_difference']:.4f} ± {summary['std_mcc_difference']:.4f}")
    
    print(f"\nResults saved to: {analysis.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()