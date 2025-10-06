#!/usr/bin/env python3
"""
QSAR Model Results Analysis and Visualization

This script analyzes the QSAR model results from the avoidome similarity expansion
workflow and creates comprehensive visualizations and summaries.

Author: AQSE Pipeline
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import json
import warnings
from typing import Dict, List, Tuple, Any
import glob

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

class QSARResultsAnalyzer:
    """Analyzes QSAR model results and creates visualizations"""
    
    def __init__(self, results_dir: str, output_dir: str):
        """
        Initialize the analyzer
        
        Args:
            results_dir: Directory containing QSAR model results
            output_dir: Directory to save analysis outputs
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "tables").mkdir(exist_ok=True)
        
        # Load all metrics
        self.metrics_df = self.load_all_metrics()
        
        # Load predictions for detailed analysis
        self.predictions_data = self.load_predictions_data()
        
    def load_all_metrics(self) -> pd.DataFrame:
        """Load all model performance metrics"""
        print("Loading model performance metrics...")
        
        metrics_files = list(self.results_dir.glob("metrics/*_metrics.json"))
        
        all_metrics = []
        for file_path in metrics_files:
            try:
                with open(file_path, 'r') as f:
                    metrics = json.load(f)
                    all_metrics.append(metrics)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        if not all_metrics:
            print("No metrics found!")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_metrics)
        
        # Extract target name from target field (remove threshold suffix)
        df['target_name'] = df['target'].str.replace('_(high|medium|low)$', '', regex=True)
        
        print(f"Loaded metrics for {len(df)} models across {df['target_name'].nunique()} targets")
        return df
    
    def load_predictions_data(self) -> Dict[str, pd.DataFrame]:
        """Load prediction data for all models"""
        print("Loading prediction data...")
        
        predictions_data = {}
        pred_files = list(self.results_dir.glob("predictions/*_predictions.csv"))
        
        for file_path in pred_files:
            try:
                # Extract model identifier from filename
                model_id = file_path.stem.replace('_predictions', '')
                pred_df = pd.read_csv(file_path)
                predictions_data[model_id] = pred_df
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        print(f"Loaded predictions for {len(predictions_data)} models")
        return predictions_data
    
    def create_performance_overview(self):
        """Create overview plots of model performance"""
        print("Creating performance overview plots...")
        
        # Set up the figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('QSAR Model Performance Overview', fontsize=16, fontweight='bold')
        
        # 1. R² distribution by threshold
        sns.boxplot(data=self.metrics_df, x='threshold', y='val_r2', ax=axes[0, 0])
        axes[0, 0].set_title('Validation R² by Similarity Threshold')
        axes[0, 0].set_ylabel('R²')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. RMSE distribution by threshold
        sns.boxplot(data=self.metrics_df, x='threshold', y='val_rmse', ax=axes[0, 1])
        axes[0, 1].set_title('Validation RMSE by Similarity Threshold')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Q² distribution by threshold
        sns.boxplot(data=self.metrics_df, x='threshold', y='q2', ax=axes[0, 2])
        axes[0, 2].set_title('Q² by Similarity Threshold')
        axes[0, 2].set_ylabel('Q²')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Training sample size by threshold
        sns.boxplot(data=self.metrics_df, x='threshold', y='n_train', ax=axes[1, 0])
        axes[1, 0].set_title('Training Sample Size by Threshold')
        axes[1, 0].set_ylabel('Number of Samples')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Validation sample size by threshold
        sns.boxplot(data=self.metrics_df, x='threshold', y='n_val', ax=axes[1, 1])
        axes[1, 1].set_title('Validation Sample Size by Threshold')
        axes[1, 1].set_ylabel('Number of Samples')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. R² vs Q² correlation
        sns.scatterplot(data=self.metrics_df, x='val_r2', y='q2', hue='threshold', ax=axes[1, 2])
        axes[1, 2].set_title('Validation R² vs Q²')
        axes[1, 2].set_xlabel('Validation R²')
        axes[1, 2].set_ylabel('Q²')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Add diagonal line for reference
        min_val = min(self.metrics_df['val_r2'].min(), self.metrics_df['q2'].min())
        max_val = max(self.metrics_df['val_r2'].max(), self.metrics_df['q2'].max())
        axes[1, 2].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "performance_overview.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_target_performance_analysis(self):
        """Create detailed analysis of performance by target"""
        print("Creating target performance analysis...")
        
        # Calculate average performance per target
        target_performance = self.metrics_df.groupby('target_name').agg({
            'val_r2': ['mean', 'std', 'min', 'max'],
            'val_rmse': ['mean', 'std', 'min', 'max'],
            'q2': ['mean', 'std', 'min', 'max'],
            'n_train': 'mean',
            'n_val': 'mean'
        }).round(3)
        
        # Flatten column names
        target_performance.columns = ['_'.join(col).strip() for col in target_performance.columns]
        target_performance = target_performance.reset_index()
        
        # Sort by average R²
        target_performance = target_performance.sort_values('val_r2_mean', ascending=False)
        
        # Save target performance table
        target_performance.to_csv(self.output_dir / "tables" / "target_performance_summary.csv", index=False)
        
        # Create target performance plots
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle('Target Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Top 20 targets by R²
        top_targets = target_performance.head(20)
        y_pos = np.arange(len(top_targets))
        
        axes[0, 0].barh(y_pos, top_targets['val_r2_mean'], 
                       xerr=top_targets['val_r2_std'], capsize=3)
        axes[0, 0].set_yticks(y_pos)
        axes[0, 0].set_yticklabels(top_targets['target_name'], fontsize=8)
        axes[0, 0].set_xlabel('Average Validation R²')
        axes[0, 0].set_title('Top 20 Targets by Validation R²')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. R² vs Training Sample Size
        sns.scatterplot(data=target_performance, x='n_train_mean', y='val_r2_mean', 
                       size='n_val_mean', sizes=(20, 200), ax=axes[0, 1])
        axes[0, 1].set_xlabel('Average Training Sample Size')
        axes[0, 1].set_ylabel('Average Validation R²')
        axes[0, 1].set_title('R² vs Training Sample Size')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Q² vs R² by target
        sns.scatterplot(data=target_performance, x='val_r2_mean', y='q2_mean', 
                       size='n_train_mean', sizes=(20, 200), ax=axes[1, 0])
        axes[1, 0].set_xlabel('Average Validation R²')
        axes[1, 0].set_ylabel('Average Q²')
        axes[1, 0].set_title('Q² vs R² by Target')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add diagonal line
        min_val = min(target_performance['val_r2_mean'].min(), target_performance['q2_mean'].min())
        max_val = max(target_performance['val_r2_mean'].max(), target_performance['q2_mean'].max())
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        # 4. RMSE distribution by target (top 20)
        top_targets_rmse = target_performance.head(20)
        y_pos = np.arange(len(top_targets_rmse))
        
        axes[1, 1].barh(y_pos, top_targets_rmse['val_rmse_mean'], 
                       xerr=top_targets_rmse['val_rmse_std'], capsize=3)
        axes[1, 1].set_yticks(y_pos)
        axes[1, 1].set_yticklabels(top_targets_rmse['target_name'], fontsize=8)
        axes[1, 1].set_xlabel('Average Validation RMSE')
        axes[1, 1].set_title('Top 20 Targets by Validation RMSE (Lower is Better)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "target_performance_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        return target_performance
    
    def create_prediction_analysis(self):
        """Create analysis of prediction quality"""
        print("Creating prediction quality analysis...")
        
        # Select a few representative models for detailed analysis
        representative_models = []
        
        # Get best performing model for each threshold
        for threshold in ['high', 'medium', 'low']:
            threshold_data = self.metrics_df[self.metrics_df['threshold'] == threshold]
            best_model = threshold_data.loc[threshold_data['val_r2'].idxmax()]
            representative_models.append(best_model['target'])
        
        # Add a few more diverse examples
        diverse_models = self.metrics_df.nlargest(6, 'val_r2')['target'].tolist()
        representative_models.extend(diverse_models)
        representative_models = list(set(representative_models))[:6]  # Limit to 6 models
        
        # Create prediction plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Prediction Quality Analysis - Representative Models', fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        for i, model_id in enumerate(representative_models):
            if model_id in self.predictions_data:
                pred_data = self.predictions_data[model_id]
                
                # Scatter plot of actual vs predicted
                axes[i].scatter(pred_data['actual'], pred_data['predicted'], alpha=0.6, s=20)
                
                # Perfect prediction line
                min_val = min(pred_data['actual'].min(), pred_data['predicted'].min())
                max_val = max(pred_data['actual'].max(), pred_data['predicted'].max())
                axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
                
                # Calculate R² for this plot
                from sklearn.metrics import r2_score
                r2 = r2_score(pred_data['actual'], pred_data['predicted'])
                
                axes[i].set_xlabel('Actual pChEMBL')
                axes[i].set_ylabel('Predicted pChEMBL')
                axes[i].set_title(f'{model_id}\nR² = {r2:.3f}')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "prediction_quality_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_threshold_comparison(self):
        """Create detailed comparison across similarity thresholds"""
        print("Creating threshold comparison analysis...")
        
        # Pivot data for easier comparison
        r2_pivot = self.metrics_df.pivot(index='target_name', columns='threshold', values='val_r2')
        rmse_pivot = self.metrics_df.pivot(index='target_name', columns='threshold', values='val_rmse')
        q2_pivot = self.metrics_df.pivot(index='target_name', columns='threshold', values='q2')
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Similarity Threshold Comparison', fontsize=16, fontweight='bold')
        
        # 1. R² comparison
        r2_pivot.plot(kind='bar', ax=axes[0, 0], width=0.8)
        axes[0, 0].set_title('Validation R² by Target and Threshold')
        axes[0, 0].set_ylabel('R²')
        axes[0, 0].legend(title='Threshold')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. RMSE comparison
        rmse_pivot.plot(kind='bar', ax=axes[0, 1], width=0.8)
        axes[0, 1].set_title('Validation RMSE by Target and Threshold')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].legend(title='Threshold')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Q² comparison
        q2_pivot.plot(kind='bar', ax=axes[1, 0], width=0.8)
        axes[1, 0].set_title('Q² by Target and Threshold')
        axes[1, 0].set_ylabel('Q²')
        axes[1, 0].legend(title='Threshold')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Threshold performance summary
        threshold_summary = self.metrics_df.groupby('threshold').agg({
            'val_r2': ['mean', 'std'],
            'val_rmse': ['mean', 'std'],
            'q2': ['mean', 'std']
        }).round(3)
        
        # Create a simple bar plot of means
        metrics = ['val_r2', 'val_rmse', 'q2']
        x = np.arange(len(metrics))
        width = 0.25
        
        for i, threshold in enumerate(['high', 'medium', 'low']):
            means = [threshold_summary.loc[threshold, (metric, 'mean')] for metric in metrics]
            stds = [threshold_summary.loc[threshold, (metric, 'std')] for metric in metrics]
            
            axes[1, 1].bar(x + i*width, means, width, label=threshold, 
                          yerr=stds, capsize=5, alpha=0.8)
        
        axes[1, 1].set_xlabel('Metrics')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_title('Average Performance by Threshold')
        axes[1, 1].set_xticks(x + width)
        axes[1, 1].set_xticklabels(['R²', 'RMSE', 'Q²'])
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "threshold_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save threshold comparison tables
        r2_pivot.to_csv(self.output_dir / "tables" / "r2_by_threshold.csv")
        rmse_pivot.to_csv(self.output_dir / "tables" / "rmse_by_threshold.csv")
        q2_pivot.to_csv(self.output_dir / "tables" / "q2_by_threshold.csv")
        threshold_summary.to_csv(self.output_dir / "tables" / "threshold_summary_stats.csv")
    
    def create_model_quality_assessment(self):
        """Create model quality assessment plots"""
        print("Creating model quality assessment...")
        
        # Define quality categories based on R²
        def categorize_quality(r2):
            if r2 >= 0.7:
                return 'Excellent (R² ≥ 0.7)'
            elif r2 >= 0.5:
                return 'Good (0.5 ≤ R² < 0.7)'
            elif r2 >= 0.3:
                return 'Fair (0.3 ≤ R² < 0.5)'
            else:
                return 'Poor (R² < 0.3)'
        
        self.metrics_df['quality_category'] = self.metrics_df['val_r2'].apply(categorize_quality)
        
        # Create quality assessment plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Quality Assessment', fontsize=16, fontweight='bold')
        
        # 1. Quality distribution
        quality_counts = self.metrics_df['quality_category'].value_counts()
        colors = ['#2ecc71', '#f39c12', '#e74c3c', '#95a5a6']  # Green, Orange, Red, Gray
        axes[0, 0].pie(quality_counts.values, labels=quality_counts.index, autopct='%1.1f%%', 
                      colors=colors[:len(quality_counts)])
        axes[0, 0].set_title('Model Quality Distribution')
        
        # 2. Quality by threshold
        quality_threshold = pd.crosstab(self.metrics_df['quality_category'], 
                                      self.metrics_df['threshold'])
        quality_threshold.plot(kind='bar', ax=axes[0, 1], stacked=True, color=colors[:len(quality_counts)])
        axes[0, 1].set_title('Model Quality by Similarity Threshold')
        axes[0, 1].set_ylabel('Number of Models')
        axes[0, 1].legend(title='Threshold')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. R² vs Q² colored by quality
        quality_colors = {'Excellent (R² ≥ 0.7)': '#2ecc71',
                         'Good (0.5 ≤ R² < 0.7)': '#f39c12',
                         'Fair (0.3 ≤ R² < 0.5)': '#e74c3c',
                         'Poor (R² < 0.3)': '#95a5a6'}
        
        for category in self.metrics_df['quality_category'].unique():
            data = self.metrics_df[self.metrics_df['quality_category'] == category]
            axes[1, 0].scatter(data['val_r2'], data['q2'], 
                              label=category, color=quality_colors[category], alpha=0.7)
        
        axes[1, 0].set_xlabel('Validation R²')
        axes[1, 0].set_ylabel('Q²')
        axes[1, 0].set_title('R² vs Q² by Model Quality')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add diagonal line
        min_val = min(self.metrics_df['val_r2'].min(), self.metrics_df['q2'].min())
        max_val = max(self.metrics_df['val_r2'].max(), self.metrics_df['q2'].max())
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        # 4. Sample size vs quality
        sns.boxplot(data=self.metrics_df, x='quality_category', y='n_train', ax=axes[1, 1])
        axes[1, 1].set_title('Training Sample Size by Model Quality')
        axes[1, 1].set_ylabel('Training Sample Size')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "model_quality_assessment.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save quality assessment table
        quality_summary = self.metrics_df.groupby('quality_category').agg({
            'val_r2': ['count', 'mean', 'std'],
            'val_rmse': ['mean', 'std'],
            'q2': ['mean', 'std'],
            'n_train': ['mean', 'std']
        }).round(3)
        quality_summary.to_csv(self.output_dir / "tables" / "model_quality_summary.csv")
    
    def generate_summary_report(self, target_performance: pd.DataFrame):
        """Generate a comprehensive summary report"""
        print("Generating summary report...")
        
        report = []
        report.append("# QSAR Model Results Summary Report")
        report.append("=" * 50)
        report.append("")
        
        # Overall statistics
        report.append("## Overall Statistics")
        report.append(f"- Total models created: {len(self.metrics_df)}")
        report.append(f"- Number of targets: {self.metrics_df['target_name'].nunique()}")
        report.append(f"- Similarity thresholds: {', '.join(self.metrics_df['threshold'].unique())}")
        report.append("")
        
        # Performance metrics
        report.append("## Performance Metrics Summary")
        report.append(f"- Average validation R²: {self.metrics_df['val_r2'].mean():.3f} ± {self.metrics_df['val_r2'].std():.3f}")
        report.append(f"- Average validation RMSE: {self.metrics_df['val_rmse'].mean():.3f} ± {self.metrics_df['val_rmse'].std():.3f}")
        report.append(f"- Average Q²: {self.metrics_df['q2'].mean():.3f} ± {self.metrics_df['q2'].std():.3f}")
        report.append("")
        
        # Best performing models
        report.append("## Top 10 Best Performing Models (by Validation R²)")
        top_models = self.metrics_df.nlargest(10, 'val_r2')[['target', 'threshold', 'val_r2', 'val_rmse', 'q2', 'n_train', 'n_val']]
        report.append(top_models.to_string(index=False))
        report.append("")
        
        # Quality distribution
        quality_counts = self.metrics_df['quality_category'].value_counts()
        report.append("## Model Quality Distribution")
        for category, count in quality_counts.items():
            percentage = (count / len(self.metrics_df)) * 100
            report.append(f"- {category}: {count} models ({percentage:.1f}%)")
        report.append("")
        
        # Threshold comparison
        report.append("## Performance by Similarity Threshold")
        threshold_stats = self.metrics_df.groupby('threshold').agg({
            'val_r2': ['mean', 'std'],
            'val_rmse': ['mean', 'std'],
            'q2': ['mean', 'std']
        }).round(3)
        
        for threshold in ['high', 'medium', 'low']:
            if threshold in threshold_stats.index:
                stats = threshold_stats.loc[threshold]
                report.append(f"### {threshold.upper()} Threshold")
                report.append(f"- Average R²: {stats[('val_r2', 'mean')]:.3f} ± {stats[('val_r2', 'std')]:.3f}")
                report.append(f"- Average RMSE: {stats[('val_rmse', 'mean')]:.3f} ± {stats[('val_rmse', 'std')]:.3f}")
                report.append(f"- Average Q²: {stats[('q2', 'mean')]:.3f} ± {stats[('q2', 'std')]:.3f}")
                report.append("")
        
        # Top performing targets
        report.append("## Top 10 Best Performing Targets (Average across thresholds)")
        top_targets = target_performance.head(10)[['target_name', 'val_r2_mean', 'val_r2_std', 'val_rmse_mean', 'q2_mean']]
        report.append(top_targets.to_string(index=False))
        report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        excellent_models = len(self.metrics_df[self.metrics_df['val_r2'] >= 0.7])
        good_models = len(self.metrics_df[(self.metrics_df['val_r2'] >= 0.5) & (self.metrics_df['val_r2'] < 0.7)])
        
        report.append(f"- {excellent_models} models achieved excellent performance (R² ≥ 0.7)")
        report.append(f"- {good_models} models achieved good performance (0.5 ≤ R² < 0.7)")
        
        if excellent_models > 0:
            report.append("- Focus on the excellent performing models for further development")
        
        if good_models > 0:
            report.append("- Consider additional feature engineering for good performing models")
        
        poor_models = len(self.metrics_df[self.metrics_df['val_r2'] < 0.3])
        if poor_models > 0:
            report.append(f"- {poor_models} models showed poor performance (R² < 0.3) - consider alternative approaches")
        
        report.append("")
        report.append("## Files Generated")
        report.append("- `performance_overview.png`: Overall performance metrics")
        report.append("- `target_performance_analysis.png`: Detailed target analysis")
        report.append("- `prediction_quality_analysis.png`: Prediction quality examples")
        report.append("- `threshold_comparison.png`: Similarity threshold comparison")
        report.append("- `model_quality_assessment.png`: Model quality distribution")
        report.append("- `target_performance_summary.csv`: Detailed target statistics")
        report.append("- `threshold_summary_stats.csv`: Threshold comparison statistics")
        report.append("- `model_quality_summary.csv`: Quality category statistics")
        
        # Save report
        report_text = "\n".join(report)
        with open(self.output_dir / "QSAR_Results_Summary_Report.md", 'w') as f:
            f.write(report_text)
        
        print("Summary report saved to QSAR_Results_Summary_Report.md")
        return report_text
    
    def run_complete_analysis(self):
        """Run the complete analysis workflow"""
        print("Starting QSAR results analysis...")
        print(f"Results directory: {self.results_dir}")
        print(f"Output directory: {self.output_dir}")
        print("")
        
        if self.metrics_df.empty:
            print("No metrics data found. Please check the results directory.")
            return
        
        # Create all visualizations
        self.create_performance_overview()
        target_performance = self.create_target_performance_analysis()
        self.create_prediction_analysis()
        self.create_threshold_comparison()
        self.create_model_quality_assessment()
        
        # Generate summary report
        report = self.generate_summary_report(target_performance)
        
        print("\n" + "="*60)
        print("QSAR RESULTS ANALYSIS COMPLETED")
        print("="*60)
        print(f"Analysis saved to: {self.output_dir}")
        print(f"Total models analyzed: {len(self.metrics_df)}")
        print(f"Targets analyzed: {self.metrics_df['target_name'].nunique()}")
        print(f"Average validation R²: {self.metrics_df['val_r2'].mean():.3f}")
        print("="*60)

def main():
    """Main function to run the analysis"""
    
    # Set up paths
    results_dir = "/zfsdata/data/cristina/avoidome_qsar_models_temp"
    output_dir = "/home/serramelendezcsm/RA/Avoidome/analyses/avoidome_qsar_similarity_expansion/04_qsar_models"
    
    # Create analyzer and run analysis
    analyzer = QSARResultsAnalyzer(results_dir, output_dir)
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()


