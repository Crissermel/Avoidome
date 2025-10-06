#!/usr/bin/env python3
"""
Comprehensive Analysis of QSAR Model Results
Analyzes results from all 4 model sets and generates detailed report
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_results():
    """Load results from all model types"""
    results = {}
    
    # Load Morgan-only results (50-sample threshold)
    morgan_50 = pd.read_csv('modeling_summary.csv')
    results['Morgan_50'] = morgan_50
    
    # Load ESM+Morgan results (50-sample threshold)
    esm_morgan_50 = pd.read_csv('esm_morgan_modeling_summary.csv')
    results['ESM_Morgan_50'] = esm_morgan_50
    
    # Check if 30-sample threshold results exist
    if Path('modeling_summary_30.csv').exists():
        morgan_30 = pd.read_csv('modeling_summary_30.csv')
        results['Morgan_30'] = morgan_30
    
    if Path('esm_morgan_modeling_summary_30.csv').exists():
        esm_morgan_30 = pd.read_csv('esm_morgan_modeling_summary_30.csv')
        results['ESM_Morgan_30'] = esm_morgan_30
    
    return results

def analyze_model_performance(results):
    """Analyze performance metrics for each model type"""
    analysis = {}
    
    for model_name, df in results.items():
        completed = df[df['status'] == 'completed']
        
        if len(completed) > 0:
            analysis[model_name] = {
                'total_proteins': len(df),
                'completed': len(completed),
                'skipped': len(df[df['status'] == 'skipped']),
                'errors': len(df[df['status'] == 'error']),
                'completion_rate': len(completed) / len(df) * 100,
                'avg_r2': completed['regression_r2'].mean(),
                'avg_rmse': completed['regression_rmse'].mean(),
                'avg_accuracy': completed['classification_accuracy'].mean(),
                'avg_f1': completed['classification_f1'].mean(),
                'avg_auc': completed['classification_auc'].mean(),
                'std_r2': completed['regression_r2'].std(),
                'std_accuracy': completed['classification_accuracy'].std(),
                'best_r2': completed['regression_r2'].max(),
                'best_accuracy': completed['classification_accuracy'].max(),
                'worst_r2': completed['regression_r2'].min(),
                'worst_accuracy': completed['classification_accuracy'].min()
            }
        else:
            analysis[model_name] = {
                'total_proteins': len(df),
                'completed': 0,
                'skipped': len(df[df['status'] == 'skipped']),
                'errors': len(df[df['status'] == 'error']),
                'completion_rate': 0,
                'avg_r2': np.nan,
                'avg_rmse': np.nan,
                'avg_accuracy': np.nan,
                'avg_f1': np.nan,
                'avg_auc': np.nan,
                'std_r2': np.nan,
                'std_accuracy': np.nan,
                'best_r2': np.nan,
                'best_accuracy': np.nan,
                'worst_r2': np.nan,
                'worst_accuracy': np.nan
            }
    
    return analysis

def create_summary_table(analysis):
    """Create comprehensive summary table"""
    summary_data = []
    
    for model_name, metrics in analysis.items():
        summary_data.append({
            'Model Type': model_name,
            'Total Proteins': metrics['total_proteins'],
            'Completed': metrics['completed'],
            'Skipped': metrics['skipped'],
            'Errors': metrics['errors'],
            'Completion Rate (%)': f"{metrics['completion_rate']:.1f}",
            'Avg R²': f"{metrics['avg_r2']:.3f}" if not np.isnan(metrics['avg_r2']) else "N/A",
            'Avg RMSE': f"{metrics['avg_rmse']:.3f}" if not np.isnan(metrics['avg_rmse']) else "N/A",
            'Avg Accuracy': f"{metrics['avg_accuracy']:.3f}" if not np.isnan(metrics['avg_accuracy']) else "N/A",
            'Avg F1': f"{metrics['avg_f1']:.3f}" if not np.isnan(metrics['avg_f1']) else "N/A",
            'Avg AUC': f"{metrics['avg_auc']:.3f}" if not np.isnan(metrics['avg_auc']) else "N/A",
            'Best R²': f"{metrics['best_r2']:.3f}" if not np.isnan(metrics['best_r2']) else "N/A",
            'Best Accuracy': f"{metrics['best_accuracy']:.3f}" if not np.isnan(metrics['best_accuracy']) else "N/A"
        })
    
    return pd.DataFrame(summary_data)

def create_detailed_comparison_table(results):
    """Create detailed protein-by-protein comparison table"""
    # Get all completed proteins across all models
    all_proteins = set()
    for df in results.values():
        completed = df[df['status'] == 'completed']
        all_proteins.update(completed['protein_name'].tolist())
    
    all_proteins = sorted(list(all_proteins))
    
    comparison_data = []
    
    for protein in all_proteins:
        row = {'Protein': protein}
        
        for model_name, df in results.items():
            protein_data = df[df['protein_name'] == protein]
            
            if len(protein_data) > 0 and protein_data.iloc[0]['status'] == 'completed':
                data = protein_data.iloc[0]
                row[f'{model_name}_R2'] = f"{data['regression_r2']:.3f}"
                row[f'{model_name}_RMSE'] = f"{data['regression_rmse']:.3f}"
                row[f'{model_name}_Accuracy'] = f"{data['classification_accuracy']:.3f}"
                row[f'{model_name}_F1'] = f"{data['classification_f1']:.3f}"
                row[f'{model_name}_AUC'] = f"{data['classification_auc']:.3f}"
                row[f'{model_name}_Samples'] = int(data['n_samples'])
            else:
                row[f'{model_name}_R2'] = "N/A"
                row[f'{model_name}_RMSE'] = "N/A"
                row[f'{model_name}_Accuracy'] = "N/A"
                row[f'{model_name}_F1'] = "N/A"
                row[f'{model_name}_AUC'] = "N/A"
                row[f'{model_name}_Samples'] = "N/A"
        
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)

def generate_report(results, analysis, summary_table, comparison_table):
    """Generate comprehensive report"""
    report = []
    
    report.append("# QSAR Model Performance Analysis Report")
    report.append("=" * 60)
    report.append("")
    
    report.append("## Executive Summary")
    report.append("")
    report.append("This report analyzes the performance of four QSAR model configurations:")
    report.append("1. **Morgan + Physicochemical** (50-sample threshold)")
    report.append("2. **ESM + Morgan + Physicochemical** (50-sample threshold)")
    report.append("3. **Morgan + Physicochemical** (30-sample threshold) - if available")
    report.append("4. **ESM + Morgan + Physicochemical** (30-sample threshold) - if available")
    report.append("")
    
    report.append("## Model Performance Summary")
    report.append("")
    report.append(summary_table.to_string(index=False))
    report.append("")
    
    report.append("## Key Findings")
    report.append("")
    
    # Find best performing models
    valid_models = {k: v for k, v in analysis.items() if not np.isnan(v['avg_r2'])}
    
    if valid_models:
        best_r2_model = max(valid_models.items(), key=lambda x: x[1]['avg_r2'])
        best_acc_model = max(valid_models.items(), key=lambda x: x[1]['avg_accuracy'])
        
        report.append(f"### Best Regression Performance (R²)")
        report.append(f"- **Model**: {best_r2_model[0]}")
        report.append(f"- **Average R²**: {best_r2_model[1]['avg_r2']:.3f}")
        report.append(f"- **Best R²**: {best_r2_model[1]['best_r2']:.3f}")
        report.append("")
        
        report.append(f"### Best Classification Performance (Accuracy)")
        report.append(f"- **Model**: {best_acc_model[0]}")
        report.append(f"- **Average Accuracy**: {best_acc_model[1]['avg_accuracy']:.3f}")
        report.append(f"- **Best Accuracy**: {best_acc_model[1]['best_accuracy']:.3f}")
        report.append("")
    
    report.append("### Model Comparison Insights")
    report.append("")
    
    # Compare Morgan vs ESM+Morgan
    morgan_models = {k: v for k, v in valid_models.items() if 'Morgan' in k and 'ESM' not in k}
    esm_models = {k: v for k, v in valid_models.items() if 'ESM' in k}
    
    if morgan_models and esm_models:
        morgan_avg_r2 = np.mean([v['avg_r2'] for v in morgan_models.values()])
        esm_avg_r2 = np.mean([v['avg_r2'] for v in esm_models.values()])
        morgan_avg_acc = np.mean([v['avg_accuracy'] for v in morgan_models.values()])
        esm_avg_acc = np.mean([v['avg_accuracy'] for v in esm_models.values()])
        
        report.append(f"- **ESM Addition Impact on R²**: {esm_avg_r2 - morgan_avg_r2:+.3f}")
        report.append(f"- **ESM Addition Impact on Accuracy**: {esm_avg_acc - morgan_avg_acc:+.3f}")
        report.append("")
    
    # Compare threshold impact
    if 'Morgan_50' in valid_models and 'Morgan_30' in valid_models:
        r2_diff = valid_models['Morgan_30']['avg_r2'] - valid_models['Morgan_50']['avg_r2']
        acc_diff = valid_models['Morgan_30']['avg_accuracy'] - valid_models['Morgan_50']['avg_accuracy']
        report.append(f"- **30-sample vs 50-sample threshold (Morgan)**:")
        report.append(f"  - R² change: {r2_diff:+.3f}")
        report.append(f"  - Accuracy change: {acc_diff:+.3f}")
        report.append("")
    
    if 'ESM_Morgan_50' in valid_models and 'ESM_Morgan_30' in valid_models:
        r2_diff = valid_models['ESM_Morgan_30']['avg_r2'] - valid_models['ESM_Morgan_50']['avg_r2']
        acc_diff = valid_models['ESM_Morgan_30']['avg_accuracy'] - valid_models['ESM_Morgan_50']['avg_accuracy']
        report.append(f"- **30-sample vs 50-sample threshold (ESM+Morgan)**:")
        report.append(f"  - R² change: {r2_diff:+.3f}")
        report.append(f"  - Accuracy change: {acc_diff:+.3f}")
        report.append("")
    
    report.append("## Detailed Protein-by-Protein Comparison")
    report.append("")
    report.append("The following table shows performance metrics for each protein across all model types:")
    report.append("")
    report.append(comparison_table.to_string(index=False))
    report.append("")
    
    report.append("## Recommendations")
    report.append("")
    report.append("1. **Feature Engineering**: ESM embeddings provide additional protein-specific information")
    report.append("2. **Sample Size**: Lower thresholds (30 samples) may capture more proteins but with potentially lower reliability")
    report.append("3. **Model Selection**: Consider the trade-off between model complexity and performance")
    report.append("4. **Data Quality**: Focus on proteins with sufficient high-quality bioactivity data")
    report.append("")
    
    return "\n".join(report)

def main():
    """Main analysis function"""
    print("Loading QSAR model results...")
    results = load_results()
    
    print("Analyzing model performance...")
    analysis = analyze_model_performance(results)
    
    print("Creating summary table...")
    summary_table = create_summary_table(analysis)
    
    print("Creating detailed comparison table...")
    comparison_table = create_detailed_comparison_table(results)
    
    print("Generating comprehensive report...")
    report = generate_report(results, analysis, summary_table, comparison_table)
    
    # Save results
    summary_table.to_csv('model_performance_summary.csv', index=False)
    comparison_table.to_csv('detailed_protein_comparison.csv', index=False)
    
    with open('QSAR_Analysis_Report.md', 'w') as f:
        f.write(report)
    
    print("\nAnalysis complete!")
    print("Files generated:")
    print("- model_performance_summary.csv")
    print("- detailed_protein_comparison.csv")
    print("- QSAR_Analysis_Report.md")
    
    # Print summary to console
    print("\n" + "="*60)
    print("QUICK SUMMARY")
    print("="*60)
    print(summary_table.to_string(index=False))

if __name__ == "__main__":
    main()