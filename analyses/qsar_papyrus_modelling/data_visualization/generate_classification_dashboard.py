#!/usr/bin/env python3
"""
Generate Classification Dashboard for Papyrus QSAR
This script generates comprehensive visualizations and reports for the classification results.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add the parent directory to the path
sys.path.append('/home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_modelling')

from data_visualization.visualization_utils import PapyrusVisualizer

def generate_classification_dashboard():
    """Generate comprehensive classification dashboard"""
    
    print("Generating Classification Dashboard for Papyrus QSAR...")
    
    # Initialize visualizer
    visualizer = PapyrusVisualizer()
    
    # Load classification results
    classification_path = "/home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_modelling/classification_results.csv"
    
    if not os.path.exists(classification_path):
        print(f"Classification results not found at: {classification_path}")
        return None
    
    classification_df = pd.read_csv(classification_path)
    print(f"Loaded classification results: {len(classification_df)} total records")
    
    # Filter successful models
    successful_models = classification_df[classification_df['accuracy'].notna()]
    print(f"Successful models: {len(successful_models)} records")
    
    if len(successful_models) == 0:
        print("No successful classification models found!")
        return None
    
    # Generate classification performance plots
    print("Generating classification performance plots...")
    classification_stats = visualizer.generate_classification_performance_plots(classification_df)
    
    # Generate detailed analysis
    print("Generating detailed classification analysis...")
    classification_detailed_stats = visualizer.generate_classification_detailed_analysis(classification_df)
    
    # Generate summary report
    print("Generating summary report...")
    generate_classification_summary_report(classification_df, classification_stats, classification_detailed_stats)
    
    print(f"Classification dashboard generated successfully!")
    print(f"All plots saved to: {visualizer.output_dir}")
    
    return {
        'classification_stats': classification_stats,
        'classification_detailed_stats': classification_detailed_stats
    }

def generate_classification_summary_report(classification_df, classification_stats, detailed_stats):
    """Generate a comprehensive summary report"""
    
    # Filter successful models
    successful_models = classification_df[classification_df['accuracy'].notna()]
    
    # Calculate summary statistics
    total_proteins = len(classification_df['protein'].unique())
    proteins_with_models = len(successful_models['protein'].unique())
    
    # Calculate average metrics across all folds
    avg_accuracy = successful_models['accuracy'].mean()
    avg_precision = successful_models['precision'].mean()
    avg_recall = successful_models['recall'].mean()
    avg_f1 = successful_models['f1_score'].mean()
    
    # Find best performing proteins
    protein_performance = successful_models.groupby('protein').agg({
        'accuracy': 'mean',
        'precision': 'mean',
        'recall': 'mean',
        'f1_score': 'mean',
        'n_samples': 'first',
        'n_active': 'first',
        'n_inactive': 'first'
    }).reset_index()
    
    best_accuracy_protein = protein_performance.loc[protein_performance['accuracy'].idxmax()]
    best_f1_protein = protein_performance.loc[protein_performance['f1_score'].idxmax()]
    
    # Create summary report
    report_path = "/home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_modelling/classification_summary_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PAPYRUS QSAR CLASSIFICATION SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total proteins analyzed: {total_proteins}\n")
        f.write(f"Proteins with successful models: {proteins_with_models}\n")
        f.write(f"Success rate: {proteins_with_models/total_proteins*100:.1f}%\n")
        f.write(f"Total cross-validation folds: {len(successful_models)}\n\n")
        
        f.write("OVERALL PERFORMANCE METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Average Accuracy: {avg_accuracy:.3f}\n")
        f.write(f"Average Precision: {avg_precision:.3f}\n")
        f.write(f"Average Recall: {avg_recall:.3f}\n")
        f.write(f"Average F1-Score: {avg_f1:.3f}\n\n")
        
        f.write("TOP PERFORMING PROTEINS\n")
        f.write("-" * 40 + "\n")
        f.write("Best by Accuracy:\n")
        f.write(f"  Protein: {best_accuracy_protein['protein']}\n")
        f.write(f"  Accuracy: {best_accuracy_protein['accuracy']:.3f}\n")
        f.write(f"  F1-Score: {best_accuracy_protein['f1_score']:.3f}\n")
        f.write(f"  Samples: {best_accuracy_protein['n_samples']} (Active: {best_accuracy_protein['n_active']}, Inactive: {best_accuracy_protein['n_inactive']})\n\n")
        
        f.write("Best by F1-Score:\n")
        f.write(f"  Protein: {best_f1_protein['protein']}\n")
        f.write(f"  F1-Score: {best_f1_protein['f1_score']:.3f}\n")
        f.write(f"  Accuracy: {best_f1_protein['accuracy']:.3f}\n")
        f.write(f"  Samples: {best_f1_protein['n_samples']} (Active: {best_f1_protein['n_active']}, Inactive: {best_f1_protein['n_inactive']})\n\n")
        
        f.write("DETAILED PERFORMANCE ANALYSIS\n")
        f.write("-" * 40 + "\n")
        
        # Performance by fold
        fold_performance = successful_models.groupby('fold').agg({
            'accuracy': 'mean',
            'precision': 'mean',
            'recall': 'mean',
            'f1_score': 'mean'
        }).reset_index()
        
        f.write("Performance by Cross-Validation Fold:\n")
        for _, row in fold_performance.iterrows():
            f.write(f"  Fold {int(row['fold'])}: Accuracy={row['accuracy']:.3f}, F1={row['f1_score']:.3f}\n")
        f.write("\n")
        
        # Class balance analysis
        class_balance = successful_models.groupby('protein').agg({
            'n_active': 'first',
            'n_inactive': 'first'
        }).reset_index()
        
        class_balance['total'] = class_balance['n_active'] + class_balance['n_inactive']
        class_balance['active_ratio'] = class_balance['n_active'] / class_balance['total']
        
        f.write("Class Balance Analysis:\n")
        f.write(f"  Average active ratio: {class_balance['active_ratio'].mean():.3f}\n")
        f.write(f"  Standard deviation: {class_balance['active_ratio'].std():.3f}\n")
        f.write(f"  Most balanced protein: {class_balance.loc[abs(class_balance['active_ratio'] - 0.5).idxmin(), 'protein']}\n")
        f.write(f"  Most imbalanced protein: {class_balance.loc[abs(class_balance['active_ratio'] - 0.5).idxmax(), 'protein']}\n\n")
        
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 40 + "\n")
        f.write("1. Focus on proteins with high F1-scores for reliable predictions\n")
        f.write("2. Consider class balance when interpreting results\n")
        f.write("3. Models with >0.8 accuracy show good performance\n")
        f.write("4. Proteins with balanced classes tend to perform better\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("Report generated on: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        f.write("=" * 80 + "\n")
    
    print(f"Summary report saved to: {report_path}")
    
    # Create performance comparison table
    create_performance_comparison_table(protein_performance)

def create_performance_comparison_table(protein_performance):
    """Create a detailed performance comparison table"""
    
    # Sort by accuracy
    sorted_performance = protein_performance.sort_values('accuracy', ascending=False)
    
    # Create table plot
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    for _, row in sorted_performance.iterrows():
        table_data.append([
            row['protein'],
            f"{row['accuracy']:.3f}",
            f"{row['precision']:.3f}",
            f"{row['recall']:.3f}",
            f"{row['f1_score']:.3f}",
            f"{row['n_samples']}",
            f"{row['n_active']}",
            f"{row['n_inactive']}",
            f"{row['n_active']/(row['n_active']+row['n_inactive']):.3f}"
        ])
    
    # Create table
    table = ax.table(cellText=table_data,
                    colLabels=['Protein', 'Accuracy', 'Precision', 'Recall', 
                             'F1-Score', 'Total', 'Active', 'Inactive', 'Active Ratio'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.12, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08])
    
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(table_data) + 1):
        for j in range(9):
            cell = table[(i, j)]
            if i == 0:  # Header row
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                # Color code based on performance
                if j == 1:  # Accuracy column
                    accuracy = float(table_data[i-1][j])
                    if accuracy > 0.85:
                        cell.set_facecolor('#90EE90')  # Light green
                    elif accuracy > 0.75:
                        cell.set_facecolor('#FFE4B5')  # Light orange
                    else:
                        cell.set_facecolor('#FFB6C1')  # Light red
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    plt.title('Classification Performance Comparison (Sorted by Accuracy)', 
             fontsize=16, fontweight='bold', pad=20)
    
    output_path = "/home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_modelling/data_visualization/plots/classification_performance_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Performance comparison table saved to: {output_path}")

if __name__ == "__main__":
    stats = generate_classification_dashboard()
    
    if stats:
        print("\nClassification Dashboard Summary:")
        print(f"- Classification performance plots generated")
        print(f"- Detailed analysis plots generated")
        print(f"- Summary report generated")
        print(f"- Performance comparison table generated")
        print("\nAll plots and reports saved to: analyses/qsar_papyrus_modelling/data_visualization/plots/")
    else:
        print("Error: Could not generate classification dashboard. Check if classification results exist.") 