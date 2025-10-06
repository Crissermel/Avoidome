#!/usr/bin/env python3
"""
Data Visualization Utilities for Papyrus QSAR Dashboard
Generates and saves plots for bioactivity data and QSAR results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path

# Set style for consistent plots
plt.style.use('default')
sns.set_palette("husl")

class PapyrusVisualizer:
    """Class for generating and saving Papyrus QSAR visualizations"""
    
    def __init__(self, output_dir="analyses/qsar_papyrus_modelling/data_visualization/plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_bioactivity_overview_plots(self, bioactivity_df):
        """Generate bioactivity overview plots"""
        
        # Summary statistics
        total_proteins = len(bioactivity_df)
        proteins_with_data = len(bioactivity_df[bioactivity_df['total_activities'] > 0])
        multi_organism_proteins = len(bioactivity_df[bioactivity_df['num_organisms'] > 1])
        total_activities = bioactivity_df['total_activities'].sum()
        
        # Data availability visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Organism activity counts
        organism_data = {
            'Human': bioactivity_df['human_activities'].sum(),
            'Mouse': bioactivity_df['mouse_activities'].sum(),
            'Rat': bioactivity_df['rat_activities'].sum()
        }
        ax1.bar(organism_data.keys(), organism_data.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax1.set_title('Total Activities by Organism')
        ax1.set_ylabel('Number of Activities')
        
        # Proteins with data per organism
        proteins_with_data_by_org = {
            'Human': len(bioactivity_df[bioactivity_df['human_activities'] > 0]),
            'Mouse': len(bioactivity_df[bioactivity_df['mouse_activities'] > 0]),
            'Rat': len(bioactivity_df[bioactivity_df['rat_activities'] > 0])
        }
        ax2.bar(proteins_with_data_by_org.keys(), proteins_with_data_by_org.values(), 
                color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax2.set_title('Proteins with Data by Organism')
        ax2.set_ylabel('Number of Proteins')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'bioactivity_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'total_proteins': total_proteins,
            'proteins_with_data': proteins_with_data,
            'multi_organism_proteins': multi_organism_proteins,
            'total_activities': total_activities
        }
    
    def generate_bioactivity_points_plots(self, bioactivity_df, min_threshold=100, max_threshold=1000):
        """Generate bioactivity points distribution plots"""
        
        # Filter proteins with data
        proteins_with_data = bioactivity_df[bioactivity_df['total_activities'] > 0]
        
        # Filter data based on threshold
        filtered_data = proteins_with_data[
            (proteins_with_data['total_activities'] >= min_threshold) & 
            (proteins_with_data['total_activities'] <= max_threshold)
        ]
        
        # Bioactivity points distribution
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Sort by total activities
        sorted_data = filtered_data.sort_values('total_activities', ascending=True)
        
        # Create horizontal bar plot
        bars = ax.barh(range(len(sorted_data)), sorted_data['total_activities'], 
                      color='#1f77b4', alpha=0.7)
        
        # Customize the plot
        ax.set_yticks(range(len(sorted_data)))
        ax.set_yticklabels(sorted_data['protein_name'], fontsize=10)
        ax.set_xlabel('Number of Bioactivity Points')
        ax.set_title(f'Bioactivity Points per Protein (Range: {min_threshold}-{max_threshold})')
        
        # Add value labels on bars
        for i, (bar, activity) in enumerate(zip(bars, sorted_data['total_activities'])):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                   str(int(activity)), ha='left', va='center', fontsize=9)
        
        # Add threshold lines
        ax.axvline(x=min_threshold, color='red', linestyle='--', alpha=0.7, label=f'Min Threshold: {min_threshold}')
        ax.axvline(x=max_threshold, color='red', linestyle='--', alpha=0.7, label=f'Max Threshold: {max_threshold}')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'bioactivity_points_{min_threshold}_{max_threshold}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Organism breakdown
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Calculate organism-specific statistics
        organism_stats = {
            'Human': filtered_data['human_activities'].sum(),
            'Mouse': filtered_data['mouse_activities'].sum(),
            'Rat': filtered_data['rat_activities'].sum()
        }
        
        # Pie chart of organism distribution
        activities = [organism_stats['Human'], organism_stats['Mouse'], organism_stats['Rat']]
        labels = ['Human', 'Mouse', 'Rat']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        ax1.pie(activities, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Distribution of Activities by Organism')
        
        # Bar chart of organism counts
        organism_counts = {
            'Human': len(filtered_data[filtered_data['human_activities'] > 0]),
            'Mouse': len(filtered_data[filtered_data['mouse_activities'] > 0]),
            'Rat': len(filtered_data[filtered_data['rat_activities'] > 0])
        }
        
        ax2.bar(organism_counts.keys(), organism_counts.values(), color=colors)
        ax2.set_title('Number of Proteins with Data by Organism')
        ax2.set_ylabel('Number of Proteins')
        
        # Add value labels on bars
        for i, (org, count) in enumerate(organism_counts.items()):
            ax2.text(i, count + 0.1, str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'organism_breakdown_{min_threshold}_{max_threshold}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'proteins_in_range': len(filtered_data),
            'total_activities': filtered_data['total_activities'].sum(),
            'average_activities': filtered_data['total_activities'].mean(),
            'median_activities': filtered_data['total_activities'].median(),
            'organism_stats': organism_stats,
            'organism_counts': organism_counts
        }
    
    def generate_qsar_performance_plots(self, results_df):
        """Generate QSAR performance visualization plots"""
        
        # Calculate mean metrics for each protein
        protein_metrics = results_df.groupby('protein').agg({
            'r2': 'mean',
            'rmse': 'mean', 
            'mae': 'mean',
            'n_samples': 'first'
        }).reset_index()
        protein_metrics.columns = ['protein_name', 'mean_r2', 'mean_rmse', 'mean_mae', 'total_activities']
        
        # Performance visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # R² distribution
        ax1.hist(protein_metrics['mean_r2'], bins=15, alpha=0.7, color='#1f77b4')
        ax1.set_title('Distribution of R² Scores')
        ax1.set_xlabel('R² Score')
        ax1.set_ylabel('Number of Models')
        ax1.axvline(protein_metrics['mean_r2'].mean(), color='red', linestyle='--', 
                    label=f'Mean: {protein_metrics["mean_r2"].mean():.3f}')
        ax1.legend()
        
        # RMSE distribution
        ax2.hist(protein_metrics['mean_rmse'], bins=15, alpha=0.7, color='#ff7f0e')
        ax2.set_title('Distribution of RMSE Scores')
        ax2.set_xlabel('RMSE Score')
        ax2.set_ylabel('Number of Models')
        ax2.axvline(protein_metrics['mean_rmse'].mean(), color='red', linestyle='--', 
                    label=f'Mean: {protein_metrics["mean_rmse"].mean():.3f}')
        ax2.legend()
        
        # MAE distribution
        ax3.hist(protein_metrics['mean_mae'], bins=15, alpha=0.7, color='#2ca02c')
        ax3.set_title('Distribution of MAE Scores')
        ax3.set_xlabel('MAE Score')
        ax3.set_ylabel('Number of Models')
        ax3.axvline(protein_metrics['mean_mae'].mean(), color='red', linestyle='--', 
                    label=f'Mean: {protein_metrics["mean_mae"].mean():.3f}')
        ax3.legend()
        
        # R² vs RMSE scatter
        ax4.scatter(protein_metrics['mean_r2'], protein_metrics['mean_rmse'], alpha=0.6, color='#d62728')
        ax4.set_title('R² vs RMSE Relationship')
        ax4.set_xlabel('R² Score')
        ax4.set_ylabel('RMSE Score')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'qsar_performance_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Top performing models
        top_models = protein_metrics.nlargest(10, 'mean_r2')
        
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(range(len(top_models)), top_models['mean_r2'], color='#1f77b4', alpha=0.7)
        ax.set_yticks(range(len(top_models)))
        ax.set_yticklabels(top_models['protein_name'], fontsize=10)
        ax.set_xlabel('R² Score')
        ax.set_title('Top 10 Performing Models (by R² Score)')
        
        # Add value labels on bars
        for i, (bar, r2) in enumerate(zip(bars, top_models['mean_r2'])):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{r2:.3f}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'top_performing_models.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'total_models': len(protein_metrics),
            'average_r2': protein_metrics['mean_r2'].mean(),
            'average_rmse': protein_metrics['mean_rmse'].mean(),
            'average_mae': protein_metrics['mean_mae'].mean(),
            'top_models': top_models
        }
    
    def generate_protein_details_plots(self, results_df, bioactivity_df, selected_protein):
        """Generate protein-specific detail plots"""
        
        # Calculate mean metrics for the selected protein
        protein_data = results_df[results_df['protein'] == selected_protein]
        bioactivity_data = bioactivity_df[bioactivity_df['protein_name'] == selected_protein]
        
        if len(protein_data) == 0 or len(bioactivity_data) == 0:
            return None
        
        # Cross-validation results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        fold_r2_values = protein_data['r2'].tolist()
        fold_rmse_values = protein_data['rmse'].tolist()
        
        # R² across folds
        ax1.plot(range(1, len(fold_r2_values)+1), fold_r2_values, 'o-', color='#1f77b4', linewidth=2, markersize=8)
        ax1.set_title(f'R² Score Across {len(fold_r2_values)}-Fold CV')
        ax1.set_xlabel('Fold')
        ax1.set_ylabel('R² Score')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # RMSE across folds
        ax2.plot(range(1, len(fold_rmse_values)+1), fold_rmse_values, 'o-', color='#ff7f0e', linewidth=2, markersize=8)
        ax2.set_title(f'RMSE Across {len(fold_rmse_values)}-Fold CV')
        ax2.set_xlabel('Fold')
        ax2.set_ylabel('RMSE Score')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{selected_protein}_cv_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Bioactivity distribution for selected protein
        if bioactivity_data.iloc[0]['total_activities'] > 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            activities = [bioactivity_data.iloc[0]['human_activities'], 
                        bioactivity_data.iloc[0]['mouse_activities'], 
                        bioactivity_data.iloc[0]['rat_activities']]
            organisms = ['Human', 'Mouse', 'Rat']
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            
            bars = ax.bar(organisms, activities, color=colors)
            ax.set_title(f'Bioactivity Distribution for {selected_protein}')
            ax.set_ylabel('Number of Activities')
            
            # Add value labels on bars
            for bar, activity in zip(bars, activities):
                if activity > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           str(activity), ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'{selected_protein}_bioactivity_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        return {
            'protein_metrics': {
                'mean_r2': protein_data['r2'].mean(),
                'mean_rmse': protein_data['rmse'].mean(),
                'mean_mae': protein_data['mae'].mean(),
                'total_activities': bioactivity_data.iloc[0]['total_activities'],
                'human_activities': bioactivity_data.iloc[0]['human_activities'],
                'mouse_activities': bioactivity_data.iloc[0]['mouse_activities'],
                'rat_activities': bioactivity_data.iloc[0]['rat_activities'],
                'organisms_with_data': bioactivity_data.iloc[0]['organisms_with_data'],
                'num_organisms': bioactivity_data.iloc[0]['num_organisms']
            }
        }

    def generate_classification_performance_plots(self, classification_results_df):
        """Generate classification performance plots"""
        
        # Filter successful models (those with accuracy values)
        successful_models = classification_results_df[classification_results_df['accuracy'].notna()]
        
        if len(successful_models) == 0:
            print("No successful classification models found")
            return None
        
        # Get unique proteins with successful models
        proteins_with_models = successful_models['protein'].unique()
        
        # Calculate average metrics per protein
        protein_metrics = []
        for protein in proteins_with_models:
            protein_data = successful_models[successful_models['protein'] == protein]
            avg_metrics = {
                'protein': protein,
                'avg_accuracy': protein_data['accuracy'].mean(),
                'avg_precision': protein_data['precision'].mean(),
                'avg_recall': protein_data['recall'].mean(),
                'avg_f1_score': protein_data['f1_score'].mean(),
                'n_samples': protein_data['n_samples'].iloc[0],
                'n_active': protein_data['n_active'].iloc[0],
                'n_inactive': protein_data['n_inactive'].iloc[0],
                'n_folds': len(protein_data)
            }
            protein_metrics.append(avg_metrics)
        
        metrics_df = pd.DataFrame(protein_metrics)
        
        # Create classification performance plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Accuracy distribution
        accuracy_data = metrics_df['avg_accuracy'].values
        ax1.hist(accuracy_data, bins=15, alpha=0.7, color='#1f77b4', edgecolor='black')
        ax1.axvline(accuracy_data.mean(), color='red', linestyle='--', label=f'Mean: {accuracy_data.mean():.3f}')
        ax1.set_xlabel('Average Accuracy')
        ax1.set_ylabel('Number of Proteins')
        ax1.set_title('Distribution of Classification Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. F1-Score vs Accuracy scatter
        ax2.scatter(metrics_df['avg_accuracy'], metrics_df['avg_f1_score'], 
                   alpha=0.7, s=100, c='#ff7f0e')
        ax2.set_xlabel('Average Accuracy')
        ax2.set_ylabel('Average F1-Score')
        ax2.set_title('Accuracy vs F1-Score')
        ax2.grid(True, alpha=0.3)
        
        # Add protein labels for points with high performance
        for _, row in metrics_df.iterrows():
            if row['avg_accuracy'] > 0.85 and row['avg_f1_score'] > 0.7:
                ax2.annotate(row['protein'], (row['avg_accuracy'], row['avg_f1_score']), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 3. Class balance analysis
        class_balance = []
        for _, row in metrics_df.iterrows():
            total = row['n_active'] + row['n_inactive']
            if total > 0:
                active_ratio = row['n_active'] / total
                class_balance.append({
                    'protein': row['protein'],
                    'active_ratio': active_ratio,
                    'avg_accuracy': row['avg_accuracy'],
                    'total_samples': total
                })
        
        balance_df = pd.DataFrame(class_balance)
        ax3.scatter(balance_df['active_ratio'], balance_df['avg_accuracy'], 
                   alpha=0.7, s=100, c='#2ca02c')
        ax3.set_xlabel('Ratio of Active Compounds')
        ax3.set_ylabel('Average Accuracy')
        ax3.set_title('Class Balance vs Classification Accuracy')
        ax3.grid(True, alpha=0.3)
        
        # 4. Top performing proteins
        top_proteins = metrics_df.nlargest(10, 'avg_accuracy')[['protein', 'avg_accuracy', 'avg_f1_score']]
        bars = ax4.barh(range(len(top_proteins)), top_proteins['avg_accuracy'], 
                       color='#d62728', alpha=0.7)
        ax4.set_yticks(range(len(top_proteins)))
        ax4.set_yticklabels(top_proteins['protein'], fontsize=10)
        ax4.set_xlabel('Average Accuracy')
        ax4.set_title('Top 10 Proteins by Classification Accuracy')
        
        # Add accuracy values on bars
        for i, (bar, accuracy) in enumerate(zip(bars, top_proteins['avg_accuracy'])):
            ax4.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{accuracy:.3f}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'classification_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create detailed metrics table
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        table_data = []
        for _, row in metrics_df.iterrows():
            table_data.append([
                row['protein'],
                f"{row['avg_accuracy']:.3f}",
                f"{row['avg_precision']:.3f}",
                f"{row['avg_recall']:.3f}",
                f"{row['avg_f1_score']:.3f}",
                f"{row['n_samples']}",
                f"{row['n_active']}",
                f"{row['n_inactive']}"
            ])
        
        # Create table
        table = ax.table(cellText=table_data,
                        colLabels=['Protein', 'Avg Accuracy', 'Avg Precision', 'Avg Recall', 
                                 'Avg F1-Score', 'Total Samples', 'Active', 'Inactive'],
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.15, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Style the table
        for i in range(len(table_data) + 1):
            for j in range(8):
                cell = table[(i, j)]
                if i == 0:  # Header row
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        plt.title('Classification Performance Summary', fontsize=16, fontweight='bold', pad=20)
        plt.savefig(self.output_dir / 'classification_summary_table.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'total_proteins': len(classification_results_df['protein'].unique()),
            'successful_models': len(proteins_with_models),
            'avg_accuracy': metrics_df['avg_accuracy'].mean(),
            'avg_f1_score': metrics_df['avg_f1_score'].mean(),
            'best_protein': metrics_df.loc[metrics_df['avg_accuracy'].idxmax(), 'protein'],
            'best_accuracy': metrics_df['avg_accuracy'].max()
        }
    
    def generate_classification_detailed_analysis(self, classification_results_df):
        """Generate detailed classification analysis plots"""
        
        # Filter successful models
        successful_models = classification_results_df[classification_results_df['accuracy'].notna()]
        
        if len(successful_models) == 0:
            return None
        
        # Create detailed analysis plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Precision vs Recall for all folds
        ax1.scatter(successful_models['precision'], successful_models['recall'], 
                   alpha=0.6, s=50, c='#1f77b4')
        ax1.set_xlabel('Precision')
        ax1.set_ylabel('Recall')
        ax1.set_title('Precision vs Recall (All Folds)')
        ax1.grid(True, alpha=0.3)
        
        # Add diagonal line for reference
        ax1.plot([0, 1], [0, 1], 'r--', alpha=0.5)
        
        # 2. Performance by fold
        fold_performance = successful_models.groupby('fold').agg({
            'accuracy': 'mean',
            'precision': 'mean',
            'recall': 'mean',
            'f1_score': 'mean'
        }).reset_index()
        
        x = np.arange(len(fold_performance))
        width = 0.2
        
        ax2.bar(x - width*1.5, fold_performance['accuracy'], width, label='Accuracy', alpha=0.8)
        ax2.bar(x - width*0.5, fold_performance['precision'], width, label='Precision', alpha=0.8)
        ax2.bar(x + width*0.5, fold_performance['recall'], width, label='Recall', alpha=0.8)
        ax2.bar(x + width*1.5, fold_performance['f1_score'], width, label='F1-Score', alpha=0.8)
        
        ax2.set_xlabel('Fold')
        ax2.set_ylabel('Score')
        ax2.set_title('Average Performance by Cross-Validation Fold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'Fold {i}' for i in fold_performance['fold']])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Class distribution analysis
        class_distribution = successful_models.groupby('protein').agg({
            'n_active': 'first',
            'n_inactive': 'first'
        }).reset_index()
        
        class_distribution['total'] = class_distribution['n_active'] + class_distribution['n_inactive']
        class_distribution['active_ratio'] = class_distribution['n_active'] / class_distribution['total']
        
        ax3.hist(class_distribution['active_ratio'], bins=15, alpha=0.7, color='#2ca02c', edgecolor='black')
        ax3.axvline(class_distribution['active_ratio'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {class_distribution["active_ratio"].mean():.3f}')
        ax3.set_xlabel('Ratio of Active Compounds')
        ax3.set_ylabel('Number of Proteins')
        ax3.set_title('Distribution of Class Balance')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Sample size vs Performance
        sample_size_performance = successful_models.groupby('protein').agg({
            'n_samples': 'first',
            'accuracy': 'mean',
            'f1_score': 'mean'
        }).reset_index()
        
        scatter = ax4.scatter(sample_size_performance['n_samples'], sample_size_performance['accuracy'],
                             c=sample_size_performance['f1_score'], cmap='viridis', s=100, alpha=0.7)
        ax4.set_xlabel('Number of Samples')
        ax4.set_ylabel('Average Accuracy')
        ax4.set_title('Sample Size vs Performance')
        ax4.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Average F1-Score')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'classification_detailed_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'total_folds': len(successful_models),
            'avg_fold_accuracy': successful_models['accuracy'].mean(),
            'avg_fold_f1': successful_models['f1_score'].mean(),
            'class_imbalance': class_distribution['active_ratio'].std()
        }

def generate_all_visualizations():
    """Generate all visualizations for the dashboard"""
    
    # Initialize visualizer
    visualizer = PapyrusVisualizer()
    
    # Load data
    bioactivity_path = "/home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_modelling/multi_organism_results.csv"
    results_path = "/home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_modelling/prediction_results.csv"
    classification_path = "/home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_modelling/classification_results.csv"
    
    if not os.path.exists(bioactivity_path) or not os.path.exists(results_path):
        print("Data files not found!")
        return
    
    bioactivity_df = pd.read_csv(bioactivity_path)
    results_df = pd.read_csv(results_path)
    
    # Generate all plots
    print("Generating bioactivity overview plots...")
    bioactivity_stats = visualizer.generate_bioactivity_overview_plots(bioactivity_df)
    
    print("Generating bioactivity points plots...")
    bioactivity_points_stats = visualizer.generate_bioactivity_points_plots(bioactivity_df)
    
    print("Generating QSAR performance plots...")
    qsar_stats = visualizer.generate_qsar_performance_plots(results_df)
    
    print("Generating protein detail plots...")
    # Generate plots for top 5 proteins
    top_proteins = bioactivity_df[bioactivity_df['total_activities'] > 0].nlargest(5, 'total_activities')['protein_name'].tolist()
    for protein in top_proteins:
        visualizer.generate_protein_details_plots(results_df, bioactivity_df, protein)
    
    # Generate classification plots if classification results exist
    if os.path.exists(classification_path):
        print("Generating classification performance plots...")
        classification_df = pd.read_csv(classification_path)
        classification_stats = visualizer.generate_classification_performance_plots(classification_df)
        
        print("Generating classification detailed analysis...")
        classification_detailed_stats = visualizer.generate_classification_detailed_analysis(classification_df)
        
        print(f"All visualizations saved to: {visualizer.output_dir}")
        
        return {
            'bioactivity_stats': bioactivity_stats,
            'bioactivity_points_stats': bioactivity_points_stats,
            'qsar_stats': qsar_stats,
            'classification_stats': classification_stats,
            'classification_detailed_stats': classification_detailed_stats
        }
    else:
        print("Classification results not found, skipping classification plots...")
        print(f"All visualizations saved to: {visualizer.output_dir}")
        
        return {
            'bioactivity_stats': bioactivity_stats,
            'bioactivity_points_stats': bioactivity_points_stats,
            'qsar_stats': qsar_stats
        }

if __name__ == "__main__":
    generate_all_visualizations() 