#!/usr/bin/env python3
"""
QSAR Model Results Plotting Script

This script generates comprehensive plots for QSAR modeling results including:
- Model performance distributions
- Cross-organism comparisons
- Sample size analysis
- Performance heatmaps
- Feature importance analysis

Usage:
    python plot_qsar_results.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
from datetime import datetime

# Set style
plt.style.use('default')
sns.set_palette("husl")

def load_data():
    """Load QSAR modeling data"""
    morgan_path = "analyses/standardized_qsar_models/modeling_summary.csv"
    esm_path = "analyses/standardized_qsar_models/esm_morgan_modeling_summary.csv"
    
    morgan_df = pd.read_csv(morgan_path)
    esm_df = pd.read_csv(esm_path)
    
    # Add model type labels
    morgan_df['model_type'] = 'Morgan'
    esm_df['model_type'] = 'ESM+Morgan'
    
    # Combine data
    combined_df = pd.concat([morgan_df, esm_df], ignore_index=True)
    
    return combined_df, morgan_df, esm_df

def create_output_directory():
    """Create output directory for plots"""
    output_dir = Path("analyses/standardized_qsar_models/plots")
    output_dir.mkdir(exist_ok=True)
    return output_dir

def plot_model_distribution(data_df, output_dir):
    """Plot model distribution by type and organism"""
    completed_df = data_df[data_df['status'] == 'completed'].copy()
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('QSAR Model Distribution Analysis', fontsize=16, fontweight='bold')
    
    # 1. Model count by organism
    org_counts = completed_df['organism'].value_counts()
    axes[0, 0].pie(org_counts.values, labels=org_counts.index, autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('Model Distribution by Organism')
    
    # 2. Model count by type
    type_counts = completed_df['model_type'].value_counts()
    axes[0, 1].bar(type_counts.index, type_counts.values, color=['#1f77b4', '#ff7f0e'])
    axes[0, 1].set_title('Model Distribution by Type')
    axes[0, 1].set_ylabel('Number of Models')
    
    # 3. Model count by organism and type
    cross_tab = pd.crosstab(completed_df['organism'], completed_df['model_type'])
    cross_tab.plot(kind='bar', ax=axes[1, 0], color=['#1f77b4', '#ff7f0e'])
    axes[1, 0].set_title('Model Count by Organism and Type')
    axes[1, 0].set_ylabel('Number of Models')
    axes[1, 0].legend(title='Model Type')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Sample size distribution by organism
    sample_data = completed_df.groupby('organism')['n_samples'].agg(['mean', 'std']).reset_index()
    axes[1, 1].bar(sample_data['organism'], sample_data['mean'], 
                   yerr=sample_data['std'], capsize=5, color=['#2ca02c', '#d62728', '#9467bd'])
    axes[1, 1].set_title('Average Sample Size by Organism')
    axes[1, 1].set_ylabel('Average Number of Samples')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_performance_metrics(data_df, output_dir):
    """Plot performance metrics distributions"""
    completed_df = data_df[data_df['status'] == 'completed'].copy()
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('QSAR Model Performance Metrics', fontsize=16, fontweight='bold')
    
    # 1. R² distribution by model type
    reg_data = completed_df[completed_df['regression_r2'].notna()]
    if len(reg_data) > 0:
        for model_type in reg_data['model_type'].unique():
            type_data = reg_data[reg_data['model_type'] == model_type]
            axes[0, 0].hist(type_data['regression_r2'], alpha=0.7, label=model_type, bins=20)
        axes[0, 0].set_title('R² Score Distribution by Model Type')
        axes[0, 0].set_xlabel('R² Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
    
    # 2. RMSE distribution by model type
    if len(reg_data) > 0:
        for model_type in reg_data['model_type'].unique():
            type_data = reg_data[reg_data['model_type'] == model_type]
            axes[0, 1].hist(type_data['regression_rmse'], alpha=0.7, label=model_type, bins=20)
        axes[0, 1].set_title('RMSE Distribution by Model Type')
        axes[0, 1].set_xlabel('RMSE')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
    
    # 3. Accuracy distribution by model type
    class_data = completed_df[completed_df['classification_accuracy'].notna()]
    if len(class_data) > 0:
        for model_type in class_data['model_type'].unique():
            type_data = class_data[class_data['model_type'] == model_type]
            axes[0, 2].hist(type_data['classification_accuracy'], alpha=0.7, label=model_type, bins=20)
        axes[0, 2].set_title('Accuracy Distribution by Model Type')
        axes[0, 2].set_xlabel('Accuracy')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].legend()
    
    # 4. F1 score distribution by model type
    if len(class_data) > 0:
        for model_type in class_data['model_type'].unique():
            type_data = class_data[class_data['model_type'] == model_type]
            axes[1, 0].hist(type_data['classification_f1'], alpha=0.7, label=model_type, bins=20)
        axes[1, 0].set_title('F1 Score Distribution by Model Type')
        axes[1, 0].set_xlabel('F1 Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
    
    # 5. AUC distribution by model type
    if len(class_data) > 0:
        for model_type in class_data['model_type'].unique():
            type_data = class_data[class_data['model_type'] == model_type]
            axes[1, 1].hist(type_data['classification_auc'], alpha=0.7, label=model_type, bins=20)
        axes[1, 1].set_title('AUC Distribution by Model Type')
        axes[1, 1].set_xlabel('AUC')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
    
    # 6. Sample size vs R² correlation
    if len(reg_data) > 0:
        scatter = axes[1, 2].scatter(reg_data['n_samples'], reg_data['regression_r2'], 
                                   c=reg_data['model_type'].map({'Morgan': 0, 'ESM+Morgan': 1}), 
                                   cmap='viridis', alpha=0.7)
        axes[1, 2].set_title('Sample Size vs R² Score')
        axes[1, 2].set_xlabel('Number of Samples')
        axes[1, 2].set_ylabel('R² Score')
        cbar = plt.colorbar(scatter, ax=axes[1, 2])
        cbar.set_label('Model Type')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_organism_comparison(data_df, output_dir):
    """Plot cross-organism performance comparison"""
    completed_df = data_df[data_df['status'] == 'completed'].copy()
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Cross-Organism Performance Comparison', fontsize=16, fontweight='bold')
    
    # 1. R² by organism and model type
    reg_data = completed_df[completed_df['regression_r2'].notna()]
    if len(reg_data) > 0:
        sns.boxplot(data=reg_data, x='organism', y='regression_r2', hue='model_type', ax=axes[0, 0])
        axes[0, 0].set_title('R² Score by Organism and Model Type')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. RMSE by organism and model type
    if len(reg_data) > 0:
        sns.boxplot(data=reg_data, x='organism', y='regression_rmse', hue='model_type', ax=axes[0, 1])
        axes[0, 1].set_title('RMSE by Organism and Model Type')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Accuracy by organism and model type
    class_data = completed_df[completed_df['classification_accuracy'].notna()]
    if len(class_data) > 0:
        sns.boxplot(data=class_data, x='organism', y='classification_accuracy', hue='model_type', ax=axes[1, 0])
        axes[1, 0].set_title('Accuracy by Organism and Model Type')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. F1 score by organism and model type
    if len(class_data) > 0:
        sns.boxplot(data=class_data, x='organism', y='classification_f1', hue='model_type', ax=axes[1, 1])
        axes[1, 1].set_title('F1 Score by Organism and Model Type')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'organism_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_performance_heatmap(data_df, output_dir):
    """Plot performance heatmap"""
    completed_df = data_df[data_df['status'] == 'completed'].copy()
    
    # Create R² heatmap
    reg_data = completed_df[completed_df['regression_r2'].notna()]
    if len(reg_data) > 0:
        # Pivot table for heatmap
        r2_pivot = reg_data.pivot_table(values='regression_r2', index='protein_name', 
                                       columns='organism', aggfunc='mean')
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(r2_pivot.fillna(0), annot=True, cmap='RdYlBu_r', center=0.5, 
                   cbar_kws={'label': 'R² Score'})
        plt.title('R² Performance Heatmap by Protein and Organism', fontsize=14, fontweight='bold')
        plt.xlabel('Organism')
        plt.ylabel('Protein')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_dir / 'r2_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create accuracy heatmap
    class_data = completed_df[completed_df['classification_accuracy'].notna()]
    if len(class_data) > 0:
        acc_pivot = class_data.pivot_table(values='classification_accuracy', index='protein_name', 
                                          columns='organism', aggfunc='mean')
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(acc_pivot.fillna(0), annot=True, cmap='RdYlGn', center=0.8, 
                   cbar_kws={'label': 'Accuracy'})
        plt.title('Accuracy Performance Heatmap by Protein and Organism', fontsize=14, fontweight='bold')
        plt.xlabel('Organism')
        plt.ylabel('Protein')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_dir / 'accuracy_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

def plot_top_performers(data_df, output_dir):
    """Plot top performing models"""
    completed_df = data_df[data_df['status'] == 'completed'].copy()
    
    # Top regression performers
    reg_data = completed_df[completed_df['regression_r2'].notna()].copy()
    if len(reg_data) > 0:
        top_reg = reg_data.nlargest(15, 'regression_r2')
        
        plt.figure(figsize=(12, 8))
        colors = ['#1f77b4' if x == 'Morgan' else '#ff7f0e' for x in top_reg['model_type']]
        bars = plt.barh(range(len(top_reg)), top_reg['regression_r2'], color=colors)
        plt.yticks(range(len(top_reg)), [f"{row['protein_name']} ({row['organism']})" 
                                        for _, row in top_reg.iterrows()])
        plt.xlabel('R² Score')
        plt.title('Top 15 Regression Models by R² Score', fontsize=14, fontweight='bold')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#1f77b4', label='Morgan'),
                          Patch(facecolor='#ff7f0e', label='ESM+Morgan')]
        plt.legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'top_regression_models.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Top classification performers
    class_data = completed_df[completed_df['classification_accuracy'].notna()].copy()
    if len(class_data) > 0:
        top_class = class_data.nlargest(15, 'classification_accuracy')
        
        plt.figure(figsize=(12, 8))
        colors = ['#1f77b4' if x == 'Morgan' else '#ff7f0e' for x in top_class['model_type']]
        bars = plt.barh(range(len(top_class)), top_class['classification_accuracy'], color=colors)
        plt.yticks(range(len(top_class)), [f"{row['protein_name']} ({row['organism']})" 
                                          for _, row in top_class.iterrows()])
        plt.xlabel('Accuracy')
        plt.title('Top 15 Classification Models by Accuracy', fontsize=14, fontweight='bold')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#1f77b4', label='Morgan'),
                          Patch(facecolor='#ff7f0e', label='ESM+Morgan')]
        plt.legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'top_classification_models.png', dpi=300, bbox_inches='tight')
        plt.close()

def plot_sample_size_analysis(data_df, output_dir):
    """Plot sample size analysis"""
    completed_df = data_df[data_df['status'] == 'completed'].copy()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Sample Size Analysis', fontsize=16, fontweight='bold')
    
    # 1. Sample size distribution by organism
    for organism in completed_df['organism'].unique():
        org_data = completed_df[completed_df['organism'] == organism]
        axes[0, 0].hist(org_data['n_samples'], alpha=0.7, label=organism, bins=20)
    axes[0, 0].set_title('Sample Size Distribution by Organism')
    axes[0, 0].set_xlabel('Number of Samples')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    
    # 2. Sample size vs R² correlation
    reg_data = completed_df[completed_df['regression_r2'].notna()]
    if len(reg_data) > 0:
        scatter = axes[0, 1].scatter(reg_data['n_samples'], reg_data['regression_r2'], 
                                   c=reg_data['organism'].map({'human': 0, 'mouse': 1, 'rat': 2}), 
                                   cmap='viridis', alpha=0.7)
        axes[0, 1].set_title('Sample Size vs R² Score')
        axes[0, 1].set_xlabel('Number of Samples')
        axes[0, 1].set_ylabel('R² Score')
        cbar = plt.colorbar(scatter, ax=axes[0, 1])
        cbar.set_label('Organism')
    
    # 3. Sample size vs accuracy correlation
    class_data = completed_df[completed_df['classification_accuracy'].notna()]
    if len(class_data) > 0:
        scatter = axes[1, 0].scatter(class_data['n_samples'], class_data['classification_accuracy'], 
                                    c=class_data['organism'].map({'human': 0, 'mouse': 1, 'rat': 2}), 
                                    cmap='viridis', alpha=0.7)
        axes[1, 0].set_title('Sample Size vs Accuracy')
        axes[1, 0].set_xlabel('Number of Samples')
        axes[1, 0].set_ylabel('Accuracy')
        cbar = plt.colorbar(scatter, ax=axes[1, 0])
        cbar.set_label('Organism')
    
    # 4. Average sample size by organism and model type
    sample_stats = completed_df.groupby(['organism', 'model_type'])['n_samples'].agg(['mean', 'std']).reset_index()
    
    # Create a pivot table for easier plotting
    pivot_mean = sample_stats.pivot(index='organism', columns='model_type', values='mean')
    pivot_std = sample_stats.pivot(index='organism', columns='model_type', values='std')
    
    x_pos = np.arange(len(pivot_mean.index))
    width = 0.35
    colors = ['#1f77b4', '#ff7f0e']
    
    for i, model_type in enumerate(pivot_mean.columns):
        means = pivot_mean[model_type].values
        stds = pivot_std[model_type].values
        axes[1, 1].bar(x_pos + i*width, means, width, 
                      yerr=stds, capsize=5, label=model_type, color=colors[i])
    
    axes[1, 1].set_title('Average Sample Size by Organism and Model Type')
    axes[1, 1].set_xlabel('Organism')
    axes[1, 1].set_ylabel('Average Number of Samples')
    axes[1, 1].set_xticks(x_pos + width/2)
    axes[1, 1].set_xticklabels(pivot_mean.index)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sample_size_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_interactive_plots(data_df, output_dir):
    """Create interactive Plotly plots"""
    completed_df = data_df[data_df['status'] == 'completed'].copy()
    
    # Interactive performance comparison
    reg_data = completed_df[completed_df['regression_r2'].notna()]
    if len(reg_data) > 0:
        fig = px.box(reg_data, x='organism', y='regression_r2', color='model_type',
                    title='R² Score Distribution by Organism and Model Type',
                    labels={'regression_r2': 'R² Score', 'organism': 'Organism', 'model_type': 'Model Type'})
        fig.write_html(str(output_dir / 'interactive_r2_comparison.html'))
    
    # Interactive sample size vs performance
    if len(reg_data) > 0:
        fig = px.scatter(reg_data, x='n_samples', y='regression_r2', color='organism', 
                        size='classification_accuracy', hover_data=['protein_name', 'model_type'],
                        title='Sample Size vs R² Score (Size = Accuracy)',
                        labels={'n_samples': 'Number of Samples', 'regression_r2': 'R² Score'})
        fig.write_html(str(output_dir / 'interactive_sample_vs_r2.html'))
    
    # Interactive model count by organism
    org_counts = completed_df.groupby(['organism', 'model_type']).size().reset_index(name='count')
    fig = px.bar(org_counts, x='organism', y='count', color='model_type',
                title='Model Count by Organism and Type',
                labels={'count': 'Number of Models', 'organism': 'Organism', 'model_type': 'Model Type'})
    fig.write_html(str(output_dir / 'interactive_model_counts.html'))

def generate_summary_report(data_df, output_dir):
    """Generate a summary report of the plots"""
    completed_df = data_df[data_df['status'] == 'completed'].copy()
    
    report = f"""
# QSAR Model Results Visualization Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Location:** {output_dir}

## Summary Statistics

### Model Counts
- **Total Completed Models:** {len(completed_df)}
- **Morgan Models:** {len(completed_df[completed_df['model_type'] == 'Morgan'])}
- **ESM+Morgan Models:** {len(completed_df[completed_df['model_type'] == 'ESM+Morgan'])}

### Organism Distribution
- **Human:** {len(completed_df[completed_df['organism'] == 'human'])} models
- **Mouse:** {len(completed_df[completed_df['organism'] == 'mouse'])} models
- **Rat:** {len(completed_df[completed_df['organism'] == 'rat'])} models

### Performance Metrics
"""
    
    reg_data = completed_df[completed_df['regression_r2'].notna()]
    if len(reg_data) > 0:
        report += f"""
#### Regression Performance
- **Average R²:** {reg_data['regression_r2'].mean():.3f} ± {reg_data['regression_r2'].std():.3f}
- **Average RMSE:** {reg_data['regression_rmse'].mean():.3f} ± {reg_data['regression_rmse'].std():.3f}
- **Best R²:** {reg_data['regression_r2'].max():.3f} ({reg_data.loc[reg_data['regression_r2'].idxmax(), 'protein_name']})
"""
    
    class_data = completed_df[completed_df['classification_accuracy'].notna()]
    if len(class_data) > 0:
        report += f"""
#### Classification Performance
- **Average Accuracy:** {class_data['classification_accuracy'].mean():.3f} ± {class_data['classification_accuracy'].std():.3f}
- **Average F1:** {class_data['classification_f1'].mean():.3f} ± {class_data['classification_f1'].std():.3f}
- **Average AUC:** {class_data['classification_auc'].mean():.3f} ± {class_data['classification_auc'].std():.3f}
- **Best Accuracy:** {class_data['classification_accuracy'].max():.3f} ({class_data.loc[class_data['classification_accuracy'].idxmax(), 'protein_name']})
"""
    
    report += f"""
## Generated Plots

### Static Plots (PNG)
1. **model_distribution.png** - Model distribution analysis
2. **performance_metrics.png** - Performance metrics distributions
3. **organism_comparison.png** - Cross-organism performance comparison
4. **r2_heatmap.png** - R² performance heatmap by protein and organism
5. **accuracy_heatmap.png** - Accuracy performance heatmap by protein and organism
6. **top_regression_models.png** - Top 15 regression models by R² score
7. **top_classification_models.png** - Top 15 classification models by accuracy
8. **sample_size_analysis.png** - Sample size analysis and correlations

### Interactive Plots (HTML)
1. **interactive_r2_comparison.html** - Interactive R² comparison
2. **interactive_sample_vs_r2.html** - Interactive sample size vs R² analysis
3. **interactive_model_counts.html** - Interactive model count visualization

## Usage
- Open PNG files for static visualizations
- Open HTML files in a web browser for interactive plots
- All plots are high-resolution (300 DPI) and suitable for publications

---
*Report generated by plot_qsar_results.py*
"""
    
    with open(output_dir / 'plotting_report.md', 'w') as f:
        f.write(report)

def main():
    """Main function to generate all plots"""
    print("QSAR Model Results Plotting")
    print("=" * 40)
    
    # Load data
    print("Loading data...")
    data_df, morgan_df, esm_df = load_data()
    print(f"Loaded {len(data_df)} total records")
    print(f"Completed models: {len(data_df[data_df['status'] == 'completed'])}")
    
    # Create output directory
    output_dir = create_output_directory()
    print(f"Output directory: {output_dir}")
    
    # Generate plots
    print("\nGenerating plots...")
    
    print("1. Model distribution plots...")
    plot_model_distribution(data_df, output_dir)
    
    print("2. Performance metrics plots...")
    plot_performance_metrics(data_df, output_dir)
    
    print("3. Organism comparison plots...")
    plot_organism_comparison(data_df, output_dir)
    
    print("4. Performance heatmaps...")
    plot_performance_heatmap(data_df, output_dir)
    
    print("5. Top performers plots...")
    plot_top_performers(data_df, output_dir)
    
    print("6. Sample size analysis plots...")
    plot_sample_size_analysis(data_df, output_dir)
    
    print("7. Interactive plots...")
    create_interactive_plots(data_df, output_dir)
    
    print("8. Generating summary report...")
    generate_summary_report(data_df, output_dir)
    
    print(f"\nAll plots generated successfully!")
    print(f"Check the output directory: {output_dir}")
    print("=" * 40)

if __name__ == "__main__":
    main()