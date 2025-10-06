#!/usr/bin/env python3
"""
QSAR Modeling Dashboard for Avoidome

date: 11/09/2025

A comprehensive dashboard displaying QSAR modeling results including:
- Number of samples for each protein and organism
- Bioactivity data statistics
- Model performance metrics
- Cross-organism analysis
- Model comparison across different approaches

Usage:
    streamlit run qsar_modeling_dashboard.py
"""

import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="QSAR Modeling Dashboard - Avoidome",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .protein-info {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .success-metric {
        color: #28a745;
        font-weight: bold;
    }
    .warning-metric {
        color: #ffc107;
        font-weight: bold;
    }
    .error-metric {
        color: #dc3545;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Navigation structure
PAGES = {
    "Introduction": {
        "Avoidome List Introduction": "introduction"
    },
    "QSAR Standard": {
        "Model Overview": "qsar_model_overview",
        "Summary Statistics": "qsar_summary_statistics",
        "Model Performance": "qsar_model_performance",
        "Cross-Organism Comparison": "qsar_cross_organism",
        "Sample Size Analysis": "qsar_sample_size_analysis",
        "Embedding Comparison": "qsar_embedding_comparison"
    },
    "AQSE Analysis": {
        "AQSE Workflow": "aqse_workflow",
        "AQSE Models": "aqse_models",
        "Threshold Analysis": "threshold_analysis",
        "Sample Expansion": "sample_expansion"
    },
    "Model Failure Analysis": {
        "Failure Summary": "failure_summary",
      "Failure Causes": "failure_causes"
  },
  "Discussions": {
      "Presentation Discussion": "presentation_discussion"
  },
  "Visualization": {
        "Distribution Plots": "plot_distribution",
        "Performance Plots": "plot_performance", 
        "Performance Heatmaps": "plot_heatmaps",
        "Top Performers": "plot_top_performers",
        "Interactive Plots": "plot_interactive"
    }
}

# Data loading functions
@st.cache_data
def load_morgan_summary():
    """Load Morgan models summary"""
    path = "analyses/standardized_qsar_models/modeling_summary.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def load_esm_morgan_summary():
    """Load ESM+Morgan models summary"""
    path = "analyses/standardized_qsar_models/esm_morgan_modeling_summary.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def load_alphafold_morgan_summary():
    """Load AlphaFold+Morgan models summary"""
    # Load regression results
    reg_path = "analyses/standardized_qsar_models/af_morgan_regression/regression_results.json"
    class_path = "analyses/standardized_qsar_models/af_morgan_classification/classification_results.json"
    
    results = []
    
    if os.path.exists(reg_path):
        with open(reg_path, 'r') as f:
            reg_data = json.load(f)
        
        for item in reg_data:
            results.append({
                'uniprot_id': item['uniprot_id'],
                'protein_name': item['protein_name'],
                'organism': 'human',  # AlphaFold models are human-specific
                'model_type': 'AlphaFold+Morgan',
                'model_subtype': 'Regression',
                'status': 'completed',
                'n_samples': item['n_samples'],
                'regression_r2': item['cv_scores']['r2'],
                'regression_rmse': np.sqrt(item['cv_scores']['mse']),
                'regression_mae': item['cv_scores']['mae'],
                'classification_accuracy': None,
                'classification_f1': None,
                'classification_auc': None
            })
    
    if os.path.exists(class_path):
        with open(class_path, 'r') as f:
            class_data = json.load(f)
        
        for item in class_data:
            results.append({
                'uniprot_id': item['uniprot_id'],
                'protein_name': item['protein_name'],
                'organism': 'human',  # AlphaFold models are human-specific
                'model_type': 'AlphaFold+Morgan',
                'model_subtype': 'Classification',
                'status': 'completed',
                'n_samples': item['n_samples'],
                'regression_r2': None,
                'regression_rmse': None,
                'regression_mae': None,
                'classification_accuracy': item['cv_scores']['accuracy'],
                'classification_f1': item['cv_scores']['f1'],
                'classification_auc': item['cv_scores']['auc'] if 'auc' in item['cv_scores'] and not pd.isna(item['cv_scores']['auc']) else None
            })
    
    if results:
        return pd.DataFrame(results)
    return None

@st.cache_data
def load_protein_list():
    """Load protein list with UniProt IDs"""
    path = "processed_data/papyrus_protein_check_results.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def get_model_counts():
    """Get counts of models by type and organism"""
    base_path = Path("analyses/standardized_qsar_models")
    counts = {}
    
    for model_type in ["morgan_regression", "morgan_classification", "esm_morgan_regression", "esm_morgan_classification"]:
        counts[model_type] = {}
        for organism in ["human", "mouse", "rat"]:
            model_path = base_path / model_type / organism
            if model_path.exists():
                model_files = list(model_path.glob("*/model.pkl"))
                counts[model_type][organism] = len(model_files)
            else:
                counts[model_type][organism] = 0
    
    return counts

@st.cache_data
def load_sample_data():
    """Load sample data from summary files"""
    morgan_df = load_morgan_summary()
    esm_df = load_esm_morgan_summary()
    alphafold_df = load_alphafold_morgan_summary()
    
    dataframes = []
    
    if morgan_df is not None:
        morgan_df['model_type'] = 'Morgan'
        # Add model subtype
        morgan_reg = morgan_df.copy()
        morgan_reg['model_subtype'] = 'Regression'
        morgan_class = morgan_df.copy()
        morgan_class['model_subtype'] = 'Classification'
        dataframes.extend([morgan_reg, morgan_class])
    
    if esm_df is not None:
        esm_df['model_type'] = 'ESM+Morgan'
        # Add model subtype
        esm_reg = esm_df.copy()
        esm_reg['model_subtype'] = 'Regression'
        esm_class = esm_df.copy()
        esm_class['model_subtype'] = 'Classification'
        dataframes.extend([esm_reg, esm_class])
    
    if alphafold_df is not None:
        dataframes.append(alphafold_df)
    
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        return combined_df
    
    return None

@st.cache_data
def get_available_plots():
    """Get list of available plot files"""
    plots_dir = Path("analyses/standardized_qsar_models/plots")
    if not plots_dir.exists():
        return {}, {}
    
    png_files = {}
    html_files = {}
    
    # Static plots
    static_plots = {
        'model_distribution.png': 'Model Distribution Analysis',
        'performance_metrics.png': 'Performance Metrics Distributions',
        'organism_comparison.png': 'Cross-Organism Performance Comparison',
        'r2_heatmap.png': 'R² Performance Heatmap',
        'accuracy_heatmap.png': 'Accuracy Performance Heatmap',
        'top_regression_models.png': 'Top Regression Models',
        'top_classification_models.png': 'Top Classification Models',
        'sample_size_analysis.png': 'Sample Size Analysis'
    }
    
    for filename, description in static_plots.items():
        file_path = plots_dir / filename
        if file_path.exists():
            png_files[filename] = description
    
    # Interactive plots
    interactive_plots = {
        'interactive_r2_comparison.html': 'Interactive R² Comparison',
        'interactive_sample_vs_r2.html': 'Interactive Sample Size vs R²',
        'interactive_model_counts.html': 'Interactive Model Counts'
    }
    
    for filename, description in interactive_plots.items():
        file_path = plots_dir / filename
        if file_path.exists():
            html_files[filename] = description
    
    return png_files, html_files

# MCC Analysis data loading functions
@st.cache_data
def load_mcc_analysis_data():
    """Load MCC analysis results"""
    mcc_dir = Path("analyses/mcc_comparison/results")
    
    if not mcc_dir.exists():
        return None, None, None
    
    # Load summary data
    summary_path = mcc_dir / "mcc_analysis_summary.csv"
    morgan_detailed_path = mcc_dir / "morgan_comparison_detailed.csv"
    esm_detailed_path = mcc_dir / "esm_morgan_comparison_detailed.csv"
    
    summary_df = None
    morgan_df = None
    esm_df = None
    
    if summary_path.exists():
        summary_df = pd.read_csv(summary_path)
    
    if morgan_detailed_path.exists():
        morgan_df = pd.read_csv(morgan_detailed_path)
    
    if esm_detailed_path.exists():
        esm_df = pd.read_csv(esm_detailed_path)
    
    return summary_df, morgan_df, esm_df

@st.cache_data
def load_mcc_plots():
    """Load MCC analysis plots"""
    plots_dir = Path("analyses/mcc_comparison/plots")
    
    if not plots_dir.exists():
        return {}
    
    plot_files = {}
    
    # MCC analysis plots
    mcc_plots = {
        'mcc_difference_distributions.png': 'MCC Difference Distributions',
        'mcc_scatter_comparison.png': 'MCC Scatter Plot Comparison',
        'win_rate_comparison.png': 'Model Win Rate Comparison'
    }
    
    for filename, description in mcc_plots.items():
        file_path = plots_dir / filename
        if file_path.exists():
            plot_files[filename] = description
    
    return plot_files

@st.cache_data
def load_aqse_results():
    """Load AQSE model results"""
    csv_path = "analyses/avoidome_qsar_similarity_expansion/04_qsar_models_temp/results/aqse_model_results.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        
        # Fix the similar_proteins count by counting actual similar proteins
        if 'similar_proteins' in df.columns:
            # Convert string representation of list to actual list and count
            def count_similar_proteins(similar_proteins_str):
                if pd.isna(similar_proteins_str) or similar_proteins_str == '':
                    return 0
                try:
                    # Remove quotes and brackets, split by comma, count non-empty elements
                    cleaned = similar_proteins_str.strip("[]'\"")
                    if cleaned == '':
                        return 0
                    # Split by comma and count non-empty elements
                    proteins = [p.strip().strip("'\"") for p in cleaned.split(',')]
                    return len([p for p in proteins if p != ''])
                except:
                    return 0
            
            df['n_similar_proteins'] = df['similar_proteins'].apply(count_similar_proteins)
        
        return df
    return None

@st.cache_data
def load_standard_qsar_results():
    """Load standard QSAR model results for comparison - Human ESM+Morgan regression only"""
    import json
    from pathlib import Path
    
    # Load human ESM+Morgan regression results from individual JSON files
    human_dir = Path("analyses/standardized_qsar_models/esm_morgan_regression/human")
    
    if not human_dir.exists():
        return None
    
    results = []
    
    # Iterate through all protein directories
    for protein_dir in human_dir.iterdir():
        if protein_dir.is_dir():
            results_file = protein_dir / "results.json"
            if results_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        data = json.load(f)
                    
                    # Extract the metrics we need
                    results.append({
                        'protein_name': data['protein_name'],
                        'organism': data['organism'],
                        'uniprot_id': data['uniprot_id'],
                        'status': 'completed',
                        'n_samples': data['n_samples'],
                        'regression_r2': data['overall_metrics']['cv_r2'],
                        'regression_rmse': data['overall_metrics']['cv_rmse'],
                        'regression_mae': data['overall_metrics']['cv_mae'],
                        'model_type': 'Standard ESM+Morgan',
                        'model_subtype': 'Regression'
                    })
                except Exception as e:
                    print(f"Error loading {results_file}: {e}")
                    continue
    
    if results:
        return pd.DataFrame(results)
    return None

@st.cache_data
def load_protein_mapping():
    """Load protein name to UniProt ID mapping"""
    mapping_path = "analyses/qsar_papyrus_esm_emb/data_overview_results.csv"
    if os.path.exists(mapping_path):
        return pd.read_csv(mapping_path)
    return None

def create_summary_metrics(data_df):
    """Create summary metrics cards"""
    if data_df is None:
        return
    
    col1, col2, col3 = st.columns(3)
    

    
    with col1:
        total_proteins = data_df['protein_name'].nunique()
        st.metric("Unique Proteins", total_proteins, help="Proteins with at least one successful model")
    
    with col2:
        total_samples = data_df[data_df['status'] == 'completed']['n_samples'].sum()
        st.metric("Total Samples", f"{total_samples:,}", help="Total bioactivity data points")
    
    with col3:
        avg_r2 = data_df[data_df['status'] == 'completed']['regression_r2'].mean()
        st.metric("Avg R² Score", f"{avg_r2:.3f}", help="Average regression performance")

def create_organism_distribution_chart(data_df):
    """Create organism distribution chart"""
    if data_df is None:
        return
    
    completed_df = data_df[data_df['status'] == 'completed']
    
    # Count by organism
    org_counts = completed_df['organism'].value_counts()
    
    fig = px.pie(
        values=org_counts.values,
        names=org_counts.index,
        title="Model Distribution by Organism",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

def create_model_type_distribution(data_df):
    """Create model type distribution chart"""
    if data_df is None:
        return
    
    completed_df = data_df[data_df['status'] == 'completed']
    
    # Count by model type and subtype
    model_counts = completed_df.groupby(['model_type', 'model_subtype']).size().reset_index(name='count')
    
    fig = px.bar(
        model_counts,
        x='model_type',
        y='count',
        color='model_subtype',
        title="Model Distribution by Type",
        barmode='group',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    fig.update_layout(xaxis_title="Model Type", yaxis_title="Number of Models")
    st.plotly_chart(fig, use_container_width=True)

def create_performance_heatmap(data_df):
    """Create performance heatmap"""
    if data_df is None:
        return
    
    completed_df = data_df[data_df['status'] == 'completed'].copy()
    
    if len(completed_df) == 0:
        st.warning("No completed models to display")
        return
    
    # Create pivot table for R² scores
    r2_pivot = completed_df.pivot_table(
        values='regression_r2',
        index='protein_name',
        columns='organism',
        aggfunc='mean'
    )
    
    # Create heatmap
    fig = px.imshow(
        r2_pivot.fillna(0),
        title="R² Performance Heatmap by Protein and Organism",
        color_continuous_scale='RdYlBu_r',
        aspect='auto'
    )
    
    fig.update_layout(
        xaxis_title="Organism",
        yaxis_title="Protein",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_sample_size_analysis(data_df):
    """Create sample size analysis"""
    if data_df is None:
        return
    
    completed_df = data_df[data_df['status'] == 'completed'].copy()
    
    if len(completed_df) == 0:
        st.warning("No completed models to display")
        return
    
    # Sample size distribution
    fig = px.histogram(
        completed_df,
        x='n_samples',
        color='organism',
        title="Sample Size Distribution by Organism",
        nbins=20,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_layout(
        xaxis_title="Number of Samples",
        yaxis_title="Count",
        barmode='overlay'
    )
    
    # Set opacity on traces instead of layout
    for trace in fig.data:
        trace.opacity = 0.7
    
    st.plotly_chart(fig, use_container_width=True)

def create_performance_comparison(data_df):
    """Create performance comparison between model types"""
    if data_df is None:
        return
    
    completed_df = data_df[data_df['status'] == 'completed'].copy()
    
    if len(completed_df) == 0:
        st.warning("No completed models to display")
        return
    
    # Regression performance comparison
    reg_df = completed_df[completed_df['model_subtype'] == 'Regression']
    
    fig = px.box(
        reg_df,
        x='model_type',
        y='regression_r2',
        color='organism',
        title="Regression R² Performance by Model Type and Organism",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_layout(
        xaxis_title="Model Type",
        yaxis_title="R² Score"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_classification_performance(data_df):
    """Create classification performance comparison"""
    if data_df is None:
        return
    
    completed_df = data_df[data_df['status'] == 'completed'].copy()
    
    if len(completed_df) == 0:
        st.warning("No completed models to display")
        return
    
    # Classification performance comparison
    class_df = completed_df[completed_df['model_subtype'] == 'Classification']
    
    fig = px.box(
        class_df,
        x='model_type',
        y='classification_f1',
        color='organism',
        title="Classification F1 Score by Model Type and Organism",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_layout(
        xaxis_title="Model Type",
        yaxis_title="F1 Score"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_top_performers_table(data_df):
    """Create top performers table"""
    if data_df is None:
        return
    
    completed_df = data_df[data_df['status'] == 'completed'].copy()
    
    if len(completed_df) == 0:
        st.warning("No completed models to display")
        return
    
    # Top regression performers
    reg_df = completed_df[completed_df['model_subtype'] == 'Regression']
    top_reg = reg_df.nlargest(10, 'regression_r2')[
        ['protein_name', 'organism', 'model_type', 'n_samples', 'regression_r2', 'regression_rmse']
    ]
    
    st.subheader("Top 10 Regression Models (by R²)")
    st.dataframe(top_reg, use_container_width=True)
    
    # Top classification performers
    class_df = completed_df[completed_df['model_subtype'] == 'Classification']
    top_class = class_df.nlargest(10, 'classification_f1')[
        ['protein_name', 'organism', 'model_type', 'n_samples', 'classification_accuracy', 'classification_f1', 'classification_auc']
    ]
    
    st.subheader("Top 10 Classification Models (by F1 Score)")
    st.dataframe(top_class, use_container_width=True)

# MCC Analysis visualization functions
def create_mcc_summary_metrics(summary_df):
    """Create MCC analysis summary metrics"""
    if summary_df is None or len(summary_df) == 0:
        st.warning("No MCC analysis data available")
        return
    
    st.subheader("MCC Analysis Summary")
    
    col1, col2 = st.columns(2)
    
    for _, row in summary_df.iterrows():
        model_type = row['model_type'].replace('_', '+').upper()
        
        with col1 if row['model_type'] == 'morgan' else col2:
            st.markdown(f"**{model_type} Models**")
            
            # Create metrics
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric(
                    "Total Comparisons", 
                    int(row['total_comparisons']),
                    help="Number of protein-organism combinations compared"
                )
            
            with col_b:
                st.metric(
                    "Classification Wins", 
                    f"{int(row['classification_wins'])} ({row['classification_win_rate']:.1%})",
                    help="Number and percentage of cases where classification performed better"
                )
            
            with col_c:
                st.metric(
                    "Mean MCC Difference", 
                    f"{row['mean_mcc_difference']:.4f}",
                    f"±{row['std_mcc_difference']:.4f}",
                    help="Average MCC difference (Classification - Regression)"
                )

def create_mcc_win_rate_chart(summary_df):
    """Create win rate comparison chart"""
    if summary_df is None or len(summary_df) == 0:
        return
    
    # Prepare data for plotting
    plot_data = []
    for _, row in summary_df.iterrows():
        model_type = row['model_type'].replace('_', '+').upper()
        plot_data.extend([
            {'Model Type': model_type, 'Approach': 'Classification', 'Wins': int(row['classification_wins'])},
            {'Model Type': model_type, 'Approach': 'Regression', 'Wins': int(row['regression_wins'])}
        ])
    
    df_plot = pd.DataFrame(plot_data)
    
    fig = px.bar(
        df_plot,
        x='Model Type',
        y='Wins',
        color='Approach',
        title="Model Performance Comparison: Classification vs Regression",
        color_discrete_map={'Classification': '#2E8B57', 'Regression': '#DC143C'},
        barmode='group'
    )
    
    fig.update_layout(
        xaxis_title="Model Type",
        yaxis_title="Number of Wins",
        legend_title="Model Approach"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_mcc_difference_analysis(morgan_df, esm_df):
    """Create MCC difference analysis"""
    if morgan_df is None and esm_df is None:
        st.warning("No detailed MCC analysis data available")
        return
    
    st.subheader("MCC Difference Analysis")
    
    # Create comparison plots
    col1, col2 = st.columns(2)
    
    with col1:
        if morgan_df is not None and len(morgan_df) > 0:
            st.markdown("**Morgan Models**")
            
            # MCC difference distribution
            fig = px.histogram(
                morgan_df,
                x='mcc_difference',
                title="MCC Difference Distribution (Morgan)",
                nbins=20,
                color_discrete_sequence=['#1f77b4']
            )
            fig.add_vline(x=0, line_dash="dash", line_color="red", 
                         annotation_text="No Difference", annotation_position="top")
            fig.update_layout(xaxis_title="MCC Difference (Classification - Regression)", 
                            yaxis_title="Frequency")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if esm_df is not None and len(esm_df) > 0:
            st.markdown("**ESM+Morgan Models**")
            
            # MCC difference distribution
            fig = px.histogram(
                esm_df,
                x='mcc_difference',
                title="MCC Difference Distribution (ESM+Morgan)",
                nbins=20,
                color_discrete_sequence=['#ff7f0e']
            )
            fig.add_vline(x=0, line_dash="dash", line_color="red", 
                         annotation_text="No Difference", annotation_position="top")
            fig.update_layout(xaxis_title="MCC Difference (Classification - Regression)", 
                            yaxis_title="Frequency")
            st.plotly_chart(fig, use_container_width=True)

def create_mcc_scatter_plot(morgan_df, esm_df):
    """Create MCC scatter plot comparison"""
    if morgan_df is None and esm_df is None:
        return
    
    st.subheader("MCC Scatter Plot Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if morgan_df is not None and len(morgan_df) > 0:
            st.markdown("**Morgan Models**")
            
            # Extract MCC values from JSON strings
            morgan_plot_df = morgan_df.copy()
            
            # Parse regression metrics
            reg_mcc = []
            class_mcc = []
            for idx, row in morgan_plot_df.iterrows():
                try:
                    reg_metrics = eval(row['regression_metrics']) if isinstance(row['regression_metrics'], str) else row['regression_metrics']
                    class_metrics = eval(row['classification_metrics']) if isinstance(row['classification_metrics'], str) else row['classification_metrics']
                    reg_mcc.append(reg_metrics['mcc'])
                    class_mcc.append(class_metrics['mcc'])
                except:
                    reg_mcc.append(np.nan)
                    class_mcc.append(np.nan)
            
            morgan_plot_df['reg_mcc'] = reg_mcc
            morgan_plot_df['class_mcc'] = class_mcc
            
            fig = px.scatter(
                morgan_plot_df,
                x='reg_mcc',
                y='class_mcc',
                title="Regression vs Classification MCC (Morgan)",
                color='organism',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            
            # Add diagonal line
            min_val = min(morgan_plot_df['reg_mcc'].min(), morgan_plot_df['class_mcc'].min())
            max_val = max(morgan_plot_df['reg_mcc'].max(), morgan_plot_df['class_mcc'].max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(dash='dash', color='red'),
                name='Equal Performance',
                showlegend=True
            ))
            
            fig.update_layout(
                xaxis_title="Regression MCC",
                yaxis_title="Classification MCC"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if esm_df is not None and len(esm_df) > 0:
            st.markdown("**ESM+Morgan Models**")
            
            # Extract MCC values from JSON strings
            esm_plot_df = esm_df.copy()
            
            # Parse regression metrics
            reg_mcc = []
            class_mcc = []
            for idx, row in esm_plot_df.iterrows():
                try:
                    reg_metrics = eval(row['regression_metrics']) if isinstance(row['regression_metrics'], str) else row['regression_metrics']
                    class_metrics = eval(row['classification_metrics']) if isinstance(row['classification_metrics'], str) else row['classification_metrics']
                    reg_mcc.append(reg_metrics['mcc'])
                    class_mcc.append(class_metrics['mcc'])
                except:
                    reg_mcc.append(np.nan)
                    class_mcc.append(np.nan)
            
            esm_plot_df['reg_mcc'] = reg_mcc
            esm_plot_df['class_mcc'] = class_mcc
            
            fig = px.scatter(
                esm_plot_df,
                x='reg_mcc',
                y='class_mcc',
                title="Regression vs Classification MCC (ESM+Morgan)",
                color='organism',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            
            # Add diagonal line
            min_val = min(esm_plot_df['reg_mcc'].min(), esm_plot_df['class_mcc'].min())
            max_val = max(esm_plot_df['reg_mcc'].max(), esm_plot_df['class_mcc'].max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(dash='dash', color='red'),
                name='Equal Performance',
                showlegend=True
            ))
            
            fig.update_layout(
                xaxis_title="Regression MCC",
                yaxis_title="Classification MCC"
            )
            st.plotly_chart(fig, use_container_width=True)

def create_mcc_detailed_table(morgan_df, esm_df):
    """Create detailed MCC comparison table"""
    if morgan_df is None and esm_df is None:
        st.warning("No detailed MCC analysis data available")
        return
    
    st.subheader("Detailed MCC Comparison Results")
    
    # Create tabs for different model types
    tab1, tab2 = st.tabs(["Morgan Models", "ESM+Morgan Models"])
    
    with tab1:
        if morgan_df is not None and len(morgan_df) > 0:
            # Select relevant columns for display
            display_cols = [
                'protein_name', 'organism', 'n_samples',
                'mcc_difference', 'accuracy_difference', 'f1_difference',
                'better_model', 'mcc_improvement'
            ]
            
            # Rename columns for better display
            display_df = morgan_df[display_cols].copy()
            display_df.columns = [
                'Protein', 'Organism', 'Samples',
                'MCC Difference', 'Accuracy Difference', 'F1 Difference',
                'Better Model', 'MCC Improvement'
            ]
            
            # Round numeric columns
            numeric_cols = ['MCC Difference', 'Accuracy Difference', 'F1 Difference', 'MCC Improvement']
            display_df[numeric_cols] = display_df[numeric_cols].round(4)
            
            st.dataframe(display_df, use_container_width=True)
        else:
            st.warning("No Morgan model data available")
    
    with tab2:
        if esm_df is not None and len(esm_df) > 0:
            # Select relevant columns for display
            display_cols = [
                'protein_name', 'organism', 'n_samples',
                'mcc_difference', 'accuracy_difference', 'f1_difference',
                'better_model', 'mcc_improvement'
            ]
            
            # Rename columns for better display
            display_df = esm_df[display_cols].copy()
            display_df.columns = [
                'Protein', 'Organism', 'Samples',
                'MCC Difference', 'Accuracy Difference', 'F1 Difference',
                'Better Model', 'MCC Improvement'
            ]
            
            # Round numeric columns
            numeric_cols = ['MCC Difference', 'Accuracy Difference', 'F1 Difference', 'MCC Improvement']
            display_df[numeric_cols] = display_df[numeric_cols].round(4)
            
            st.dataframe(display_df, use_container_width=True)
        else:
            st.warning("No ESM+Morgan model data available")

# AQSE Analysis functions
def create_aqse_summary_metrics(aqse_df):
    """Create AQSE model summary metrics"""
    if aqse_df is None:
        st.warning("No AQSE model data available")
        return
    
    st.subheader("AQSE Model Summary")
    
    # Count different model types
    total_models = len(aqse_df)
    threshold_models = len(aqse_df[aqse_df['model_type'].str.contains('Threshold', na=False)])
    standard_models = len(aqse_df[aqse_df['model_type'] == 'Standard'])
    unique_proteins = aqse_df['target_name'].nunique()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total AQSE Models", total_models, help="Total AQSE models created")
    
    with col2:
        st.metric("Unique Proteins", unique_proteins, help="Proteins with AQSE models")
    
    with col3:
        st.metric("Threshold Models", threshold_models, help="Models using similarity thresholds (high, medium, low)")
    
    with col4:
        st.metric("Standard Models", standard_models, help="Standard AQSE models")

def create_aqse_performance_comparison(aqse_df):
    """Create AQSE performance comparison by model type"""
    if aqse_df is None:
        return
    
    st.subheader("AQSE Performance by Model Type")
    
    # Filter out NaN values for plotting
    plot_df = aqse_df.dropna(subset=['r2', 'q2'])
    
    if len(plot_df) == 0:
        st.warning("No valid performance data for plotting")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # R² comparison
        fig_r2 = px.box(
            plot_df, 
            x='model_type', 
            y='r2',
            title='R² Score by AQSE Model Type',
            color='model_type',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_r2.update_layout(showlegend=False)
        st.plotly_chart(fig_r2, use_container_width=True)
    
    with col2:
        # Q² comparison
        fig_q2 = px.box(
            plot_df, 
            x='model_type', 
            y='q2',
            title='Q² Score by AQSE Model Type',
            color='model_type',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_q2.update_layout(showlegend=False)
        st.plotly_chart(fig_q2, use_container_width=True)
    
    # Add performance summary
    st.subheader("Performance Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_r2 = plot_df['r2'].mean()
        st.metric("Average R²", f"{avg_r2:.3f}")
    
    with col2:
        avg_q2 = plot_df['q2'].mean()
        st.metric("Average Q²", f"{avg_q2:.3f}")
    
    with col3:
        best_r2 = plot_df['r2'].max()
        st.metric("Best R²", f"{best_r2:.3f}")

def create_aqse_threshold_analysis(aqse_df):
    """Create threshold-specific analysis"""
    if aqse_df is None:
        return
    
    st.subheader("Threshold-Specific Analysis")
    
    # Filter threshold models
    threshold_df = aqse_df[aqse_df['model_type'].str.contains('Threshold', na=False)]
    
    if len(threshold_df) == 0:
        st.warning("No threshold models found")
        return
    
    # Extract threshold from model_type
    threshold_df = threshold_df.copy()
    threshold_df['threshold'] = threshold_df['model_type'].str.extract(r'\((\w+)\)')
    
    col1, col2 = st.columns(2)
    
    with col1:
        # R² by threshold
        fig_r2 = px.box(
            threshold_df, 
            x='threshold', 
            y='r2',
            title='R² Score by Similarity Threshold',
            color='threshold',
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        st.plotly_chart(fig_r2, use_container_width=True)
    
    with col2:
        # Sample size by threshold
        fig_samples = px.box(
            threshold_df, 
            x='threshold', 
            y='n_train',
            title='Training Samples by Similarity Threshold',
            color='threshold',
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        st.plotly_chart(fig_samples, use_container_width=True)
    
    # Add threshold summary
    st.subheader("Threshold Performance Summary")
    
    threshold_summary = threshold_df.groupby('threshold').agg({
        'r2': ['mean', 'std', 'count'],
        'q2': ['mean', 'std'],
        'n_train': ['mean', 'std'],
        'n_similar_proteins': ['mean', 'std']
    }).round(3)
    
    st.dataframe(threshold_summary, use_container_width=True)

def create_aqse_dumbbell_plot(aqse_df, standard_df):
    """Create dumbbell plot comparing standard QSAR vs AQSE models at different thresholds"""
    if aqse_df is None or standard_df is None:
        st.warning("Missing data for dumbbell plot comparison")
        return
    
    st.subheader("AQSE vs Standard QSAR Performance Comparison")
    st.markdown("**Dumbbell Plot: Standard ESM+Morgan vs AQSE Threshold Models (Absolute R² Scores)**")
    
    # Filter for threshold models only
    threshold_df = aqse_df[aqse_df['model_type'].str.contains('Threshold', na=False)].copy()
    
    if len(threshold_df) == 0:
        st.warning("No threshold AQSE models found for comparison")
        return
    
    # Extract threshold from model_type
    threshold_df['threshold'] = threshold_df['model_type'].str.extract(r'\((\w+)\)')
    
    # Get standard QSAR results for comparison (ESM+Morgan regression only)
    standard_reg = standard_df[standard_df['status'] == 'completed'].copy()
    
    if len(standard_reg) == 0:
        st.warning("No standard QSAR regression models found for comparison")
        return
    
    # Create comparison data - one row per protein-model combination
    comparison_data = []
    
    # Get unique proteins from AQSE data
    aqse_proteins = threshold_df['target_name'].unique()
    
    for protein in aqse_proteins:
        # Get standard QSAR performance for this protein
        standard_protein = standard_reg[standard_reg['protein_name'] == protein]
        
        if len(standard_protein) == 0:
            continue  # Skip if no standard model for this protein
        
        # Use the first (and should be only) standard model for this protein
        standard_r2 = standard_protein['regression_r2'].iloc[0]
        standard_samples = standard_protein['n_samples'].iloc[0]
        
        # Add standard model data point
        comparison_data.append({
            'protein': protein,
            'uniprot_id': standard_protein['uniprot_id'].iloc[0] if 'uniprot_id' in standard_protein.columns else protein,
            'model_type': 'Standard',
            'threshold': 'Standard',
            'r2': standard_r2,
            'n_samples': standard_samples,
            'n_similar_proteins': 0,
            'is_standard': True
        })
        
        # Get AQSE models for this protein
        aqse_protein = threshold_df[threshold_df['target_name'] == protein]
        
        for _, row in aqse_protein.iterrows():
            comparison_data.append({
                'protein': protein,
                'uniprot_id': row['uniprot_id'],
                'model_type': 'AQSE',
                'threshold': row['threshold'],
                'r2': row['r2'],
                'n_samples': row['n_train'],
                'n_similar_proteins': row['n_similar_proteins'],
                'is_standard': False
            })
    
    if len(comparison_data) == 0:
        st.warning("No common proteins found between AQSE and standard QSAR models")
        return
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create dumbbell plot
    fig = go.Figure()
    
    # Define colors for different model types
    model_colors = {
        'Standard': '#2C3E50',      # Dark blue-gray for benchmark
        'high': '#27AE60',          # Green for high threshold
        'medium': '#F39C12',        # Orange for medium threshold  
        'low': '#E74C3C'            # Red for low threshold
    }
    
    # Get unique proteins for y-axis ordering
    unique_proteins = sorted(comparison_df['protein'].unique())
    
    # Create y-axis positions for each protein
    protein_positions = {protein: i for i, protein in enumerate(unique_proteins)}
    
    # Calculate global sample size range for consistent scaling
    all_samples = comparison_df['n_samples'].values
    min_global_samples = all_samples.min()
    max_global_samples = all_samples.max()
    
    # Define marker size range (5-25 pixels)
    min_marker_size = 5
    max_marker_size = 25
    
    # Add data points for each model type
    for model_type in ['Standard', 'high', 'medium', 'low']:
        if model_type == 'Standard':
            model_data = comparison_df[comparison_df['is_standard'] == True]
            model_name = 'Standard QSAR'
        else:
            model_data = comparison_df[(comparison_df['threshold'] == model_type) & (comparison_df['is_standard'] == False)]
            model_name = f'AQSE {model_type.title()}'
        
        if len(model_data) == 0:
            continue
            
        # Create y positions for this model type
        y_positions = []
        for _, row in model_data.iterrows():
            base_y = protein_positions[row['protein']]
            # Add small vertical offset for multiple models per protein
            if model_type == 'Standard':
                y_offset = 0
            elif model_type == 'high':
                y_offset = 0.15
            elif model_type == 'medium':
                y_offset = 0.05
            else:  # low
                y_offset = -0.1
            
            y_positions.append(base_y + y_offset)
        
        # Calculate marker sizes based on sample size using global scaling
        if max_global_samples > min_global_samples:
            # Normalize to min_marker_size - max_marker_size range using global min/max
            marker_sizes = min_marker_size + (model_data['n_samples'] - min_global_samples) / (max_global_samples - min_global_samples) * (max_marker_size - min_marker_size)
        else:
            # If all samples are the same size, use default size
            marker_sizes = [15] * len(model_data)
        
        # Add scatter points
        fig.add_trace(go.Scatter(
            x=model_data['r2'],
            y=y_positions,
            mode='markers',
            marker=dict(
                size=marker_sizes,
                color=model_colors[model_type],
                line=dict(width=1, color='white'),
                sizemode='diameter'
            ),
            name=model_name,
            customdata=list(zip(model_data['protein'], 
                              [model_name] * len(model_data),
                              model_data['n_samples'],
                              model_data['n_similar_proteins'])),
            hovertemplate="<b>%{customdata[0]}</b><br>" +
                         "Model: %{customdata[1]}<br>" +
                         "R² Score: %{x:.3f}<br>" +
                         "Total Samples: %{customdata[2]}<br>" +
                         "Similar Proteins: %{customdata[3]}<br>" +
                         "<extra></extra>",
            showlegend=True
        ))
    
    # Add connecting lines between standard and AQSE models for each protein
    for protein in unique_proteins:
        protein_data = comparison_df[comparison_df['protein'] == protein]
        standard_data = protein_data[protein_data['is_standard'] == True]
        aqse_data = protein_data[protein_data['is_standard'] == False]
        
        if len(standard_data) > 0 and len(aqse_data) > 0:
            standard_r2 = standard_data['r2'].iloc[0]
            base_y = protein_positions[protein]
            
            # Connect standard to each AQSE model
            for _, aqse_row in aqse_data.iterrows():
                aqse_r2 = aqse_row['r2']
                threshold = aqse_row['threshold']
                
                # Add connecting line
                fig.add_trace(go.Scatter(
                    x=[standard_r2, aqse_r2],
                    y=[base_y, base_y],
                    mode='lines',
                    line=dict(
                        color=model_colors[threshold],
                        width=2,
                        dash='dot'
                    ),
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    # Update layout
    fig.update_layout(
        title="AQSE vs Standard QSAR Performance Comparison<br><sub>Absolute R² Scores - Each Row is a Protein | Marker Size ∝ Training Samples</sub>",
        xaxis_title="R² Score",
        yaxis_title="Protein",
        height=max(400, len(unique_proteins) * 40),  # Dynamic height based on number of proteins
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(unique_proteins))),
            ticktext=unique_proteins,
            showgrid=True
        )
    )
    
    # Add size legend annotation
    fig.add_annotation(
        x=0.02, y=0.98,
        xref="paper", yref="paper",
        text=f"<b>Marker Size Legend:</b><br>Small: {min_global_samples:.0f} samples<br>Large: {max_global_samples:.0f} samples",
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=10)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add summary statistics
    st.subheader("Performance Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Calculate average R² for each model type
        standard_avg = comparison_df[comparison_df['is_standard'] == True]['r2'].mean() if len(comparison_df[comparison_df['is_standard'] == True]) > 0 else 0
        aqse_avg = comparison_df[comparison_df['is_standard'] == False]['r2'].mean() if len(comparison_df[comparison_df['is_standard'] == False]) > 0 else 0
        avg_improvement = aqse_avg - standard_avg
        st.metric("Average R² Improvement", f"{avg_improvement:+.3f}", 
                 help="Average difference between AQSE and Standard models")
    
    with col2:
        # Count models that improved
        protein_improvements = []
        for protein in unique_proteins:
            protein_data = comparison_df[comparison_df['protein'] == protein]
            standard_data = protein_data[protein_data['is_standard'] == True]
            aqse_data = protein_data[protein_data['is_standard'] == False]
            
            if len(standard_data) > 0 and len(aqse_data) > 0:
                standard_r2 = standard_data['r2'].iloc[0]
                best_aqse_r2 = aqse_data['r2'].max()
                protein_improvements.append(best_aqse_r2 > standard_r2)
        
        models_improved = sum(protein_improvements)
        total_proteins = len(protein_improvements)
        improvement_rate = models_improved / total_proteins * 100 if total_proteins > 0 else 0
        st.metric("Proteins Improved", f"{models_improved}/{total_proteins} ({improvement_rate:.1f}%)",
                 help="Number and percentage of proteins where best AQSE model outperformed standard")
    
    with col3:
        # Average sample expansion
        sample_expansions = []
        for protein in unique_proteins:
            protein_data = comparison_df[comparison_df['protein'] == protein]
            standard_data = protein_data[protein_data['is_standard'] == True]
            aqse_data = protein_data[protein_data['is_standard'] == False]
            
            if len(standard_data) > 0 and len(aqse_data) > 0:
                standard_samples = standard_data['n_samples'].iloc[0]
                avg_aqse_samples = aqse_data['n_samples'].mean()
                sample_expansions.append(avg_aqse_samples - standard_samples)
        
        avg_expansion = np.mean(sample_expansions) if sample_expansions else 0
        st.metric("Average Sample Expansion", f"+{avg_expansion:.0f}",
                 help="Average increase in training samples from AQSE")
    
    # Detailed comparison table
    st.subheader("Detailed Comparison Table")
    
    # Prepare display data
    display_df = comparison_df.copy()
    display_df = display_df.sort_values(['protein', 'is_standard', 'threshold'])
    
    # Select and rename columns for display
    display_columns = {
        'protein': 'Protein',
        'model_type': 'Model Type',
        'threshold': 'Threshold', 
        'r2': 'R² Score',
        'n_samples': 'Total Samples',
        'n_similar_proteins': 'Similar Proteins'
    }
    
    display_df = display_df[list(display_columns.keys())].rename(columns=display_columns)
    
    # Round numeric columns
    display_df['R² Score'] = display_df['R² Score'].round(3)
    
    st.dataframe(display_df, use_container_width=True)

def create_sample_expansion_dumbbell_plot(aqse_df, standard_df):
    """Create dumbbell plot showing sample expansion for all proteins"""
    if aqse_df is None or standard_df is None:
        st.warning("Missing data for sample expansion dumbbell plot")
        return
    
    st.subheader("Sample Expansion Analysis - All Proteins")
    st.markdown("**Dumbbell Plot: Standard QSAR vs AQSE Models with Sample Expansion (Absolute R² Scores)**")
    
    # Get all AQSE models (including standard and threshold models)
    all_aqse_df = aqse_df.copy()
    
    # Extract threshold from model_type for AQSE models
    all_aqse_df['threshold'] = all_aqse_df['model_type'].str.extract(r'\((\w+)\)')
    all_aqse_df['threshold'] = all_aqse_df['threshold'].fillna('Standard')
    
    # Get standard QSAR results for comparison (ESM+Morgan regression only)
    standard_reg = standard_df[standard_df['status'] == 'completed'].copy()
    
    if len(standard_reg) == 0:
        st.warning("No standard QSAR regression models found for comparison")
        return
    
    # Create comparison data - one row per protein-model combination
    comparison_data = []
    
    # Get all unique proteins from both datasets
    aqse_proteins = set(all_aqse_df['target_name'].unique())
    standard_proteins = set(standard_reg['protein_name'].unique())
    all_proteins = sorted(aqse_proteins.union(standard_proteins))
    
    for protein in all_proteins:
        # Get standard QSAR performance for this protein
        standard_protein = standard_reg[standard_reg['protein_name'] == protein]
        
        if len(standard_protein) > 0:
            # Use the first (and should be only) standard model for this protein
            standard_r2 = standard_protein['regression_r2'].iloc[0]
            standard_samples = standard_protein['n_samples'].iloc[0]
            
            # Add standard model data point
            comparison_data.append({
                'protein': protein,
                'uniprot_id': standard_protein['uniprot_id'].iloc[0] if 'uniprot_id' in standard_protein.columns else protein,
                'model_type': 'Standard',
                'threshold': 'Standard',
                'r2': standard_r2,
                'n_samples': standard_samples,
                'n_similar_proteins': 0,
                'is_standard': True,
                'sample_expansion': 0
            })
        
        # Get AQSE models for this protein
        aqse_protein = all_aqse_df[all_aqse_df['target_name'] == protein]
        
        for _, row in aqse_protein.iterrows():
            # Calculate sample expansion
            if len(standard_protein) > 0:
                sample_expansion = row['n_train'] - standard_samples
            else:
                sample_expansion = row['n_train']  # No standard model to compare to
            
            comparison_data.append({
                'protein': protein,
                'uniprot_id': row['uniprot_id'],
                'model_type': 'AQSE',
                'threshold': row['threshold'],
                'r2': row['r2'],
                'n_samples': row['n_train'],
                'n_similar_proteins': row['n_similar_proteins'],
                'is_standard': False,
                'sample_expansion': sample_expansion
            })
    
    if len(comparison_data) == 0:
        st.warning("No data found for sample expansion analysis")
        return
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create dumbbell plot
    fig = go.Figure()
    
    # Define colors for different model types
    model_colors = {
        'Standard': '#2C3E50',      # Dark blue-gray for benchmark
        'Standard': '#2C3E50',      # AQSE Standard
        'high': '#27AE60',          # Green for high threshold
        'medium': '#F39C12',        # Orange for medium threshold  
        'low': '#E74C3C'            # Red for low threshold
    }
    
    # Get unique proteins for y-axis ordering
    unique_proteins = sorted(comparison_df['protein'].unique())
    
    # Create y-axis positions for each protein
    protein_positions = {protein: i for i, protein in enumerate(unique_proteins)}
    
    # Calculate global sample size range for consistent scaling
    all_samples = comparison_df['n_samples'].values
    min_global_samples = all_samples.min()
    max_global_samples = all_samples.max()
    
    # Define marker size range (5-25 pixels)
    min_marker_size = 5
    max_marker_size = 25
    
    # Add data points for each model type
    for model_type in ['Standard', 'high', 'medium', 'low']:
        if model_type == 'Standard':
            model_data = comparison_df[comparison_df['is_standard'] == True]
            model_name = 'Standard QSAR'
        else:
            model_data = comparison_df[(comparison_df['threshold'] == model_type) & (comparison_df['is_standard'] == False)]
            model_name = f'AQSE {model_type.title()}'
        
        if len(model_data) == 0:
            continue
            
        # Create y positions for this model type
        y_positions = []
        for _, row in model_data.iterrows():
            base_y = protein_positions[row['protein']]
            # Add small vertical offset for multiple models per protein
            if model_type == 'Standard':
                y_offset = 0
            elif model_type == 'high':
                y_offset = 0.15
            elif model_type == 'medium':
                y_offset = 0.05
            else:  # low
                y_offset = -0.1
            
            y_positions.append(base_y + y_offset)
        
        # Calculate marker sizes based on sample size using global scaling
        if max_global_samples > min_global_samples:
            # Normalize to min_marker_size - max_marker_size range using global min/max
            marker_sizes = min_marker_size + (model_data['n_samples'] - min_global_samples) / (max_global_samples - min_global_samples) * (max_marker_size - min_marker_size)
        else:
            # If all samples are the same size, use default size
            marker_sizes = [15] * len(model_data)
        
        # Add scatter points
        fig.add_trace(go.Scatter(
            x=model_data['r2'],
            y=y_positions,
            mode='markers',
            marker=dict(
                size=marker_sizes,
                color=model_colors[model_type],
                line=dict(width=1, color='white'),
                sizemode='diameter'
            ),
            name=model_name,
            customdata=list(zip(model_data['protein'], 
                              [model_name] * len(model_data),
                              model_data['n_samples'],
                              model_data['n_similar_proteins'],
                              model_data['sample_expansion'])),
            hovertemplate="<b>%{customdata[0]}</b><br>" +
                         "Model: %{customdata[1]}<br>" +
                         "R² Score: %{x:.3f}<br>" +
                         "Total Samples: %{customdata[2]}<br>" +
                         "Similar Proteins: %{customdata[3]}<br>" +
                         "Sample Expansion: +%{customdata[4]}<br>" +
                         "<extra></extra>",
            showlegend=True
        ))
    
    # Add connecting lines between standard and AQSE models for each protein
    for protein in unique_proteins:
        protein_data = comparison_df[comparison_df['protein'] == protein]
        standard_data = protein_data[protein_data['is_standard'] == True]
        aqse_data = protein_data[protein_data['is_standard'] == False]
        
        if len(standard_data) > 0 and len(aqse_data) > 0:
            standard_r2 = standard_data['r2'].iloc[0]
            base_y = protein_positions[protein]
            
            # Connect standard to each AQSE model
            for _, aqse_row in aqse_data.iterrows():
                aqse_r2 = aqse_row['r2']
                threshold = aqse_row['threshold']
                
                # Add connecting line
                fig.add_trace(go.Scatter(
                    x=[standard_r2, aqse_r2],
                    y=[base_y, base_y],
                    mode='lines',
                    line=dict(
                        color=model_colors[threshold],
                        width=2,
                        dash='dot'
                    ),
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    # Update layout
    fig.update_layout(
        title="Sample Expansion Analysis - All Proteins<br><sub>Absolute R² Scores - Each Row is a Protein | Marker Size ∝ Training Samples</sub>",
        xaxis_title="R² Score",
        yaxis_title="Protein",
        height=max(400, len(unique_proteins) * 40),  # Dynamic height based on number of proteins
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(unique_proteins))),
            ticktext=unique_proteins,
            showgrid=True
        )
    )
    
    # Add size legend annotation
    fig.add_annotation(
        x=0.02, y=0.98,
        xref="paper", yref="paper",
        text=f"<b>Marker Size Legend:</b><br>Small: {min_global_samples:.0f} samples<br>Large: {max_global_samples:.0f} samples",
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=10)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add summary statistics
    st.subheader("Sample Expansion Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Calculate average R² for each model type
        standard_avg = comparison_df[comparison_df['is_standard'] == True]['r2'].mean() if len(comparison_df[comparison_df['is_standard'] == True]) > 0 else 0
        aqse_avg = comparison_df[comparison_df['is_standard'] == False]['r2'].mean() if len(comparison_df[comparison_df['is_standard'] == False]) > 0 else 0
        avg_improvement = aqse_avg - standard_avg
        st.metric("Average R² Improvement", f"{avg_improvement:+.3f}", 
                 help="Average difference between AQSE and Standard models")
    
    with col2:
        # Count models that improved
        protein_improvements = []
        for protein in unique_proteins:
            protein_data = comparison_df[comparison_df['protein'] == protein]
            standard_data = protein_data[protein_data['is_standard'] == True]
            aqse_data = protein_data[protein_data['is_standard'] == False]
            
            if len(standard_data) > 0 and len(aqse_data) > 0:
                standard_r2 = standard_data['r2'].iloc[0]
                best_aqse_r2 = aqse_data['r2'].max()
                protein_improvements.append(best_aqse_r2 > standard_r2)
        
        models_improved = sum(protein_improvements)
        total_proteins = len(protein_improvements)
        improvement_rate = models_improved / total_proteins * 100 if total_proteins > 0 else 0
        st.metric("Proteins Improved", f"{models_improved}/{total_proteins} ({improvement_rate:.1f}%)",
                 help="Number and percentage of proteins where best AQSE model outperformed standard")
    
    with col3:
        # Average sample expansion
        sample_expansions = comparison_df[comparison_df['sample_expansion'] > 0]['sample_expansion']
        avg_expansion = sample_expansions.mean() if len(sample_expansions) > 0 else 0
        st.metric("Average Sample Expansion", f"+{avg_expansion:.0f}",
                 help="Average increase in training samples from AQSE")
    
    with col4:
        # Total sample expansion
        total_expansion = comparison_df['sample_expansion'].sum()
        st.metric("Total Sample Expansion", f"+{total_expansion:.0f}",
                 help="Total increase in training samples across all AQSE models")
    
    # Detailed comparison table
    st.subheader("Detailed Sample Expansion Table")
    
    # Prepare display data
    display_df = comparison_df.copy()
    display_df = display_df.sort_values(['protein', 'is_standard', 'threshold'])
    
    # Select and rename columns for display
    display_columns = {
        'protein': 'Protein',
        'model_type': 'Model Type',
        'threshold': 'Threshold', 
        'r2': 'R² Score',
        'n_samples': 'Total Samples',
        'n_similar_proteins': 'Similar Proteins',
        'sample_expansion': 'Sample Expansion'
    }
    
    display_df = display_df[list(display_columns.keys())].rename(columns=display_columns)
    
    # Round numeric columns
    display_df['R² Score'] = display_df['R² Score'].round(3)
    
    st.dataframe(display_df, use_container_width=True)

def create_aqse_workflow_diagram():
    """Create AQSE workflow diagram and description"""
    st.header("AQSE Workflow Overview")
    
    st.markdown("""
    The **AQSE (Avoidome QSAR Similarity Expansion)** pipeline enriches the chemical space of avoidome proteins 
    through similarity-based expansion using the Papyrus database.
    """)
    
    # Workflow diagram using Mermaid
    workflow_diagram = """
    graph TD
        A["Step 1: Input Preparation<br/>Load 55 avoidome proteins<br/>Fetch UniProt sequences<br/>Create FASTA files"] --> B["Step 2: Similarity Search<br/>BLAST vs Papyrus DB"]
        
        B --> C["Similarity Thresholds"]
        C --> D["High: ≥70% identity<br/>Most conservative"]
        C --> E["Medium: ≥50% identity<br/>Balanced expansion"]
        C --> F["Low: ≥30% identity<br/>Maximum expansion"]
        
        B --> G["Step 3: Data Collection<br/>Collect bioactivity data<br/>Create expanded datasets"]
        
        D --> G
        E --> G
        F --> G
        
        G --> H["Step 4: QSAR Models<br/>Morgan fingerprints (2048)<br/>Physicochemical (14)<br/>ESM C embeddings (1280)<br/>Train Random Forest"]
        
        H --> I["Model Types"]
        I --> J["Standard QSAR <br/>No similar proteins"]
        I --> K["AQSE<br/>With similar proteins (25/55 proteins)"]
        
        B -.->|"Standard QSAR models"| J
        
        K --> L["Training Strategy<br/>Train: Similar + 80% target<br/>Test: 20% target holdout<br/>Features (3,342 total)<br/>Morgan: 2048 bits<br/>Physicochemical: 14<br/>ESM C: 1280 dims"]
        
        L --> M["Outputs<br/>Models, MetricsPredictions & plots"]
        
        
        style A fill:#e1f5fe
        style B fill:#f3e5f5
        style G fill:#e8f5e8
        style H fill:#fff3e0
        style I fill:#fce4ec
        style J fill:#ffebee
        style K fill:#e8f5e8
        style L fill:#f3e5f5
        style M fill:#e1f5fe

    """
    
    st.markdown("### Workflow Diagram")
    st.components.v1.html(f"""
    <div style="text-align: center;">
        <script src="https://cdn.jsdelivr.net/npm/mermaid@10.6.1/dist/mermaid.min.js"></script>
        <div class="mermaid">
        {workflow_diagram}
        </div>
    </div>
    """, height=1200)
    
    # Detailed step descriptions
    st.markdown("### Detailed Workflow Steps")
    
    # Step 1
    with st.expander("Step 1: Input Preparation", expanded=True):
        st.markdown("""
        **Purpose**: Prepare avoidome protein sequences for similarity search
        
        **Inputs**:
        - `avoidome_prot_list.csv`: List of avoidome proteins with UniProt IDs
        
        **Processes**:
        - Load and clean avoidome protein list
        - Fetch protein sequences from UniProt database
        - Create FASTA files for BLAST search
        - Generate BLAST configuration file
        
        **Outputs**:
        - `avoidome_sequences.csv`: Protein sequences with metadata
        - `avoidome_proteins_combined.fasta`: Combined FASTA file for BLAST
        - Individual FASTA files for each protein
        - `blast_config.txt`: BLAST configuration file
        """)
    
    # Step 2
    with st.expander("Step 2: Protein Similarity Search"):
        st.markdown("""
        **Purpose**: Find proteins similar to avoidome proteins using BLAST
        
        **Inputs**:
        - FASTA files from Step 1
        - Papyrus protein database (BLAST format)
        
        **Processes**:
        - Run BLAST search for each avoidome protein against Papyrus
        - Parse BLAST results and create similarity matrices
        - Identify similar proteins at different thresholds:
          - **High similarity**: ≥70% identity
          - **Medium similarity**: ≥50% identity  
          - **Low similarity**: ≥30% identity
        
        **Outputs**:
        - BLAST result files for each protein
        - Similarity matrices for each threshold
        - `similarity_search_summary.csv`: Summary of similarity results
        - Visualization plots (distributions, heatmaps)
        """)
    
    # Step 3
    with st.expander("Step 3: QSAR Model Creation"):
        st.markdown("""
        **Purpose**: Create QSAR models for each avoidome target using different similarity threshold protein sets
        
        **Inputs**:
        - Similarity search results from Step 2
        - Avoidome protein sequences from Step 1
        - Papyrus database (for direct bioactivity data queries)
        
        **Processes**:
        - Calculate Morgan fingerprints for all compounds in Papyrus database
        - For proteins with similar proteins: create threshold-specific models using similar proteins + target protein data
        - For proteins without similar proteins: create standard models using only target protein data
        - Generate molecular descriptors (Morgan fingerprints + physicochemical)
        - Generate protein descriptors (ESM C embeddings)
        - Train Random Forest models for each target-threshold combination
        
        **Outputs**:
        - Trained Random Forest models (.pkl files)
        - Feature data (molecular + protein descriptors)
        - Model predictions and performance metrics
        - `aqse_model_results.csv`: Performance statistics
        - Visualization plots (performance comparisons, feature importance)
        """)
    
    # Usage Information
    st.markdown("### Usage")
    
    st.code("""
# Run complete pipeline
python run_aqse_pipeline.py --papyrus-version 05.7

# Run individual steps
python run_aqse_pipeline.py --step 1
python run_aqse_pipeline.py --step 2 --papyrus-version 05.7
python run_aqse_pipeline.py --step 4 --papyrus-version 05.7
    """, language="bash")

def create_failure_summary_page():
    """Create the failure summary page with overall statistics and plots"""
    st.header("Model Failure Analysis Summary")
    
    # Define failure data based on the provided statistics
    failure_data = {
        "Total Avoidome Proteins": 52,
        "Successfully Modeled": 27,
        "Failed to Model": 25,
        "Success Rate": 52.0,
        "Failure Rate": 48.0
    }
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Proteins",
            value=failure_data["Total Avoidome Proteins"],
            help="Total number of Avoidome proteins analyzed"
        )
    
    with col2:
        st.metric(
            label="Successfully Modeled",
            value=failure_data["Successfully Modeled"],
            help="Proteins with successful QSAR models"
        )
    
    with col3:
        st.metric(
            label="Failed to Model",
            value=failure_data["Failed to Model"],
            help="Proteins that could not generate QSAR models"
        )
    
    with col4:
        st.metric(
            label="Success Rate",
            value=f"{failure_data['Success Rate']:.1f}%",
            help="Overall success rate for QSAR model generation"
        )
    
    # Create pie chart for success/failure distribution
    st.subheader("Model Success vs Failure Distribution")
    
    fig_pie = go.Figure(data=[go.Pie(
        labels=['Successfully Modeled', 'Failed to Model'],
        values=[failure_data["Successfully Modeled"], failure_data["Failed to Model"]],
        marker_colors=['#28a745', '#dc3545'],
        textinfo='label+percent+value',
        textfont_size=12
    )])
    
    fig_pie.update_layout(
        title="QSAR Model Generation Success Rate",
        font=dict(size=12),
        height=400
    )
    
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Create comprehensive organism analysis chart
    st.subheader("Success Rates by Organism and Sample Counts")
    
    # Organism data
    organism_data = {
        'Organism': ['Human', 'Rat', 'Mouse'],
        'Total Attempts': [51, 44, 43],
        'Successful': [27, 14, 3],
        'Success Rate': [52.9, 31.8, 7.0],
        'Avg Samples (Success)': [1192.4, 410.3, 335.7],
        'Avg Samples (Failed)': [3.3, 2.0, 1.5]
    }
    
    # Create subplot with two y-axes
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Success Rate by Organism', 'Average Sample Counts'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Success rate bar chart
    fig.add_trace(
        go.Bar(
            x=organism_data['Organism'],
            y=organism_data['Success Rate'],
            name='Success Rate (%)',
            marker_color=['#28a745', '#ffc107', '#dc3545'],
            text=[f"{rate:.1f}%" for rate in organism_data['Success Rate']],
            textposition='auto',
        ),
        row=1, col=1
    )
    
    # Sample counts comparison
    fig.add_trace(
        go.Bar(
            x=organism_data['Organism'],
            y=organism_data['Avg Samples (Success)'],
            name='Avg Samples (Success)',
            marker_color='#28a745',
            text=[f"{count:.1f}" for count in organism_data['Avg Samples (Success)']],
            textposition='auto',
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(
            x=organism_data['Organism'],
            y=organism_data['Avg Samples (Failed)'],
            name='Avg Samples (Failed)',
            marker_color='#dc3545',
            text=[f"{count:.1f}" for count in organism_data['Avg Samples (Failed)']],
            textposition='auto',
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title="Organism-Specific Analysis: Success Rates and Sample Counts",
        font=dict(size=12),
        height=500,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Organism", row=1, col=1)
    fig.update_xaxes(title_text="Organism", row=1, col=2)
    fig.update_yaxes(title_text="Success Rate (%)", row=1, col=1)
    fig.update_yaxes(title_text="Average Sample Count", row=1, col=2)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add detailed statistics table
    st.subheader("Detailed Organism Statistics")
    
    organism_df = pd.DataFrame(organism_data)
    organism_df['Failed'] = organism_df['Total Attempts'] - organism_df['Successful']
    
    # Format the dataframe for better display
    display_df = organism_df[['Organism', 'Total Attempts', 'Successful', 'Failed', 'Success Rate', 'Avg Samples (Success)', 'Avg Samples (Failed)']].copy()
    display_df['Success Rate'] = display_df['Success Rate'].round(1).astype(str) + '%'
    display_df['Avg Samples (Success)'] = display_df['Avg Samples (Success)'].round(1)
    display_df['Avg Samples (Failed)'] = display_df['Avg Samples (Failed)'].round(1)
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Add failure reasons plot
    st.subheader("Reasons for Model Failure")
    
    # Failure reasons data
    failure_reasons = {
        'No Data in Papyrus': 15,
        'Insufficient Data in Papyrus': 6,
        'Missing UniProt ID': 1
    }
    
    # Create horizontal bar chart for failure reasons
    reasons = list(failure_reasons.keys())
    counts = list(failure_reasons.values())
    percentages = [count/25*100 for count in counts]  # 25 total failures
    
    fig_reasons = go.Figure(data=[go.Bar(
        y=reasons,
        x=counts,
        orientation='h',
        marker_color=['#dc3545', '#ffc107', '#6c757d'],
        text=[f"{count} ({pct:.1f}%)" for count, pct in zip(counts, percentages)],
        textposition='auto',
    )])
    
    fig_reasons.update_layout(
        title="Distribution of Model Failure Causes (Human Proteins)",
        xaxis_title="Number of Proteins",
        yaxis_title="Failure Cause",
        font=dict(size=12),
        height=300
    )
    
    st.plotly_chart(fig_reasons, use_container_width=True)

def create_failure_causes_page():
    """Create the failure causes page with detailed breakdown"""
    st.header("Detailed Failure Analysis")
    
    # Define failure causes data
    failure_causes = {
        "No Data in Papyrus": {
            "count": 15,
            "percentage": 60.0,  # 15/25 * 100
            "proteins": [
                "ADH1A", "AKR7A3", "CACNB1", "CAV1", "CHRNA10", 
                "CHRNA5", "CHRNA9", "CNRIP1", "DIDO1", "FMO1", 
                "GABPA", "NAT8", "ORM1", "OXA1L", "SMPDL3A"
            ],
            "description": "0 samples available in Papyrus database"
        },
        "Insufficient Data in Papyrus": {
            "count": 6,
            "percentage": 24.0,  # 6/25 * 100
            "proteins": [
                ("AHR", 25), ("AOX1", 13), ("CHRNA3", 3), 
                ("GSTA1", 6), ("SLCO1B1", 16), ("SULT1A1", 6)
            ],
            "description": "3-25 samples available (insufficient for reliable modeling)"
        },
        "Missing UniProt ID": {
            "count": 1,
            "percentage": 4.0,  # 1/25 * 100
            "proteins": ["SLCO2B3"],
            "description": "No UniProt identifier available for data retrieval"
        }
    }
    
    # Display failure causes summary
    st.subheader("Failure Causes Breakdown")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="No Data in Papyrus",
            value=failure_causes["No Data in Papyrus"]["count"],
            delta=f"{failure_causes['No Data in Papyrus']['percentage']:.1f}%",
            delta_color="inverse",
            help="Proteins with 0 samples in Papyrus database"
        )
    
    with col2:
        st.metric(
            label="Insufficient Data",
            value=failure_causes["Insufficient Data in Papyrus"]["count"],
            delta=f"{failure_causes['Insufficient Data in Papyrus']['percentage']:.1f}%",
            delta_color="inverse",
            help="Proteins with 3-25 samples (insufficient for modeling)"
        )
    
    with col3:
        st.metric(
            label="Missing UniProt ID",
            value=failure_causes["Missing UniProt ID"]["count"],
            delta=f"{failure_causes['Missing UniProt ID']['percentage']:.1f}%",
            delta_color="inverse",
            help="Proteins without UniProt identifiers"
        )
    
    # Create horizontal bar chart for failure causes
    st.subheader("Failure Causes Distribution")
    
    causes = list(failure_causes.keys())
    counts = [failure_causes[cause]["count"] for cause in causes]
    percentages = [failure_causes[cause]["percentage"] for cause in causes]
    
    fig_hbar = go.Figure(data=[go.Bar(
        y=causes,
        x=counts,
        orientation='h',
        marker_color=['#dc3545', '#ffc107', '#6c757d'],
        text=[f"{count} ({pct:.1f}%)" for count, pct in zip(counts, percentages)],
        textposition='auto',
    )])
    
    fig_hbar.update_layout(
        title="Distribution of Model Failure Causes",
        xaxis_title="Number of Proteins",
        yaxis_title="Failure Cause",
        font=dict(size=12),
        height=400
    )
    
    st.plotly_chart(fig_hbar, use_container_width=True)
    
    # Detailed breakdown by cause
    st.subheader("Detailed Protein Lists by Failure Cause")
    
    for cause, data in failure_causes.items():
        with st.expander(f"{cause} ({data['count']} proteins - {data['percentage']:.1f}%)"):
            st.write(f"**Description:** {data['description']}")
            st.write(f"**Number of proteins:** {data['count']}")
            
            if cause == "Insufficient Data in Papyrus":
                # Special handling for proteins with sample counts
                st.write("**Proteins with sample counts:**")
                for protein, count in data['proteins']:
                    st.write(f"- {protein}: {count} samples")
            else:
                st.write("**Proteins:**")
                for protein in data['proteins']:
                    st.write(f"- {protein}")
    

def create_presentation_discussion():
    """Create the presentation discussion page"""
    
    st.markdown("### Discussions and plans")
    
    st.markdown("""
    **1. Splits**
    
    Data splitting methodologies in the QSAR models and AQSE models.
    """)
    st.markdown("[How to approach machine learning-based prediction of drug/compound–target interactions](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-023-00689-w)")

    
    st.markdown("""
    **2. Model Improvements**
    
    - **Ensemble methods**: ESM c AND Alphafold embeddinngs?
        - **Multicollinearity**: Addressing multicollinearity in structural embedding features (ongoing)
    - **Other technical model iprovements**: suggestions?
    - Fix: alphafold embedding models organism-specific
    """)
    
    st.markdown("""
    **3. AQSE Organism-Specific (ongoing)**
    
    Development of organism-specific AQSE models
    """)
    
    st.markdown("""
    **4. ADR Integration**
    
    Integration of Adverse Drug Reaction (ADR) data to link drug-protein interaction and safety assessment.

    1. Multioutput models
    Train models to predict both bioactivity (current) and ADR labels likelihood
    Architecture: Shared features and models, separate predictions   
    
    2. Multitasking: multiple proteins - link PPI - ADR
    


    """)

    st.markdown("""
    **5. Approved set of drugs integration**
    Approved set of drugs don't overlap with the papyrus drugs, but could be mapped to them.     
    Train models with 80% approved drugs similarity removed 
        ON and OFF – retrain models - comparison 
    """)


def create_qsar_model_overview():
    """Create QSAR Model Overview page"""
    st.header("QSAR Models: Fundamentals and Applications")
    
    # QSAR Model Types
    st.subheader("QSAR Model Types in This Dashboard")
    
    st.markdown("""
        ### **Organism-Specific Models**
        - **Target**: Protein from specific organism (human, mouse, rat)
        - **Data**: Bioactivity data filtered by organism
        """)


    
    # Model Architectures
    st.subheader("Model Architectures")
    
    st.markdown("""
    ### **1. Morgan Fingerprint Models**
    - **Molecular Descriptors**: Morgan fingerprints (radius 3, 2048 bits), 13 physicochemical descriptors
    
    ### **2. ESM + Morgan Models**
    - **Molecular Descriptors**: Morgan fingerprints (2048 bits), 13 physicochemical descriptors
    - **Protein Descriptors**: ESM C embeddings (1280 dimensions)

    
    ### **3. AlphaFold + Morgan Models**
    - **Molecular Descriptors**: Morgan fingerprints (2048 bits), 13 physicochemical descriptors
    - **Protein Descriptors**: AlphaFold structure embeddings (variable dimensions)

    """)
    
   
def create_qsar_summary_statistics():
    """Create QSAR Summary Statistics page"""
    st.header("Summary Statistics")
    
    # Load data
    data_df = load_sample_data()
    if data_df is None:
        st.error("Unable to load QSAR modeling data.")
        return
    
    # Summary metrics
    create_summary_metrics(data_df)
    
    # Model distribution
    st.subheader("Model Distribution")
    col1, col2 = st.columns(2)
    with col1:
        create_organism_distribution_chart(data_df)
    with col2:
        create_model_type_distribution(data_df)
    
    # Data distribution
    st.subheader("Data Distribution Analysis")
    col1, col2 = st.columns(2)
    with col1:
        create_sample_size_analysis(data_df)
    with col2:
        # Sample size by organism
        completed_df = data_df[data_df['status'] == 'completed']
        if len(completed_df) > 0:
            org_samples = completed_df.groupby('organism')['n_samples'].agg(['mean', 'median', 'std']).round(2)
            st.subheader("Sample Size Statistics by Organism")
            st.dataframe(org_samples, use_container_width=True)
    
    # Organism analysis
    st.subheader("Organism Analysis")
    completed_df = data_df[data_df['status'] == 'completed']
    if len(completed_df) > 0:
        organisms = completed_df['organism'].unique()
        
        # Display aggregated results for all organisms
        st.markdown("### Overall Performance Across All Organisms")
        
        # Model type breakdown
        model_breakdown = completed_df['model_type'].value_counts()
        subtype_breakdown = completed_df['model_subtype'].value_counts()
        organism_breakdown = completed_df['organism'].value_counts()
        
        st.markdown("**Model Selection & Breakdown:**")
        
        # Show the actual numbers and explain the selection
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.markdown("**Model Types:**")
            st.markdown(f"• **Morgan**: {model_breakdown.get('Morgan', 0)} models")
            st.markdown(f"• **ESM+Morgan**: {model_breakdown.get('ESM+Morgan', 0)} models")
            st.markdown(f"• **AlphaFold+Morgan**: {model_breakdown.get('AlphaFold+Morgan', 0)} models")
        
        with col_info2:
            st.markdown("**Model Subtypes:**")
            st.markdown(f"• **Regression**: {subtype_breakdown.get('Regression', 0)} models")
            st.markdown(f"• **Classification**: {subtype_breakdown.get('Classification', 0)} models")
            st.markdown("**Organisms:**")
            for org in sorted(organism_breakdown.index):
                st.markdown(f"• **{org.capitalize()}**: {organism_breakdown[org]} models")
        
        with col_info3:
            # Count failed models
            failed_df = data_df[data_df['status'] != 'completed']
            failed_count = len(failed_df)
            st.markdown("**Model Status:**")
            st.markdown(f"• **Completed**: {len(completed_df)} models")
            st.markdown(f"• **Failed**: {failed_count} models")
            if failed_count > 0:
                st.markdown(f"• **Success Rate**: {len(completed_df)/(len(completed_df)+failed_count):.1%}")
        
        # Explain the selection criteria
        st.markdown("**Criteria:**")
        st.markdown("• **Morgan & ESM+Morgan**: All organisms (human, mouse, rat) with sufficient data")
        st.markdown("• **AlphaFold+Morgan**: Human only (to fix)")
        st.markdown("• **Minimum requirement**: 30+ bioactivity samples per protein-organism combination")
        st.markdown("• **Each model type** includes both regression and classification variants")
        
        col1, col2, col3= st.columns(3)
        with col1:
            st.metric("Total Models", len(completed_df))
        with col2:
            avg_r2 = completed_df['regression_r2'].mean()
            st.metric("Avg R²", f"{avg_r2:.3f}")

        with col3:
            avg_samples = completed_df['n_samples'].mean()
            st.metric("Avg Samples", f"{avg_samples:.0f}")
        
        # Detailed breakdown by organism
        st.markdown("### Performance Breakdown by Organism")
        
        # Create a summary table for all organisms
        organism_summary = []
        for organism in sorted(organisms):
            org_data = completed_df[completed_df['organism'] == organism]
            organism_summary.append({
                'Organism': organism.capitalize(),
                'Total Models': len(org_data),
                'Avg R²': f"{org_data['regression_r2'].mean():.3f}",
                'Avg Samples': f"{org_data['n_samples'].mean():.0f}",
                'Min R²': f"{org_data['regression_r2'].min():.3f}",
                'Max R²': f"{org_data['regression_r2'].max():.3f}"
            })
        
        # Display the summary table
        summary_df = pd.DataFrame(organism_summary)
        st.dataframe(summary_df, use_container_width=True)

def create_qsar_model_performance():
    """Create QSAR Model Performance page"""
    st.header("Model Performance Overview")
    
    # Load data
    data_df = load_sample_data()
    if data_df is None:
        st.error("Unable to load QSAR modeling data.")
        return
    
    # ESM+Morgan R2 performance heatmap
    st.subheader("ESM+Morgan R² Performance Heatmap")
    create_performance_heatmap(data_df)
    
    # Top performers table
    st.subheader("Top Performing Models")
    create_top_performers_table(data_df)

def create_qsar_cross_organism():
    """Create QSAR Cross-Organism Comparison page"""
    st.header("Cross-Organism Comparison")
    
    # Load data
    data_df = load_sample_data()
    if data_df is None:
        st.error("Unable to load QSAR modeling data.")
        return
    
    create_performance_comparison(data_df)
    create_classification_performance(data_df)

def create_qsar_sample_size_analysis():
    """Create QSAR Sample Size Analysis page"""
    st.header("Sample Size Analysis")
    
    # Load data
    data_df = load_sample_data()
    if data_df is None:
        st.error("Unable to load QSAR modeling data.")
        return
    
    create_sample_size_analysis(data_df)
    
    # Additional sample size analysis
    completed_df = data_df[data_df['status'] == 'completed']
    if len(completed_df) > 0:
        st.subheader("Sample Size vs Performance Correlation")
        
        # Create correlation plot
        import plotly.express as px
        
        fig = px.scatter(
            completed_df, 
            x='n_samples', 
            y='regression_r2',
            color='model_type',
            title='Sample Size vs R² Performance',
            labels={'n_samples': 'Number of Samples', 'regression_r2': 'R² Score'}
        )
        st.plotly_chart(fig, use_container_width=True)

def create_qsar_embedding_comparison():
    """Create QSAR Embedding Comparison page"""
    st.header("Embedding Comparison")
    
    # Load data
    data_df = load_sample_data()
    if data_df is None:
        st.error("Unable to load QSAR modeling data.")
        return
    
    # Filter for ESM and AlphaFold models only
    embedding_df = data_df[data_df['model_type'].isin(['ESM+Morgan', 'AlphaFold+Morgan'])]
    completed_embedding = embedding_df[embedding_df['status'] == 'completed']
    
    if len(completed_embedding) > 0:
        st.subheader("ESM vs AlphaFold Performance Comparison")
        
        # Performance comparison
        embedding_performance = completed_embedding.groupby('model_type').agg({
            'regression_r2': ['mean', 'std', 'min', 'max'],
            'classification_f1': ['mean', 'std', 'min', 'max'],
            'n_samples': ['mean', 'std', 'min', 'max']
        }).round(3)
        
        st.dataframe(embedding_performance, use_container_width=True)
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # R2 comparison
            fig_r2 = px.box(
                completed_embedding, 
                x='model_type', 
                y='regression_r2',
                title='R² Score Distribution by Embedding Type'
            )
            st.plotly_chart(fig_r2, use_container_width=True)
        
        with col2:
            # F1 Score comparison
            fig_f1 = px.box(
                completed_embedding, 
                x='model_type', 
                y='classification_f1',
                title='F1 Score Distribution by Embedding Type'
            )
            st.plotly_chart(fig_f1, use_container_width=True)
    else:
        st.warning("No completed ESM or AlphaFold models found for comparison")

def create_introduction_page():
    """Create the Introduction page with Avoidome overview"""
    st.header("Avoidome Overview")
    
    # Introduction text
    st.markdown("""
    The **Avoidome** is a curated list of proteins that should be avoided as drug targets due to their critical physiological functions, 
    potential for adverse effects, or involvement in essential biological processes.
    """)
    
    # Avoidome proteins data
    avoidome_data = {
        "Protein": [
            "CYP1A2", "CYP2B6", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4",
            "AOX1", "XDH", "MAOA", "MAOB", "ALDH1A1", "ADH1A", "HSD11B1", "AKR7A3", "FMO1",
            "SULT1A1", "GSTA1", "AHR", "NR1I3", "NR1I2",
            "SLCO1B1", "SLCO2B1", "SLCO2B3", "ORM1",
            "KCNH2", "SCN5A", "CACNB1", "CAV1", "OXA1L",
            "HTR2B", "SLC6A4", "SLC6A3", "SLC6A2",
            "ADRA1A", "ADRA2A", "ADRB1", "ADRB2",
            "CHRM1", "CHRM2", "CHRM3", "CHRNA10", "CHRNA9", "CHRNA7", "CHRNA3", "CHRNA5",
            "CNRIP1", "CNR2", "HRH1",
            "GABPA", "DIDO1", "SMPDL3A", "NAT8"
        ],
        "Function": [
            "Drug metabolism (oxidation, hydroxylation, demethylation)",
            "Drug metabolism (oxidation, hydroxylation, demethylation)",
            "Drug metabolism (oxidation, hydroxylation, demethylation)",
            "Drug metabolism (oxidation, hydroxylation, demethylation)",
            "Drug metabolism (oxidation, hydroxylation, demethylation)",
            "Drug metabolism (oxidation, hydroxylation, demethylation) - most abundant, metabolizes ~50% of drugs",
            "Aldehyde oxidation, heterocycle metabolism",
            "Xanthine oxidation, purine metabolism",
            "Neurotransmitter degradation (dopamine, serotonin, norepinephrine)",
            "Neurotransmitter degradation (dopamine, serotonin, norepinephrine)",
            "Aldehyde metabolism, ethanol processing",
            "Alcohol metabolism, ethanol processing",
            "Cortisone to cortisol conversion",
            "Aldehyde/ketone detoxification",
            "Soft nucleophile oxidation (amines, sulfur compounds)",
            "Xenobiotic and hormone sulfonation",
            "Glutathione conjugation, detoxification",
            "Xenobiotic sensing, CYP1A regulation",
            "Nuclear receptor, drug-metabolizing enzyme regulation",
            "Nuclear receptor, drug-metabolizing enzyme regulation",
            "Hepatic drug uptake (OATP)",
            "Hepatic drug uptake (OATP)",
            "Hepatic drug uptake (OATP)",
            "Plasma protein binding, pharmacokinetics",
            "Cardiac repolarization (hERG channel)",
            "Cardiac action potential conduction",
            "Calcium influx regulation in excitable tissues",
            "Caveolae scaffolding, signaling, endocytosis",
            "Mitochondrial protein insertion",
            "Serotonin receptor, vascular tone, mood regulation",
            "Serotonin reuptake, antidepressant target",
            "Dopamine reuptake, psychostimulant target",
            "Norepinephrine reuptake",
            "Vasoconstriction, smooth muscle contraction",
            "Neurotransmitter release inhibition, blood pressure regulation",
            "Heart rate and contractility increase",
            "Smooth muscle relaxation, bronchodilation",
            "Muscarinic acetylcholine receptor (M1)",
            "Muscarinic acetylcholine receptor (M2)",
            "Muscarinic acetylcholine receptor (M3)",
            "Nicotinic acetylcholine receptor subunit",
            "Nicotinic acetylcholine receptor subunit",
            "Nicotinic acetylcholine receptor subunit",
            "Nicotinic acetylcholine receptor subunit",
            "Nicotinic acetylcholine receptor subunit",
            "Cannabinoid receptor signaling modulation",
            "Cannabinoid receptor type 2, immune modulation",
            "Histamine H1 receptor, allergic responses",
            "Transcription factor, mitochondrial biogenesis",
            "Apoptosis and chromatin regulation",
            "Lipid metabolism, immune signaling",
            "Amino acid derivative acetylation"
        ],
        "Protein Family": [
            "Cytochrome P450", "Cytochrome P450", "Cytochrome P450", "Cytochrome P450", "Cytochrome P450", "Cytochrome P450",
            "Oxidoreductase", "Oxidoreductase", "Monoamine oxidase", "Monoamine oxidase", "Aldehyde dehydrogenase", "Alcohol dehydrogenase", "Hydroxysteroid dehydrogenase", "Aldo-keto reductase", "Flavin monooxygenase",
            "Sulfotransferase", "Glutathione S-transferase", "Nuclear receptor", "Nuclear receptor", "Nuclear receptor",
            "OATP transporter", "OATP transporter", "OATP transporter", "Plasma protein",
            "Potassium channel", "Sodium channel", "Calcium channel", "Caveolin", "Mitochondrial protein",
            "Serotonin receptor", "SLC transporter", "SLC transporter", "SLC transporter",
            "Adrenergic receptor", "Adrenergic receptor", "Adrenergic receptor", "Adrenergic receptor",
            "Muscarinic receptor", "Muscarinic receptor", "Muscarinic receptor", "Nicotinic receptor", "Nicotinic receptor", "Nicotinic receptor", "Nicotinic receptor", "Nicotinic receptor",
            "Cannabinoid receptor interacting protein", "Cannabinoid receptor", "Histamine receptor",
            "Transcription factor", "Chromatin regulator", "Phosphodiesterase", "Acetyltransferase"
        ]
    }
    
    # Create DataFrame
    avoidome_df = pd.DataFrame(avoidome_data)
    
    # Display the table
    st.subheader("Avoidome Proteins List")
    st.dataframe(avoidome_df, use_container_width=True)
    
    # Create pie chart for general functional categories
    st.subheader("Functional Category Distribution")
    
    # Map specific families to general categories
    general_categories = {
        'Cytochrome P450': 'Drug Metabolism',
        'Oxidoreductase': 'Drug Metabolism',
        'Monoamine oxidase': 'Drug Metabolism',
        'Aldehyde dehydrogenase': 'Drug Metabolism',
        'Alcohol dehydrogenase': 'Drug Metabolism',
        'Hydroxysteroid dehydrogenase': 'Drug Metabolism',
        'Aldo-keto reductase': 'Drug Metabolism',
        'Flavin monooxygenase': 'Drug Metabolism',
        'Sulfotransferase': 'Drug Metabolism',
        'Glutathione S-transferase': 'Drug Metabolism',
        'Nuclear receptor': 'Regulatory',
        'OATP transporter': 'Transport',
        'Plasma protein': 'Transport',
        'Potassium channel': 'Ion Channels',
        'Sodium channel': 'Ion Channels',
        'Calcium channel': 'Ion Channels',
        'Caveolin': 'Membrane Proteins',
        'Mitochondrial protein': 'Membrane Proteins',
        'Serotonin receptor': 'Neurotransmitter Receptors',
        'SLC transporter': 'Neurotransmitter Transport',
        'Adrenergic receptor': 'Neurotransmitter Receptors',
        'Muscarinic receptor': 'Neurotransmitter Receptors',
        'Nicotinic receptor': 'Neurotransmitter Receptors',
        'Cannabinoid receptor interacting protein': 'Cell Signaling',
        'Cannabinoid receptor': 'Other Receptors',
        'Histamine receptor': 'Other Receptors',
        'Transcription factor': 'Cell Signaling',
        'Chromatin regulator': 'Cell Signaling',
        'Phosphodiesterase': 'Cell Signaling',
        'Acetyltransferase': 'Cell Signaling'
    }
    
    # Add general category column
    avoidome_df['General Category'] = avoidome_df['Protein Family'].map(general_categories)
    
    # Count general categories
    category_counts = avoidome_df['General Category'].value_counts()
    
    fig_pie = go.Figure(data=[go.Pie(
        labels=category_counts.index,
        values=category_counts.values,
        hole=0.3,
        textinfo='label+percent',
        textposition='auto'
    )])
    
    fig_pie.update_layout(
        title="Distribution of Avoidome Proteins by Functional Category",
        font=dict(size=12),
        height=500
    )
    
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Detailed sections
    st.subheader("Drug-Metabolizing Enzymes (Phase I & II)")
    
    st.markdown("""
    **Cytochrome P450s (CYPs):**
    - CYP1A2, CYP2B6, CYP2C9, CYP2C19, CYP2D6, CYP3A4 – Major enzymes of drug metabolism (oxidation, hydroxylation, demethylation)
    - CYP3A4 is the most abundant and metabolizes ~50% of drugs
    
    **Other oxidoreductases:**
    - AOX1 (Aldehyde oxidase 1), XDH (Xanthine dehydrogenase) – Catalyze oxidation of aldehydes and heterocycles, important in drug clearance
    - MAOA, MAOB – Mitochondrial monoamine oxidases, degrade neurotransmitters (dopamine, serotonin, norepinephrine)
    - ALDH1A1 (Aldehyde dehydrogenase), ADH1A (Alcohol dehydrogenase) – Involved in ethanol and aldehyde metabolism
    - HSD11B1 (11β-Hydroxysteroid dehydrogenase type 1) – Converts cortisone ↔ cortisol
    - AKR7A3 (Aldo-keto reductase) – Detoxification of aldehydes/ketones
    - FMO1 (Flavin-containing monooxygenase 1) – Oxidizes soft nucleophiles (amines, sulfur compounds)
    
    **Phase II conjugation & detox enzymes:**
    - SULT1A1 (Sulfotransferase) – Sulfonation of xenobiotics and hormones
    - GSTA1 (Glutathione S-transferase) – Detoxification via glutathione conjugation
    """)
    
    st.subheader("Nuclear Receptors / Regulators")
    
    st.markdown("""
    - AHR (Aryl hydrocarbon receptor) – Senses xenobiotics, regulates CYP1A expression
    - NR1I3 (CAR), NR1I2 (PXR) – Nuclear receptors controlling expression of drug-metabolizing enzymes and transporters
    """)
    
    st.subheader("Transporters")
    
    st.markdown("""
    - SLCO1B1, SLCO2B1, SLCO2B3 – Organic anion transporting polypeptides (OATPs), mediate hepatic and intestinal drug uptake
    - ORM1 (Orosomucoid/α1-acid glycoprotein) – Plasma protein binding drugs, affecting pharmacokinetics
    """)
    
    st.subheader("Cardiac & Membrane Proteins")
    
    st.markdown("""
    - KCNH2 (hERG potassium channel) – Critical for cardiac repolarization; drug target for arrhythmia risk
    - SCN5A (Cardiac sodium channel) – Essential for cardiac action potential conduction
    - CACNB1 (Voltage-gated Ca²⁺ channel subunit) – Regulates calcium influx in excitable tissues
    - CAV1 (Caveolin-1) – Scaffolding protein in caveolae; involved in signaling and endocytosis
    - OXA1L – Mitochondrial inner membrane protein, important in inserting proteins into the membrane
    """)
    
    st.subheader("Neurotransmitter Receptors & Transporters")
    
    st.markdown("""
    - HTR2B (5-HT2B receptor) – Serotonin receptor, involved in vascular tone and mood regulation
    - SLC6A4 (Serotonin transporter, SERT) – Reuptake of serotonin, antidepressant target (SSRIs)
    - SLC6A3 (Dopamine transporter, DAT) – Reuptake of dopamine, target of psychostimulants
    - SLC6A2 (Norepinephrine transporter, NET) – Reuptake of norepinephrine
    """)
    
    st.subheader("Adrenergic Receptors")
    
    st.markdown("""
    - ADRA1A (α1-adrenergic receptor) – Vasoconstriction, smooth muscle contraction
    - ADRA2A (α2-adrenergic receptor) – Inhibits neurotransmitter release, regulates blood pressure
    - ADRB1 (β1-adrenergic receptor) – Increases heart rate and contractility
    - ADRB2 (β2-adrenergic receptor) – Smooth muscle relaxation, bronchodilation
    """)
    
    st.subheader("Cholinergic Receptors")
    
    st.markdown("""
    - CHRM1, CHRM2, CHRM3 – Muscarinic acetylcholine receptors (M1–M3), regulate smooth muscle, heart, CNS functions
    - CHRNA10, CHRNA9, CHRNA7, CHRNA3, CHRNA5 – Nicotinic acetylcholine receptor subunits, ligand-gated ion channels in CNS and PNS
    """)
    
    st.subheader("Cannabinoid Pathway")
    
    st.markdown("""
    - CNRIP1 (Cannabinoid receptor interacting protein) – Modulates cannabinoid receptor function
    - CNR2 (Cannabinoid receptor type 2) – Expressed in immune system, modulates inflammation
    """)
    
    st.subheader("Histamine & Other Receptors")
    
    st.markdown("""
    - HRH1 (Histamine H1 receptor) – Mediates allergic responses, targeted by antihistamines
    """)
    
    st.subheader("Transcription Factors / Cell Signaling")
    
    st.markdown("""
    - GABPA (GA-binding protein α) – Transcription factor, regulates mitochondrial biogenesis and immune genes
    - DIDO1 (Death-inducer obliterator 1) – Apoptosis and chromatin regulation
    - SMPDL3A (Sphingomyelin phosphodiesterase acid-like 3A) – Involved in lipid metabolism, immune signaling
    - NAT8 (N-acetyltransferase 8) – Acetylation of amino acid derivatives
    """)

def main():
    """Main dashboard function"""
    st.markdown('<h1 class="main-header">QSAR Modeling Dashboard - Avoidome</h1>', unsafe_allow_html=True)
    
    # Load data
    data_df = load_sample_data()
    
    if data_df is None:
        st.error("Unable to load QSAR modeling data. Please ensure the standardized_qsar_models directory exists and contains the required files.")
        return
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    main_category = st.sidebar.selectbox("Select Category:", list(PAGES.keys()))
    
    if main_category in PAGES:
        sub_pages = PAGES[main_category]
        sub_category = st.sidebar.selectbox("Select Section:", list(sub_pages.keys()))
        current_page = sub_pages[sub_category]
    else:
        current_page = "qsar_model_overview"
    
    # Display current page
    if current_page == "introduction":
        create_introduction_page()
    
    # QSAR Standard pages
    elif current_page == "qsar_model_overview":
        create_qsar_model_overview()
    
    elif current_page == "qsar_summary_statistics":
        create_qsar_summary_statistics()
    
    elif current_page == "qsar_model_performance":
        create_qsar_model_performance()
    
    elif current_page == "qsar_cross_organism":
        create_qsar_cross_organism()
    
    elif current_page == "qsar_sample_size_analysis":
        create_qsar_sample_size_analysis()
    
    elif current_page == "qsar_embedding_comparison":
        create_qsar_embedding_comparison()
    
    # Legacy pages (keeping for backward compatibility)
    elif current_page == "overview_summary":
        st.header("Summary Statistics")
        create_summary_metrics(data_df)
        
        st.header("Model Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            create_organism_distribution_chart(data_df)
        
        with col2:
            create_model_type_distribution(data_df)
        
    
    elif current_page == "overview_distribution":
        st.header("Data Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            create_sample_size_analysis(data_df)
        
        with col2:
            # Sample size by organism
            completed_df = data_df[data_df['status'] == 'completed']
            if len(completed_df) > 0:
                org_samples = completed_df.groupby('organism')['n_samples'].agg(['mean', 'median', 'std']).round(2)
                st.subheader("Sample Size Statistics by Organism")
                st.dataframe(org_samples, use_container_width=True)
    
    elif current_page == "overview_performance":
        st.header("Performance Overview")
        create_performance_heatmap(data_df)
        create_top_performers_table(data_df)
    
    elif current_page == "protein_analysis":
        st.header("Protein Analysis")
        
        # Protein selection
        completed_df = data_df[data_df['status'] == 'completed']
        if len(completed_df) > 0:
            proteins = completed_df['protein_name'].unique()
            selected_protein = st.selectbox("Select Protein:", sorted(proteins))
            
            protein_data = completed_df[completed_df['protein_name'] == selected_protein]
            
            st.subheader(f"Analysis for {selected_protein}")
            
            # Display protein metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Models", len(protein_data))
            
            with col2:
                total_samples = protein_data['n_samples'].sum()
                st.metric("Total Samples", f"{total_samples:,}")
            
            with col3:
                avg_r2 = protein_data['regression_r2'].mean()
                st.metric("Average R²", f"{avg_r2:.3f}")
            
            # Display detailed table
            st.subheader("Model Details")
            st.dataframe(protein_data, use_container_width=True)
    
    elif current_page == "organism_analysis":
        st.header("Organism Analysis")
        
        completed_df = data_df[data_df['status'] == 'completed']
        if len(completed_df) > 0:
            organisms = completed_df['organism'].unique()
            
            for organism in sorted(organisms):
                org_data = completed_df[completed_df['organism'] == organism]
                
                st.subheader(f"{organism.title()} Models")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Models", len(org_data))
                
                with col2:
                    total_samples = org_data['n_samples'].sum()
                    st.metric("Total Samples", f"{total_samples:,}")
                
                with col3:
                    avg_r2 = org_data['regression_r2'].mean()
                    st.metric("Avg R²", f"{avg_r2:.3f}")
                
                with col4:
                    avg_acc = org_data['classification_accuracy'].mean()
                    st.metric("Avg Accuracy", f"{avg_acc:.3f}")
    
    elif current_page == "cross_organism":
        st.header("Cross-Organism Comparison")
        create_performance_comparison(data_df)
        create_classification_performance(data_df)
    
    elif current_page == "morgan_models":
        st.header("Morgan Models")
        
        morgan_data = data_df[data_df['model_type'] == 'Morgan']
        completed_morgan = morgan_data[morgan_data['status'] == 'completed']
        
        if len(completed_morgan) > 0:
            st.subheader("Morgan Model Performance")
            
            # Performance metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                reg_models = completed_morgan[completed_morgan['model_subtype'] == 'Regression']
                avg_r2 = reg_models['regression_r2'].mean()
                st.metric("Avg Regression R²", f"{avg_r2:.3f}")
            
            with col2:
                class_models = completed_morgan[completed_morgan['model_subtype'] == 'Classification']
                avg_f1 = class_models['classification_f1'].mean()
                st.metric("Avg Classification F1 Score", f"{avg_f1:.3f}")
            
            with col3:
                total_samples = completed_morgan['n_samples'].sum()
                st.metric("Total Samples", f"{total_samples:,}")
            
            # Interactive visualizations
            st.subheader("Performance Visualizations")
            
            # Create tabs for different plot types
            tab1, tab2, tab3, tab4 = st.tabs(["Performance by Organism", "Sample Size Analysis", "Top Performers", "Model Details"])
            
            with tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    # R² by organism
                    fig_r2 = px.box(completed_morgan, x='organism', y='regression_r2', 
                                   title='R² Score Distribution by Organism',
                                   color='organism', color_discrete_sequence=px.colors.qualitative.Set2)
                    fig_r2.update_layout(showlegend=False)
                    st.plotly_chart(fig_r2, use_container_width=True)
                
                with col2:
                    # F1 Score by organism
                    fig_f1 = px.box(completed_morgan, x='organism', y='classification_f1', 
                                    title='F1 Score Distribution by Organism',
                                    color='organism', color_discrete_sequence=px.colors.qualitative.Set2)
                    fig_f1.update_layout(showlegend=False)
                    st.plotly_chart(fig_f1, use_container_width=True)
            
            with tab2:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Sample size vs R²
                    fig_scatter_r2 = px.scatter(completed_morgan, x='n_samples', y='regression_r2',
                                               color='organism', size='classification_f1',
                                               hover_data=['protein_name'],
                                               title='Sample Size vs R² Score',
                                               labels={'n_samples': 'Number of Samples', 'regression_r2': 'R² Score'})
                    st.plotly_chart(fig_scatter_r2, use_container_width=True)
                
                with col2:
                    # Sample size vs F1 Score
                    fig_scatter_f1 = px.scatter(completed_morgan, x='n_samples', y='classification_f1',
                                                color='organism', size='regression_r2',
                                                hover_data=['protein_name'],
                                                title='Sample Size vs F1 Score',
                                                labels={'n_samples': 'Number of Samples', 'classification_f1': 'F1 Score'})
                    st.plotly_chart(fig_scatter_f1, use_container_width=True)
            
            with tab3:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Top regression models
                    top_reg = completed_morgan.nlargest(10, 'regression_r2')
                    fig_top_reg = px.bar(top_reg, x='regression_r2', y='protein_name',
                                       color='organism', orientation='h',
                                       title='Top 10 Regression Models (R²)',
                                       labels={'regression_r2': 'R² Score', 'protein_name': 'Protein'})
                    fig_top_reg.update_layout(height=400)
                    st.plotly_chart(fig_top_reg, use_container_width=True)
                
                with col2:
                    # Top classification models
                    top_class = completed_morgan.nlargest(10, 'classification_f1')
                    fig_top_class = px.bar(top_class, x='classification_f1', y='protein_name',
                                         color='organism', orientation='h',
                                         title='Top 10 Classification Models (F1 Score)',
                                         labels={'classification_f1': 'F1 Score', 'protein_name': 'Protein'})
                    fig_top_class.update_layout(height=400)
                    st.plotly_chart(fig_top_class, use_container_width=True)
            
            with tab4:
                # Display models
                st.subheader("All Morgan Models")
                st.dataframe(completed_morgan, use_container_width=True)
    
    elif current_page == "esm_morgan_models":
        st.header("ESM+Morgan Models")
        
        esm_data = data_df[data_df['model_type'] == 'ESM+Morgan']
        completed_esm = esm_data[esm_data['status'] == 'completed']
        
        if len(completed_esm) > 0:
            st.subheader("ESM+Morgan Model Performance")
            
            # Performance metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                reg_models = completed_esm[completed_esm['model_subtype'] == 'Regression']
                avg_r2 = reg_models['regression_r2'].mean()
                st.metric("Avg Regression R²", f"{avg_r2:.3f}")
            
            with col2:
                class_models = completed_esm[completed_esm['model_subtype'] == 'Classification']
                avg_f1 = class_models['classification_f1'].mean()
                st.metric("Avg Classification F1 Score", f"{avg_f1:.3f}")
            
            with col3:
                total_samples = completed_esm['n_samples'].sum()
                st.metric("Total Samples", f"{total_samples:,}")
            
            # Interactive visualizations
            st.subheader("Performance Visualizations")
            
            # Create tabs for different plot types
            tab1, tab2, tab3, tab4 = st.tabs(["Performance by Organism", "Sample Size Analysis", "Top Performers", "Model Details"])
            
            with tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    # R² by organism
                    fig_r2 = px.box(completed_esm, x='organism', y='regression_r2', 
                                   title='R² Score Distribution by Organism',
                                   color='organism', color_discrete_sequence=px.colors.qualitative.Set3)
                    fig_r2.update_layout(showlegend=False)
                    st.plotly_chart(fig_r2, use_container_width=True)
                
                with col2:
                    # F1 Score by organism
                    fig_f1 = px.box(completed_esm, x='organism', y='classification_f1', 
                                    title='F1 Score Distribution by Organism',
                                    color='organism', color_discrete_sequence=px.colors.qualitative.Set3)
                    fig_f1.update_layout(showlegend=False)
                    st.plotly_chart(fig_f1, use_container_width=True)
            
            with tab2:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Sample size vs R²
                    fig_scatter_r2 = px.scatter(completed_esm, x='n_samples', y='regression_r2',
                                               color='organism', size='classification_f1',
                                               hover_data=['protein_name'],
                                               title='Sample Size vs R² Score',
                                               labels={'n_samples': 'Number of Samples', 'regression_r2': 'R² Score'})
                    st.plotly_chart(fig_scatter_r2, use_container_width=True)
                
                with col2:
                    # Sample size vs F1 Score
                    fig_scatter_f1 = px.scatter(completed_esm, x='n_samples', y='classification_f1',
                                                color='organism', size='regression_r2',
                                                hover_data=['protein_name'],
                                                title='Sample Size vs F1 Score',
                                                labels={'n_samples': 'Number of Samples', 'classification_f1': 'F1 Score'})
                    st.plotly_chart(fig_scatter_f1, use_container_width=True)
            
            with tab3:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Top regression models
                    top_reg = completed_esm.nlargest(10, 'regression_r2')
                    fig_top_reg = px.bar(top_reg, x='regression_r2', y='protein_name',
                                       color='organism', orientation='h',
                                       title='Top 10 Regression Models (R²)',
                                       labels={'regression_r2': 'R² Score', 'protein_name': 'Protein'})
                    fig_top_reg.update_layout(height=400)
                    st.plotly_chart(fig_top_reg, use_container_width=True)
                
                with col2:
                    # Top classification models
                    top_class = completed_esm.nlargest(10, 'classification_f1')
                    fig_top_class = px.bar(top_class, x='classification_f1', y='protein_name',
                                         color='organism', orientation='h',
                                         title='Top 10 Classification Models (F1 Score)',
                                         labels={'classification_f1': 'F1 Score', 'protein_name': 'Protein'})
                    fig_top_class.update_layout(height=400)
                    st.plotly_chart(fig_top_class, use_container_width=True)
            
            with tab4:
                # Display models
                st.subheader("All ESM+Morgan Models")
                st.dataframe(completed_esm, use_container_width=True)
    
    elif current_page == "alphafold_morgan_models":
        st.header("AlphaFold+Morgan Models")
        
        alphafold_data = data_df[data_df['model_type'] == 'AlphaFold+Morgan']
        completed_alphafold = alphafold_data[alphafold_data['status'] == 'completed']
        
        if len(completed_alphafold) > 0:
            st.subheader("AlphaFold+Morgan Model Performance")
            
            # Performance metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                reg_models = completed_alphafold[completed_alphafold['model_subtype'] == 'Regression']
                avg_r2 = reg_models['regression_r2'].mean()
                st.metric("Avg Regression R²", f"{avg_r2:.3f}")
            
            with col2:
                class_models = completed_alphafold[completed_alphafold['model_subtype'] == 'Classification']
                avg_f1 = class_models['classification_f1'].mean()
                st.metric("Avg Classification F1 Score", f"{avg_f1:.3f}")
            
            with col3:
                total_samples = completed_alphafold['n_samples'].sum()
                st.metric("Total Samples", f"{total_samples:,}")
            
            # Interactive visualizations
            st.subheader("Performance Visualizations")
            
            # Create tabs for different plot types
            tab1, tab2, tab3, tab4 = st.tabs(["Performance Analysis", "Sample Size Analysis", "Top Performers", "Model Details"])
            
            with tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    # R² distribution
                    fig_r2 = px.histogram(completed_alphafold, x='regression_r2', 
                                       title='R² Score Distribution',
                                       nbins=20, color_discrete_sequence=['#2E8B57'])
                    fig_r2.update_layout(showlegend=False)
                    st.plotly_chart(fig_r2, use_container_width=True)
                
                with col2:
                    # F1 Score distribution
                    fig_f1 = px.histogram(completed_alphafold, x='classification_f1', 
                                        title='F1 Score Distribution',
                                        nbins=20, color_discrete_sequence=['#2E8B57'])
                    fig_f1.update_layout(showlegend=False)
                    st.plotly_chart(fig_f1, use_container_width=True)
            
            with tab2:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Sample size vs R²
                    # Filter for regression data only and clean NaN values
                    reg_data = completed_alphafold[completed_alphafold['model_subtype'] == 'Regression'].copy()
                    reg_data = reg_data.dropna(subset=['n_samples', 'regression_r2'])
                    
                    if len(reg_data) > 0:
                        fig_scatter_r2 = px.scatter(reg_data, x='n_samples', y='regression_r2',
                                                   hover_data=['protein_name'],
                                                   title='Sample Size vs R² Score',
                                                   labels={'n_samples': 'Number of Samples', 'regression_r2': 'R² Score'},
                                                   color_discrete_sequence=['#2E8B57'])
                        st.plotly_chart(fig_scatter_r2, use_container_width=True)
                    else:
                        st.warning("No valid regression data found for scatter plot")
                
                with col2:
                    # Sample size vs F1 Score
                    # Filter for classification data only and clean NaN values
                    class_data = completed_alphafold[completed_alphafold['model_subtype'] == 'Classification'].copy()
                    class_data = class_data.dropna(subset=['n_samples', 'classification_f1'])
                    
                    if len(class_data) > 0:
                        fig_scatter_f1 = px.scatter(class_data, x='n_samples', y='classification_f1',
                                                    hover_data=['protein_name'],
                                                    title='Sample Size vs F1 Score',
                                                    labels={'n_samples': 'Number of Samples', 'classification_f1': 'F1 Score'},
                                                    color_discrete_sequence=['#2E8B57'])
                        st.plotly_chart(fig_scatter_f1, use_container_width=True)
                    else:
                        st.warning("No valid classification data found for scatter plot")
            
            with tab3:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Top regression models
                    top_reg = completed_alphafold.nlargest(10, 'regression_r2')
                    fig_top_reg = px.bar(top_reg, x='regression_r2', y='protein_name',
                                       orientation='h',
                                       title='Top 10 Regression Models (R²)',
                                       labels={'regression_r2': 'R² Score', 'protein_name': 'Protein'},
                                       color_discrete_sequence=['#2E8B57'])
                    fig_top_reg.update_layout(height=400)
                    st.plotly_chart(fig_top_reg, use_container_width=True)
                
                with col2:
                    # Top classification models
                    top_class = completed_alphafold.nlargest(10, 'classification_accuracy')
                    fig_top_class = px.bar(top_class, x='classification_accuracy', y='protein_name',
                                         orientation='h',
                                         title='Top 10 Classification Models (Accuracy)',
                                         labels={'classification_accuracy': 'Accuracy', 'protein_name': 'Protein'},
                                         color_discrete_sequence=['#2E8B57'])
                    fig_top_class.update_layout(height=400)
                    st.plotly_chart(fig_top_class, use_container_width=True)
            
            with tab4:
                # Display models
                st.subheader("All AlphaFold+Morgan Models")
                st.dataframe(completed_alphafold, use_container_width=True)
        else:
            st.warning("No completed AlphaFold+Morgan models found")
    
    elif current_page == "aqse_models":
        st.header("AQSE Models")
        
        # Load AQSE data
        aqse_df = load_aqse_results()
        
        if aqse_df is None:
            st.error("AQSE model data not found. Please run the AQSE workflow first.")
            return
        
        # Display summary metrics
        create_aqse_summary_metrics(aqse_df)
        
        # Load standard QSAR data for comparison
        standard_df = load_standard_qsar_results()
        
        # Create tabs for different analysis types
        tab1, tab2, tab3, tab4 = st.tabs(["Performance Comparison", "Dumbbell Plot", "Threshold Analysis", "Model Details"])
        
        with tab1:
            # Performance comparison
            create_aqse_performance_comparison(aqse_df)
        
        with tab2:
            # Dumbbell plot comparison with standard QSAR
            if standard_df is not None:
                create_aqse_dumbbell_plot(aqse_df, standard_df)
            else:
                st.warning("Standard QSAR data not available for comparison. Please ensure ESM+Morgan models are available.")
        
        with tab3:
            # Threshold analysis
            create_aqse_threshold_analysis(aqse_df)
        
        with tab4:
            # Model details table
            st.subheader("AQSE Model Details")
            
            # Create subtabs for different model types
            subtab1, subtab2, subtab3 = st.tabs(["All Models", "Threshold Models Only", "Top Performers"])
            
            with subtab1:
                st.dataframe(aqse_df, use_container_width=True)
            
            with subtab2:
                threshold_df = aqse_df[aqse_df['model_type'].str.contains('Threshold', na=False)]
                if len(threshold_df) > 0:
                    st.dataframe(threshold_df, use_container_width=True)
                else:
                    st.warning("No threshold AQSE models found")
            
            with subtab3:
                # Top performing models
                if len(aqse_df) > 0:
                    # Sort by R² score
                    top_models = aqse_df.nlargest(10, 'r2')[
                        ['target_name', 'uniprot_id', 'model_type', 'threshold', 'r2', 'q2', 'rmse', 'n_train', 'n_similar_proteins']
                    ]
                    st.dataframe(top_models, use_container_width=True)
                else:
                    st.warning("No AQSE models found")
    
    elif current_page == "model_comparison":
        st.header("Model Comparison")
        
        completed_df = data_df[data_df['status'] == 'completed']
        
        if len(completed_df) > 0:
            # Create tabs for different comparison types
            tab1, tab2, tab3, tab4 = st.tabs(["Regression Comparison", "Classification Comparison", "Side-by-Side Plots", "Statistical Summary"])
            
            with tab1:
                st.subheader("Regression Model Comparison")
                
                reg_df = completed_df[completed_df['model_subtype'] == 'Regression']
                
                if len(reg_df) > 0:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # R² comparison box plot
                        fig_r2 = px.box(reg_df, x='model_type', y='regression_r2',
                                       title='R² Score Comparison',
                                       color='model_type', color_discrete_sequence=px.colors.qualitative.Set1)
                        st.plotly_chart(fig_r2, use_container_width=True)
                    
                    with col2:
                        # RMSE comparison box plot
                        fig_rmse = px.box(reg_df, x='model_type', y='regression_rmse',
                                         title='RMSE Comparison',
                                         color='model_type', color_discrete_sequence=px.colors.qualitative.Set1)
                        st.plotly_chart(fig_rmse, use_container_width=True)
                    
                    # Statistical summary
                    comparison_data = reg_df.groupby('model_type').agg({
                        'regression_r2': ['mean', 'std', 'count'],
                        'regression_rmse': ['mean', 'std']
                    }).round(3)
                    st.dataframe(comparison_data, use_container_width=True)
            
            with tab2:
                st.subheader("Classification Model Comparison")
                
                class_df = completed_df[completed_df['model_subtype'] == 'Classification']
                
                if len(class_df) > 0:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Accuracy comparison box plot
                        fig_acc = px.box(class_df, x='model_type', y='classification_accuracy',
                                        title='Accuracy Comparison',
                                        color='model_type', color_discrete_sequence=px.colors.qualitative.Set1)
                        st.plotly_chart(fig_acc, use_container_width=True)
                    
                    with col2:
                        # F1 score comparison box plot
                        fig_f1 = px.box(class_df, x='model_type', y='classification_f1',
                                       title='F1 Score Comparison',
                                       color='model_type', color_discrete_sequence=px.colors.qualitative.Set1)
                        st.plotly_chart(fig_f1, use_container_width=True)
                    
                    # Statistical summary
                    class_comparison = class_df.groupby('model_type').agg({
                        'classification_accuracy': ['mean', 'std', 'count'],
                        'classification_f1': ['mean', 'std'],
                        'classification_auc': ['mean', 'std']
                    }).round(3)
                    st.dataframe(class_comparison, use_container_width=True)
            
            with tab3:
                st.subheader("Side-by-Side Performance Comparison")
                
                # Combined performance comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    # R² by model type and organism
                    reg_df = completed_df[completed_df['model_subtype'] == 'Regression']
                    if len(reg_df) > 0:
                        fig_r2_org = px.box(reg_df, x='model_type', y='regression_r2', color='organism',
                                           title='R² Score by Model Type and Organism',
                                           color_discrete_sequence=px.colors.qualitative.Set2)
                        st.plotly_chart(fig_r2_org, use_container_width=True)
                
                with col2:
                    # Accuracy by model type and organism
                    class_df = completed_df[completed_df['model_subtype'] == 'Classification']
                    if len(class_df) > 0:
                        fig_acc_org = px.box(class_df, x='model_type', y='classification_accuracy', color='organism',
                                            title='Accuracy by Model Type and Organism',
                                            color_discrete_sequence=px.colors.qualitative.Set2)
                        st.plotly_chart(fig_acc_org, use_container_width=True)
                
                # Sample size comparison
                st.subheader("Sample Size Distribution by Model Type")
                fig_samples = px.box(completed_df, x='model_type', y='n_samples', color='organism',
                                    title='Sample Size Distribution by Model Type and Organism',
                                    color_discrete_sequence=px.colors.qualitative.Set2)
                st.plotly_chart(fig_samples, use_container_width=True)
            
            with tab4:
                st.subheader("Statistical Summary Tables")
                
                # Regression summary
                reg_df = completed_df[completed_df['model_subtype'] == 'Regression']
                if len(reg_df) > 0:
                    st.write("**Regression Models Summary**")
                    reg_summary = reg_df.groupby('model_type').agg({
                        'regression_r2': ['mean', 'std', 'min', 'max', 'count'],
                        'regression_rmse': ['mean', 'std', 'min', 'max'],
                        'n_samples': ['mean', 'std', 'min', 'max']
                    }).round(3)
                    st.dataframe(reg_summary, use_container_width=True)
                
                # Classification summary
                class_df = completed_df[completed_df['model_subtype'] == 'Classification']
                if len(class_df) > 0:
                    st.write("**Classification Models Summary**")
                    class_summary = class_df.groupby('model_type').agg({
                        'classification_accuracy': ['mean', 'std', 'min', 'max', 'count'],
                        'classification_f1': ['mean', 'std', 'min', 'max'],
                        'classification_auc': ['mean', 'std', 'min', 'max'],
                        'n_samples': ['mean', 'std', 'min', 'max']
                    }).round(3)
                    st.dataframe(class_summary, use_container_width=True)
    
    elif current_page == "embedding_comparison":
        st.header("ESM vs AlphaFold Embedding Comparison")
        
        # Filter for ESM and AlphaFold models only
        embedding_df = data_df[data_df['model_type'].isin(['ESM+Morgan', 'AlphaFold+Morgan'])]
        completed_embedding = embedding_df[embedding_df['status'] == 'completed']
        
        st.write(f"Found {len(completed_embedding)} completed models for comparison")
        st.write(f"Model types: {completed_embedding['model_type'].value_counts().to_dict()}")
        
        if len(completed_embedding) > 0:
            st.subheader("Embedding Performance Comparison")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                esm_models = completed_embedding[completed_embedding['model_type'] == 'ESM+Morgan']
                esm_count = len(esm_models)
                st.metric("ESM Models", esm_count)
            
            with col2:
                alphafold_models = completed_embedding[completed_embedding['model_type'] == 'AlphaFold+Morgan']
                af_count = len(alphafold_models)
                st.metric("AlphaFold Models", af_count)
            
            with col3:
                if len(esm_models) > 0:
                    esm_reg = esm_models[esm_models['model_subtype'] == 'Regression']
                    if len(esm_reg) > 0 and 'regression_r2' in esm_reg.columns:
                        esm_avg_r2 = esm_reg['regression_r2'].mean()
                        st.metric("ESM Avg R²", f"{esm_avg_r2:.3f}")
                    else:
                        st.metric("ESM Avg R²", "N/A")
                else:
                    st.metric("ESM Avg R²", "N/A")
            
            with col4:
                if len(alphafold_models) > 0:
                    af_reg = alphafold_models[alphafold_models['model_subtype'] == 'Regression']
                    if len(af_reg) > 0 and 'regression_r2' in af_reg.columns:
                        af_avg_r2 = af_reg['regression_r2'].mean()
                        st.metric("AlphaFold Avg R²", f"{af_avg_r2:.3f}")
                    else:
                        st.metric("AlphaFold Avg R²", "N/A")
                else:
                    st.metric("AlphaFold Avg R²", "N/A")
            
            # Create tabs for different comparison types
            tab1, tab2, tab3, tab4 = st.tabs(["Direct Comparison", "Performance Analysis", "Protein-Level Comparison", "Statistical Tests"])
            
            with tab1:
                st.subheader("Direct Performance Comparison")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # R² comparison
                    reg_df = completed_embedding[completed_embedding['model_subtype'] == 'Regression']
                    if len(reg_df) > 0 and 'regression_r2' in reg_df.columns:
                        # Remove any rows with NaN values
                        reg_df_clean = reg_df.dropna(subset=['regression_r2'])
                        if len(reg_df_clean) > 0:
                            fig_r2 = px.box(reg_df_clean, x='model_type', y='regression_r2',
                                           title='R² Score: ESM vs AlphaFold',
                                           color='model_type',
                                           color_discrete_map={'ESM+Morgan': '#1f77b4', 'AlphaFold+Morgan': '#2E8B57'})
                            st.plotly_chart(fig_r2, use_container_width=True)
                        else:
                            st.warning("No valid R² data found for comparison")
                    else:
                        st.warning("No regression data found for comparison")
                
                with col2:
                    # Accuracy comparison
                    class_df = completed_embedding[completed_embedding['model_subtype'] == 'Classification']
                    if len(class_df) > 0 and 'classification_accuracy' in class_df.columns:
                        # Remove any rows with NaN values
                        class_df_clean = class_df.dropna(subset=['classification_accuracy'])
                        if len(class_df_clean) > 0:
                            fig_acc = px.box(class_df_clean, x='model_type', y='classification_accuracy',
                                            title='Accuracy: ESM vs AlphaFold',
                                            color='model_type',
                                            color_discrete_map={'ESM+Morgan': '#1f77b4', 'AlphaFold+Morgan': '#2E8B57'})
                            st.plotly_chart(fig_acc, use_container_width=True)
                        else:
                            st.warning("No valid accuracy data found for comparison")
                    else:
                        st.warning("No classification data found for comparison")
            
            with tab2:
                st.subheader("Performance Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # R² vs Sample Size
                    reg_df = completed_embedding[completed_embedding['model_subtype'] == 'Regression']
                    if len(reg_df) > 0:
                        # Clean data by removing NaN values
                        reg_df_clean = reg_df.dropna(subset=['n_samples', 'regression_r2'])
                        if len(reg_df_clean) > 0:
                            fig_scatter = px.scatter(reg_df_clean, x='n_samples', y='regression_r2',
                                                   color='model_type',
                                                   hover_data=['protein_name'],
                                                   title='Sample Size vs R² Score',
                                                   color_discrete_map={'ESM+Morgan': '#1f77b4', 'AlphaFold+Morgan': '#2E8B57'})
                            st.plotly_chart(fig_scatter, use_container_width=True)
                        else:
                            st.warning("No valid regression data found for scatter plot")
                    else:
                        st.warning("No regression data found")
                
                with col2:
                    # Accuracy vs Sample Size
                    class_df = completed_embedding[completed_embedding['model_subtype'] == 'Classification']
                    if len(class_df) > 0:
                        # Clean data by removing NaN values
                        class_df_clean = class_df.dropna(subset=['n_samples', 'classification_accuracy'])
                        if len(class_df_clean) > 0:
                            fig_scatter_acc = px.scatter(class_df_clean, x='n_samples', y='classification_accuracy',
                                                       color='model_type',
                                                       hover_data=['protein_name'],
                                                       title='Sample Size vs Accuracy',
                                                       color_discrete_map={'ESM+Morgan': '#1f77b4', 'AlphaFold+Morgan': '#2E8B57'})
                            st.plotly_chart(fig_scatter_acc, use_container_width=True)
                        else:
                            st.warning("No valid classification data found for scatter plot")
                    else:
                        st.warning("No classification data found")
            
            with tab3:
                st.subheader("Protein-Level Comparison")
                
                # Get common proteins between ESM and AlphaFold
                esm_proteins = set(esm_models['protein_name'].unique())
                alphafold_proteins = set(alphafold_models['protein_name'].unique())
                common_proteins = esm_proteins.intersection(alphafold_proteins)
                
                if len(common_proteins) > 0:
                    st.write(f"**Common Proteins ({len(common_proteins)}):** {', '.join(sorted(common_proteins))}")
                    
                    # Create comparison table for common proteins
                    comparison_data = []
                    for protein in sorted(common_proteins):
                        esm_reg = esm_models[(esm_models['protein_name'] == protein) & (esm_models['model_subtype'] == 'Regression')]
                        esm_class = esm_models[(esm_models['protein_name'] == protein) & (esm_models['model_subtype'] == 'Classification')]
                        af_reg = alphafold_models[(alphafold_models['protein_name'] == protein) & (alphafold_models['model_subtype'] == 'Regression')]
                        af_class = alphafold_models[(alphafold_models['protein_name'] == protein) & (alphafold_models['model_subtype'] == 'Classification')]
                        
                        comparison_data.append({
                            'Protein': protein,
                            'ESM_R2': esm_reg['regression_r2'].iloc[0] if len(esm_reg) > 0 else None,
                            'AlphaFold_R2': af_reg['regression_r2'].iloc[0] if len(af_reg) > 0 else None,
                            'ESM_Accuracy': esm_class['classification_accuracy'].iloc[0] if len(esm_class) > 0 else None,
                            'AlphaFold_Accuracy': af_class['classification_accuracy'].iloc[0] if len(af_class) > 0 else None,
                            'R2_Difference': (af_reg['regression_r2'].iloc[0] - esm_reg['regression_r2'].iloc[0]) if len(esm_reg) > 0 and len(af_reg) > 0 else None,
                            'Accuracy_Difference': (af_class['classification_accuracy'].iloc[0] - esm_class['classification_accuracy'].iloc[0]) if len(esm_class) > 0 and len(af_class) > 0 else None
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # Summary of differences
                    if 'R2_Difference' in comparison_df.columns:
                        r2_diff_mean = comparison_df['R2_Difference'].mean()
                        r2_diff_std = comparison_df['R2_Difference'].std()
                        st.write(f"**R² Difference (AlphaFold - ESM):** Mean = {r2_diff_mean:.3f}, Std = {r2_diff_std:.3f}")
                    
                    if 'Accuracy_Difference' in comparison_df.columns:
                        acc_diff_mean = comparison_df['Accuracy_Difference'].mean()
                        acc_diff_std = comparison_df['Accuracy_Difference'].std()
                        st.write(f"**Accuracy Difference (AlphaFold - ESM):** Mean = {acc_diff_mean:.3f}, Std = {acc_diff_std:.3f}")
                else:
                    st.warning("No common proteins found between ESM and AlphaFold models")
            
            with tab4:
                st.subheader("Statistical Summary")
                
                # Regression comparison
                reg_df = completed_embedding[completed_embedding['model_subtype'] == 'Regression']
                if len(reg_df) > 0:
                    st.write("**Regression Models Statistical Summary**")
                    reg_summary = reg_df.groupby('model_type').agg({
                        'regression_r2': ['mean', 'std', 'min', 'max', 'count'],
                        'regression_rmse': ['mean', 'std'],
                        'n_samples': ['mean', 'std']
                    }).round(3)
                    st.dataframe(reg_summary, use_container_width=True)
                
                # Classification comparison
                class_df = completed_embedding[completed_embedding['model_subtype'] == 'Classification']
                if len(class_df) > 0:
                    st.write("**Classification Models Statistical Summary**")
                    class_summary = class_df.groupby('model_type').agg({
                        'classification_accuracy': ['mean', 'std', 'min', 'max', 'count'],
                        'classification_f1': ['mean', 'std'],
                        'classification_auc': ['mean', 'std']
                    }).round(3)
                    st.dataframe(class_summary, use_container_width=True)
        else:
            st.warning("No completed ESM or AlphaFold models found for comparison")
    
    elif current_page == "embedding_performance":
        st.header("Embedding Performance Analysis")
        
        # Filter for ESM and AlphaFold models only
        embedding_df = data_df[data_df['model_type'].isin(['ESM+Morgan', 'AlphaFold+Morgan'])]
        completed_embedding = embedding_df[embedding_df['status'] == 'completed']
        
        if len(completed_embedding) > 0:
            st.subheader("Performance Distribution Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # R² distribution comparison
                reg_df = completed_embedding[completed_embedding['model_subtype'] == 'Regression']
                if len(reg_df) > 0:
                    fig_hist = px.histogram(reg_df, x='regression_r2', color='model_type',
                                           title='R² Score Distribution Comparison',
                                           nbins=20, barmode='overlay', opacity=0.7,
                                           color_discrete_map={'ESM+Morgan': '#1f77b4', 'AlphaFold+Morgan': '#2E8B57'})
                    st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                # Accuracy distribution comparison
                class_df = completed_embedding[completed_embedding['model_subtype'] == 'Classification']
                if len(class_df) > 0:
                    fig_hist_acc = px.histogram(class_df, x='classification_accuracy', color='model_type',
                                              title='Accuracy Distribution Comparison',
                                              nbins=20, barmode='overlay', opacity=0.7,
                                              color_discrete_map={'ESM+Morgan': '#1f77b4', 'AlphaFold+Morgan': '#2E8B57'})
                    st.plotly_chart(fig_hist_acc, use_container_width=True)
            
            # Performance correlation analysis
            st.subheader("Performance Correlation Analysis")
            
            # Create correlation matrix for common metrics
            metrics_df = completed_embedding[['model_type', 'regression_r2', 'classification_accuracy', 'n_samples']].dropna()
            
            if len(metrics_df) > 0:
                # Correlation heatmap
                numeric_cols = ['regression_r2', 'classification_accuracy', 'n_samples']
                corr_matrix = metrics_df[numeric_cols].corr()
                
                fig_heatmap = px.imshow(corr_matrix, 
                                       text_auto=True, 
                                       aspect="auto",
                                       title="Performance Metrics Correlation Matrix",
                                       color_continuous_scale='RdBu_r')
                st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.warning("No completed ESM or AlphaFold models found for analysis")
    
    elif current_page == "regression_metrics":
        st.header("Regression Metrics")
        
        completed_df = data_df[data_df['status'] == 'completed']
        reg_df = completed_df[completed_df['model_subtype'] == 'Regression']
        
        if len(reg_df) > 0:
            # R² distribution
            fig = px.histogram(
                reg_df,
                x='regression_r2',
                color='model_type',
                title="R² Score Distribution",
                nbins=20,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # RMSE distribution
            fig = px.histogram(
                reg_df,
                x='regression_rmse',
                color='model_type',
                title="RMSE Distribution",
                nbins=20,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif current_page == "classification_metrics":
        st.header("Classification Metrics")
        
        completed_df = data_df[data_df['status'] == 'completed']
        class_df = completed_df[completed_df['model_subtype'] == 'Classification']
        
        if len(class_df) > 0:
            # Accuracy distribution
            fig = px.histogram(
                class_df,
                x='classification_accuracy',
                color='model_type',
                title="Accuracy Distribution",
                nbins=20,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # F1 score distribution
            fig = px.histogram(
                class_df,
                x='classification_f1',
                color='model_type',
                title="F1 Score Distribution",
                nbins=20,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig, use_container_width=True)
    

    
    elif current_page == "mcc_analysis":
        st.header("MCC Analysis: Regression vs Classification")
        
        # Load MCC analysis data
        summary_df, morgan_df, esm_df = load_mcc_analysis_data()
        
        if summary_df is None:
            st.error("MCC analysis data not found. Please run the MCC analysis first: `python analyses/mcc_comparison/mcc_analysis.py`")
            return
        
        # Display summary metrics
        create_mcc_summary_metrics(summary_df)
        
        # Display win rate comparison
        st.header("Model Performance Comparison")
        create_mcc_win_rate_chart(summary_df)
        
        # Display MCC difference analysis
        create_mcc_difference_analysis(morgan_df, esm_df)
        
        # Display scatter plot comparison
        create_mcc_scatter_plot(morgan_df, esm_df)
        
        # Display detailed results table
        create_mcc_detailed_table(morgan_df, esm_df)
        
        # Display static plots if available
        mcc_plots = load_mcc_plots()
        if mcc_plots:
            st.header("MCC Analysis Visualizations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'mcc_difference_distributions.png' in mcc_plots:
                    st.subheader("MCC Difference Distributions")
                    st.image("analyses/mcc_comparison/plots/mcc_difference_distributions.png", 
                            caption="Distribution of MCC differences between classification and regression models")
            
            with col2:
                if 'win_rate_comparison.png' in mcc_plots:
                    st.subheader("Model Win Rate Comparison")
                    st.image("analyses/mcc_comparison/plots/win_rate_comparison.png", 
                            caption="Comparison of win rates between model approaches")
            
            if 'mcc_scatter_comparison.png' in mcc_plots:
                st.subheader("MCC Scatter Plot Comparison")
                st.image("analyses/mcc_comparison/plots/mcc_scatter_comparison.png", 
                        caption="Scatter plot comparison of regression vs classification MCC scores")
        
        # Display conclusions
        st.header("Key Findings and Recommendations")
        
        if len(summary_df) > 0:
            morgan_row = summary_df[summary_df['model_type'] == 'morgan'].iloc[0] if 'morgan' in summary_df['model_type'].values else None
            esm_row = summary_df[summary_df['model_type'] == 'esm_morgan'].iloc[0] if 'esm_morgan' in summary_df['model_type'].values else None
            
            col1, col2 = st.columns(2)
            
            with col1:
                if morgan_row is not None:
                    st.markdown("**Morgan Models**")
                    better_model = "Classification" if morgan_row['classification_win_rate'] > 0.5 else "Regression"
                    st.success(f"**Recommendation:** Use {better_model} models")
                    st.write(f"- Win Rate: {morgan_row['classification_win_rate']:.1%}")
                    st.write(f"- Mean MCC Improvement: {morgan_row['mean_mcc_difference']:.4f}")
                    st.write(f"- Total Comparisons: {int(morgan_row['total_comparisons'])}")
            
            with col2:
                if esm_row is not None:
                    st.markdown("**ESM+Morgan Models**")
                    better_model = "Classification" if esm_row['classification_win_rate'] > 0.5 else "Regression"
                    st.success(f"**Recommendation:** Use {better_model} models")
                    st.write(f"- Win Rate: {esm_row['classification_win_rate']:.1%}")
                    st.write(f"- Mean MCC Improvement: {esm_row['mean_mcc_difference']:.4f}")
                    st.write(f"- Total Comparisons: {int(esm_row['total_comparisons'])}")
        
        # Display analysis report link
        st.markdown("---")
        st.markdown("**Analysis Report:** [MCC Analysis Report](analyses/mcc_comparison/MCC_Analysis_Report.md)")
    
    # AQSE Analysis sections
    elif current_page == "aqse_workflow":
        create_aqse_workflow_diagram()
    
    
    elif current_page == "threshold_analysis":
        st.header("AQSE Threshold Analysis")
        
        # Load AQSE data
        aqse_df = load_aqse_results()
        
        if aqse_df is None:
            st.error("AQSE model data not found. Please run the AQSE workflow first.")
            return
        
        # Create threshold analysis
        create_aqse_threshold_analysis(aqse_df)
        
        # Additional threshold insights
        st.subheader("Threshold Insights")
        
        threshold_df = aqse_df[aqse_df['model_type'].str.contains('Threshold', na=False)]
        if len(threshold_df) > 0:
            # Extract threshold
            threshold_df = threshold_df.copy()
            threshold_df['threshold'] = threshold_df['model_type'].str.extract(r'\((\w+)\)')
            
            # Summary by threshold
            threshold_summary = threshold_df.groupby('threshold').agg({
                'r2': ['mean', 'std', 'count'],
                'q2': ['mean', 'std'],
                'n_train': ['mean', 'std'],
                'n_similar_proteins': ['mean', 'std']
            }).round(3)
            
            st.dataframe(threshold_summary, use_container_width=True)
    
    elif current_page == "sample_expansion":
        st.header("Sample Expansion Analysis")
        
        # Load data
        aqse_df = load_aqse_results()
        standard_df = load_standard_qsar_results()
        mapping_df = load_protein_mapping()
        
        if aqse_df is None or standard_df is None:
            st.error("Missing data for analysis. Please ensure both AQSE and Standard QSAR models are available.")
            return
        
        # Filter out models with zero sample expansion
        if 'sample_expansion' in aqse_df.columns:
            aqse_df = aqse_df[aqse_df['sample_expansion'] > 0]
            if len(aqse_df) == 0:
                st.warning("No AQSE models with sample expansion found. All models have zero expansion (equivalent to standardized QSAR models).")
                return
        
        st.subheader("Sample Expansion Summary (Models with Expansion)")
        
        # Prepare comparison data
        if mapping_df is not None:
            # First, select only the columns we need from standard_df to avoid duplicates
            standard_subset = standard_df[['protein_name', 'organism', 'status', 'n_samples', 'regression_r2', 'regression_rmse']].copy()
            
            # Then merge with mapping
            standard_mapped = standard_subset.merge(
                mapping_df[['protein_name', 'human_uniprot_id']], 
                on='protein_name', 
                how='left'
            )
            standard_mapped = standard_mapped.rename(columns={'human_uniprot_id': 'uniprot_id'})
        else:
            st.warning("No protein mapping available for analysis")
            return
        
        # Get all AQSE models (not just standard)
        aqse_all = aqse_df.copy()
        
        # Merge on UniProt ID
        expansion_df = aqse_all.merge(
            standard_mapped[['uniprot_id', 'protein_name', 'n_samples']],
            on='uniprot_id',
            how='inner',
            suffixes=('_aqse', '_standard')
        )
        
        if len(expansion_df) == 0:
            st.warning("No common proteins found for expansion analysis")
            return
        
        # Calculate expansion metrics
        expansion_df['sample_expansion'] = expansion_df['n_train'] - expansion_df['n_samples']
        expansion_df['sample_expansion_pct'] = (expansion_df['sample_expansion'] / expansion_df['n_samples']) * 100
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_expansion = expansion_df['sample_expansion'].sum()
            st.metric("Total Sample Expansion", f"{total_expansion:,}")
        
        with col2:
            avg_expansion = expansion_df['sample_expansion'].mean()
            st.metric("Avg Sample Expansion", f"{avg_expansion:.0f}")
        
        with col3:
            avg_expansion_pct = expansion_df['sample_expansion_pct'].mean()
            st.metric("Avg Expansion %", f"{avg_expansion_pct:.1f}%")
        
        with col4:
            models_with_expansion = len(expansion_df[expansion_df['sample_expansion'] > 0])
            st.metric("Models with Expansion", f"{models_with_expansion}/{len(expansion_df)}")
        
        # Expansion by model type
        st.subheader("Sample Expansion by Model Type")
        
        expansion_by_type = expansion_df.groupby('model_type').agg({
            'sample_expansion': ['mean', 'std', 'sum'],
            'sample_expansion_pct': ['mean', 'std'],
            'n_train': ['mean', 'std'],
            'n_similar_proteins': ['mean', 'std']
        }).round(2)
        
        st.dataframe(expansion_by_type, use_container_width=True)
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Sample expansion distribution
            fig_expansion = px.histogram(
                expansion_df,
                x='sample_expansion',
                color='model_type',
                title='Sample Expansion Distribution by Model Type',
                nbins=20,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig_expansion, use_container_width=True)
        
        with col2:
            # Expansion percentage distribution
            fig_pct = px.histogram(
                expansion_df,
                x='sample_expansion_pct',
                color='model_type',
                title='Sample Expansion % Distribution by Model Type',
                nbins=20,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig_pct, use_container_width=True)
        
        # Top expansion models
        st.subheader("Top Sample Expansion Models")
        
        top_expansion = expansion_df.nlargest(10, 'sample_expansion')[
            ['target_name', 'uniprot_id', 'model_type', 'n_samples', 'n_train', 
             'sample_expansion', 'sample_expansion_pct', 'n_similar_proteins']
        ]
        
        top_expansion.columns = [
            'Protein', 'UniProt ID', 'Model Type', 'Standard Samples', 'AQSE Samples',
            'Sample Expansion', 'Expansion %', 'Similar Proteins'
        ]
        
        st.dataframe(top_expansion, use_container_width=True)
        
        # Add the new dumbbell plot for all proteins
        st.markdown("---")
        create_sample_expansion_dumbbell_plot(aqse_df, standard_df)
    
    # Plotting sections
    elif current_page == "plot_distribution":
        st.header("Model Distribution Plots")
        
        png_files, html_files = get_available_plots()
        
        if not png_files:
            st.warning("No plots found. Please run the plotting script first: `python plot_qsar_results.py`")
            return
        
        # Model distribution plots
        col1, col2 = st.columns(2)
        
        with col1:
            if 'model_distribution.png' in png_files:
                st.subheader("Model Distribution Analysis")
                st.image("analyses/standardized_qsar_models/plots/model_distribution.png", 
                        caption="Model distribution across organisms and types")
        
        with col2:
            if 'organism_comparison.png' in png_files:
                st.subheader("Cross-Organism Comparison")
                st.image("analyses/standardized_qsar_models/plots/organism_comparison.png", 
                        caption="Performance comparison across organisms")
    
    elif current_page == "plot_performance":
        st.header("Performance Plots")
        
        png_files, html_files = get_available_plots()
        
        if not png_files:
            st.warning("No plots found. Please run the plotting script first: `python plot_qsar_results.py`")
            return
        
        # Performance metrics plots
        if 'performance_metrics.png' in png_files:
            st.subheader("Performance Metrics Distributions")
            st.image("analyses/standardized_qsar_models/plots/performance_metrics.png", 
                    caption="Distribution of R², RMSE, Accuracy, F1, and AUC scores")
        
        # Sample size analysis
        if 'sample_size_analysis.png' in png_files:
            st.subheader("Sample Size Analysis")
            st.image("analyses/standardized_qsar_models/plots/sample_size_analysis.png", 
                    caption="Sample size distributions and correlations with performance")
    
    elif current_page == "plot_heatmaps":
        st.header("Performance Heatmaps")
        
        png_files, html_files = get_available_plots()
        
        if not png_files:
            st.warning("No plots found. Please run the plotting script first: `python plot_qsar_results.py`")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'r2_heatmap.png' in png_files:
                st.subheader("R² Performance Heatmap")
                st.image("analyses/standardized_qsar_models/plots/r2_heatmap.png", 
                        caption="R² scores across proteins and organisms")
        
        with col2:
            if 'accuracy_heatmap.png' in png_files:
                st.subheader("Accuracy Performance Heatmap")
                st.image("analyses/standardized_qsar_models/plots/accuracy_heatmap.png", 
                        caption="Accuracy scores across proteins and organisms")
    
    elif current_page == "plot_top_performers":
        st.header("Top Performing Models")
        
        png_files, html_files = get_available_plots()
        
        if not png_files:
            st.warning("No plots found. Please run the plotting script first: `python plot_qsar_results.py`")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'top_regression_models.png' in png_files:
                st.subheader("Top 15 Regression Models")
                st.image("analyses/standardized_qsar_models/plots/top_regression_models.png", 
                        caption="Best performing regression models by R² score")
        
        with col2:
            if 'top_classification_models.png' in png_files:
                st.subheader("Top 15 Classification Models")
                st.image("analyses/standardized_qsar_models/plots/top_classification_models.png", 
                        caption="Best performing classification models by accuracy")
    
    elif current_page == "plot_interactive":
        st.header("Interactive Plots")
        
        png_files, html_files = get_available_plots()
        
        if not html_files:
            st.warning("No interactive plots found. Please run the plotting script first: `python plot_qsar_results.py`")
            return
        
        st.info("Interactive plots are HTML files that can be opened in a web browser for full interactivity.")
        
        # Display interactive plots
        for filename, description in html_files.items():
            st.subheader(description)
            
            # Read and display the HTML content
            html_path = f"analyses/standardized_qsar_models/plots/{filename}"
            if os.path.exists(html_path):
                with open(html_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                # Display the interactive plot
                st.components.v1.html(html_content, height=600, scrolling=True)
                
                # Download button
                with open(html_path, 'rb') as f:
                    st.download_button(
                        label=f"Download {filename}",
                        data=f.read(),
                        file_name=filename,
                        mime="text/html"
                    )
            else:
                st.error(f"File not found: {html_path}")
        
        # Also show static plots for reference
        st.subheader("Static Plot Reference")
        st.info("For high-resolution static versions, see the other visualization sections.")

    elif current_page == "failure_summary":
        create_failure_summary_page()
    
    elif current_page == "failure_causes":
        create_failure_causes_page()
    
    elif current_page == "presentation_discussion":
        create_presentation_discussion()

    # Footer
    st.markdown("---")
    st.markdown(f"**Dashboard Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.markdown("**Data Source:** `analyses/standardized_qsar_models/`")

if __name__ == "__main__":
    main()