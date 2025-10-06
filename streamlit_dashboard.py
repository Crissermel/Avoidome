#! /usr/bin/env python3
"""
Avoidome Streamlit Dashboard

Usage:
    Run this dashboard with:
        streamlit run streamlit_dashboard.py

Features:
    - Introduction and project overview
    - Browse Avoidome protein list
    - Visualize bioactivity points per protein (histogram)
    - Explore bioactivity type distribution (pie/bar)
    - Interactive exploration of bioactivity data
    - Visualize Avoidome protein-protein interaction network (STRINGdb)
    - QSAR Model Performance & Architecture

Requirements:
    - Python 3.8+
    - Streamlit
    - pandas, matplotlib, seaborn, networkx, pyvis, etc.

Data files:
    - Place required data files in the appropriate folders (see README or code comments).

For questions or issues, see the project README or contact the maintainers.
"""

# --- Setup ---
import os
import sys
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add the current directory to Python path to import functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from functions.data_loading import (
    load_avoidome_prot_list,
    load_bioactivity_profile,
)
from functions.plotting import (
    plot_protein_counts_bar,
    plot_bioactivity_type_pie_hist,
    plot_categorical_pie_bar,
    plot_numeric_histogram,
    plot_interactive_networkx_pyvis,
)
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

os.environ["STREAMLIT_SERVER_PORT"] = "8501"

st.title("Avoidome Dashboard")

# --- Multipage structure ---
PAGES = {
    "Introduction": "intro",
    "Avoidome Protein List": "avoidome_list",
    "Bioactivity Points per Protein": "bioactivity_hist",
    "Bioactivity Type Distribution": "bioactivity_type",
    "Explore Bioactivity Data": "explore_bioactivity",
    "Avoidome Network (STRINGdb)": "network",
    # QSAR pages:
    "QSAR Model Performance": "qsar_performance",
    "QSAR Model Architecture": "qsar_architecture",
    "QSAR Model Details": "qsar_details",
    "QSAR Predictions Comparison": "qsar_predictions",
}

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", list(PAGES.keys()))

# --- Data loading helpers ---

@st.cache_data
def load_qsar_performance_data():
    """Load QSAR model performance data"""
    path = "analyses/qsar_avoidome/model_performance_summary.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        # Convert to numeric where possible
        numeric_cols = ['R2', 'RMSE', 'MAE', 'CV_R2_Mean', 'CV_R2_Std', 'Training_Time']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    st.error(f"QSAR performance file not found: {path}")
    return None

@st.cache_data
def load_qsar_predictions_data():
    """Load QSAR predictions comparison data"""
    reports_dir = "analyses/qsar_avoidome/prediction_reports"
    
    # Load summary data
    summary_file = os.path.join(reports_dir, 'actual_vs_predicted_summary.csv')
    summary_df = None
    if os.path.exists(summary_file):
        summary_df = pd.read_csv(summary_file)
    
    # Load detailed data
    detailed_file = os.path.join(reports_dir, 'actual_vs_predicted_detailed.csv')
    detailed_df = None
    if os.path.exists(detailed_file):
        detailed_df = pd.read_csv(detailed_file)
    
    # Load error analysis
    error_file = os.path.join(reports_dir, 'prediction_errors_analysis.csv')
    error_df = None
    if os.path.exists(error_file):
        error_df = pd.read_csv(error_file)
    
    return summary_df, detailed_df, error_df

# --- QSAR Helper Functions ---
def plot_qsar_performance_metrics(df):
    """Plot QSAR performance metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # R² Distribution
    axes[0, 0].hist(df['R2'].dropna(), bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(df['R2'].mean(), color='red', linestyle='--', label=f'Mean: {df["R2"].mean():.3f}')
    axes[0, 0].set_xlabel('R² Score')
    axes[0, 0].set_ylabel('Number of Models')
    axes[0, 0].set_title('Distribution of R² Scores')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # RMSE Distribution
    axes[0, 1].hist(df['RMSE'].dropna(), bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].axvline(df['RMSE'].mean(), color='red', linestyle='--', label=f'Mean: {df["RMSE"].mean():.3f}')
    axes[0, 1].set_xlabel('RMSE')
    axes[0, 1].set_ylabel('Number of Models')
    axes[0, 1].set_title('Distribution of RMSE')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # R² vs RMSE Scatter
    axes[1, 0].scatter(df['RMSE'], df['R2'], alpha=0.6, color='purple')
    axes[1, 0].set_xlabel('RMSE')
    axes[1, 0].set_ylabel('R² Score')
    axes[1, 0].set_title('R² vs RMSE Relationship')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Training Time Distribution
    axes[1, 1].hist(df['Training_Time'].dropna(), bins=15, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 1].axvline(df['Training_Time'].mean(), color='red', linestyle='--', label=f'Mean: {df["Training_Time"].mean():.1f}s')
    axes[1, 1].set_xlabel('Training Time (seconds)')
    axes[1, 1].set_ylabel('Number of Models')
    axes[1, 1].set_title('Distribution of Training Times')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_top_performing_models(df, n=10):
    """Plot top performing models"""
    top_models = df.nlargest(n, 'R2')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Top R² scores
    bars1 = ax1.barh(range(len(top_models)), top_models['R2'], color='skyblue')
    ax1.set_yticks(range(len(top_models)))
    ax1.set_yticklabels(top_models['Target_ID'])
    ax1.set_xlabel('R² Score')
    ax1.set_title(f'Top {n} Models by R² Score')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars1, top_models['R2'])):
        ax1.text(value + 0.01, i, f'{value:.3f}', va='center')
    
    # Top RMSE scores (lower is better)
    top_rmse = df.nsmallest(n, 'RMSE')
    bars2 = ax2.barh(range(len(top_rmse)), top_rmse['RMSE'], color='lightgreen')
    ax2.set_yticks(range(len(top_rmse)))
    ax2.set_yticklabels(top_rmse['Target_ID'])
    ax2.set_xlabel('RMSE')
    ax2.set_title(f'Top {n} Models by RMSE (Lower is Better)')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars2, top_rmse['RMSE'])):
        ax2.text(value + 0.01, i, f'{value:.3f}', va='center')
    
    plt.tight_layout()
    return fig

def plot_actual_vs_predicted(detailed_df, target_id=None):
    """Plot actual vs predicted values"""
    if target_id:
        data = detailed_df[detailed_df['target_id'] == target_id]
        title = f'Actual vs Predicted pChEMBL Values - {target_id}'
    else:
        data = detailed_df
        title = 'Actual vs Predicted pChEMBL Values - All Targets'
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter plot
    ax1.scatter(data['actual_pchembl'], data['predicted_pchembl'], alpha=0.6, color='blue')
    
    # Perfect prediction line
    min_val = min(data['actual_pchembl'].min(), data['predicted_pchembl'].min())
    max_val = max(data['actual_pchembl'].max(), data['predicted_pchembl'].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
    
    ax1.set_xlabel('Actual pChEMBL')
    ax1.set_ylabel('Predicted pChEMBL')
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Error distribution
    ax2.hist(data['absolute_error'], bins=20, alpha=0.7, color='orange', edgecolor='black')
    ax2.axvline(data['absolute_error'].mean(), color='red', linestyle='--', 
                label=f'Mean: {data["absolute_error"].mean():.3f}')
    ax2.set_xlabel('Absolute Error')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Error Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_error_analysis(detailed_df, target_id=None):
    """Plot error analysis"""
    if target_id:
        data = detailed_df[detailed_df['target_id'] == target_id]
        title = f'Error Analysis - {target_id}'
    else:
        data = detailed_df
        title = 'Error Analysis - All Targets'
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Error vs Actual
    ax1.scatter(data['actual_pchembl'], data['absolute_error'], alpha=0.6, color='purple')
    ax1.set_xlabel('Actual pChEMBL')
    ax1.set_ylabel('Absolute Error')
    ax1.set_title('Error vs Actual Value')
    ax1.grid(True, alpha=0.3)
    
    # Percentage Error Distribution
    ax2.hist(data['percentage_error'], bins=20, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(data['percentage_error'].mean(), color='red', linestyle='--',
                label=f'Mean: {data["percentage_error"].mean():.1f}%')
    ax2.set_xlabel('Percentage Error (%)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Percentage Error Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Error Category Distribution
    error_counts = data['error_category'].value_counts()
    colors = ['green', 'lightgreen', 'orange', 'red']
    ax3.pie(error_counts.values, labels=error_counts.index, autopct='%1.1f%%', colors=colors)
    ax3.set_title('Error Category Distribution')
    
    # Error vs Target
    if target_id is None:
        target_errors = data.groupby('target_id')['absolute_error'].mean().sort_values(ascending=False)
        ax4.bar(range(len(target_errors)), target_errors.values, color='skyblue')
        ax4.set_xlabel('Target ID')
        ax4.set_ylabel('Mean Absolute Error')
        ax4.set_title('Mean Error by Target')
        ax4.set_xticks(range(len(target_errors)))
        ax4.set_xticklabels(target_errors.index, rotation=45)
        ax4.grid(True, alpha=0.3)
    else:
        # For single target, show error over time/compounds
        ax4.plot(range(len(data)), data['absolute_error'].sort_values(ascending=False), 'o-', color='red')
        ax4.set_xlabel('Compound Index (sorted by error)')
        ax4.set_ylabel('Absolute Error')
        ax4.set_title('Error Distribution (sorted)')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# --- Page logic ---
if PAGES[page] == "intro":
    st.header("Introduction")
    if os.path.exists("Introduction.md"):
        with open("Introduction.md", "r") as f:
            st.markdown(f.read())
    else:
        st.warning("Introduction.md not found.")

elif PAGES[page] == "avoidome_list":
    st.header("Avoidome Protein List")
    df = load_avoidome_prot_list("primary_data/avoidome_prot_list.csv")
    st.dataframe(df)
    st.write("Summary Stats:")
    st.write(df.describe(include='all'))

elif PAGES[page] == "bioactivity_hist":
    st.header("Avoidome: Bioactivity Points per Protein")
    df = load_bioactivity_profile()
    if df is not None:
        protein_counts = df["Protein Name"].value_counts().head(30)
        plot_protein_counts_bar(protein_counts, top_n=30)
    else:
        st.warning("No Avoidome bioactivity profile data found. Please run the corresponding analysis script.")

elif PAGES[page] == "bioactivity_type":
    st.header("Avoidome: Bioactivity Type Distribution")
    df = load_bioactivity_profile()
    if df is not None:
        bioactivity_counts = df["Bioactivity Type"].value_counts()
        selected_group = st.selectbox("Select Bioactivity Type", bioactivity_counts.index, key="avoidome_bioactivity_type_select")
        percent = bioactivity_counts[selected_group] / bioactivity_counts.sum() * 100
        df_hist_numeric = df[(df["Bioactivity Type"] == selected_group) & pd.to_numeric(df["Value"], errors='coerce').notnull()]
        log_scale = st.checkbox("Log scale", value=False, key="avoidome_bioactivity_type_log")
        plot_bioactivity_type_pie_hist(bioactivity_counts, percent, selected_group, df_hist_numeric, log_scale)
    else:
        st.warning("No Avoidome bioactivity profile data found.")

elif PAGES[page] == "explore_bioactivity":
    st.header("Explore Avoidome Bioactivity Data")
    df = load_bioactivity_profile()
    if df is not None:
        st.dataframe(df)
        st.write("Columns:", list(df.columns))
        selected_col = st.selectbox("Select column to explore", df.columns, key="avoidome_explore_select")
        if pd.api.types.is_numeric_dtype(df[selected_col]):
            log_scale = st.checkbox("Log scale", value=False, key="avoidome_explore_log")
            plot_numeric_histogram(df[selected_col].dropna(), selected_col, log_scale)
        else:
            value_counts = df[selected_col].value_counts()
            plot_categorical_pie_bar(value_counts, selected_col)
    else:
        st.warning("No Avoidome bioactivity profile data found.")

elif PAGES[page] == "network":
    st.header("Avoidome Network (STRINGdb)")
    
    # Load interaction data
    interaction_file = "primary_data/string_interactions_short.tsv"
    if not os.path.exists(interaction_file):
        st.error(f"Interaction file not found: {interaction_file}")
    else:
        # Read the TSV file
        df = pd.read_csv(interaction_file, sep='	')
        
        # Create the graph
        import networkx as nx
        G = nx.from_pandas_edgelist(
            df,
            source='node1',
            target='node2',
            edge_attr='combined_score'
        )
        
        # Display network visualization
        plot_interactive_networkx_pyvis(G, title="Avoidome Protein-Protein Interaction Network")

elif PAGES[page] == "qsar_performance":
    st.header("QSAR Model Performance")
    
    df = load_qsar_performance_data()
    if df is not None:
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Models", len(df))
        with col2:
            st.metric("Average R²", f"{df['R2'].mean():.3f}")
        with col3:
            st.metric("Average RMSE", f"{df['RMSE'].mean():.3f}")
        with col4:
            st.metric("Best R²", f"{df['R2'].max():.3f}")
        
        # Performance plots
        st.subheader("Performance Metrics Distribution")
        fig = plot_qsar_performance_metrics(df)
        st.pyplot(fig)
        
        # Top performing models
        st.subheader("Top Performing Models")
        n_models = st.slider("Number of top models to show", 5, 20, 10)
        fig2 = plot_top_performing_models(df, n_models)
        st.pyplot(fig2)
        
        # Detailed table
        st.subheader("Detailed Model Performance")
        st.dataframe(df.sort_values('R2', ascending=False))
        
    else:
        st.warning("QSAR performance data not found. Please run the QSAR modeling script first.")

elif PAGES[page] == "qsar_architecture":
    st.header("QSAR Model Architecture & Explanation")
    
    st.markdown("""
    ## QSAR (Quantitative Structure-Activity Relationship) Models
    
    ### **Model Type: Random Forest Regressor**
    
    **Why Random Forest?**
    - **Robust**: Handles non-linear relationships well
    - **Feature Importance**: Provides insights into which molecular features matter most
    - **No Overfitting**: Ensemble method reduces overfitting
    - **No Scaling Required**: Works well with mixed feature types
    """)
    
    st.markdown("""
    ### **Model Architecture**
    
    #### **1. Input Layer: Molecular Descriptors (522 features)**
    - **Physicochemical Properties**: Molecular weight, LogP, H-bond donors/acceptors, TPSA
    - **Morgan Fingerprints**: 512-bit molecular fingerprints (ECFP-like)
    - **Structural Descriptors**: Ring counts, aromaticity, rotatable bonds
    
    #### **2. Ensemble Layer: Multiple Decision Trees**
    - **Bootstrap Sampling**: Each tree trained on random subset of data
    - **Feature Randomization**: Each split considers random subset of features
    - **Voting/Averaging**: Final prediction is average of all tree predictions
    
    #### **3. Output Layer: pChEMBL Prediction**
    - **Continuous Value**: Predicted pChEMBL activity score
    - **Confidence**: Model provides uncertainty estimates
    """)
    
    st.markdown("""
    ### **Training Process**
    
    #### **Data Preparation:**
    1. **SMILES Input**: Canonical SMILES strings for each molecule
    2. **Descriptor Calculation**: RDKit generates 522 molecular descriptors
    3. **Target Selection**: Separate model for each protein target
    4. **Data Split**: 80% training, 20% testing
    
    #### **Model Training:**
    1. **Cross-Validation**: 5-fold CV for robust evaluation
    2. **Hyperparameter Optimization**: Grid search for optimal parameters
    3. **Feature Selection**: Automatic feature importance ranking
    4. **Model Validation**: Multiple metrics (R², RMSE, MAE)
    """)
    
    st.markdown("""
    ### **Performance Metrics**
    
    - **R² (Coefficient of Determination)**: How well the model explains variance
    - **RMSE (Root Mean Square Error)**: Average prediction error
    - **MAE (Mean Absolute Error)**: Average absolute prediction error
    - **CV R²**: Cross-validation R² score (more robust)
    
    ### **Interpretation**
    
    - **R² > 0.7**: Excellent predictive performance
    - **R² 0.5-0.7**: Good predictive performance  
    - **R² 0.3-0.5**: Moderate predictive performance
    - **R² < 0.3**: Poor predictive performance
    """)
    
    st.markdown("""
    ### **Key Innovations**
    
    1. **Global Molecule Caching**: Descriptors calculated once per unique molecule
    2. **Single-Target Models**: Specialized models for each protein target
    3. **Comprehensive Descriptors**: 522 features capturing molecular properties
    4. **Robust Validation**: 5-fold cross-validation for reliable performance estimates
    """)

elif PAGES[page] == "qsar_details":
    st.header("QSAR Model Details")
    
    df = load_qsar_performance_data()
    if df is not None:
        # Model selection
        st.subheader("Select Model for Detailed Analysis")
        selected_target = st.selectbox("Choose Target Protein", df['Target_ID'].unique())
        
        selected_model = df[df['Target_ID'] == selected_target].iloc[0]
        
        # Display model details
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Target ID", selected_target)
            st.metric("Model Type", selected_model['Model'])
            st.metric("R² Score", f"{selected_model['R2']:.3f}")
            st.metric("RMSE", f"{selected_model['RMSE']:.3f}")
        
        with col2:
            st.metric("MAE", f"{selected_model['MAE']:.3f}")
            st.metric("CV R² Mean", f"{selected_model['CV_R2_Mean']:.3f}")
            st.metric("CV R² Std", f"{selected_model['CV_R2_Std']:.3f}")
            st.metric("Training Time", f"{selected_model['Training_Time']:.1f}s")
        
        # Performance interpretation
        st.subheader("Performance Interpretation")
        r2 = selected_model['R2']
        if r2 > 0.7:
            performance = "Excellent"
        elif r2 > 0.5:
            performance = "Good"
        elif r2 > 0.3:
            performance = "Moderate"
        else:
            performance = "Poor"
        
        st.info(f"**Performance Level**: {performance}")
        
        # Model file info
        model_dir = f"analyses/qsar_avoidome/{selected_target}"
        if os.path.exists(model_dir):
            st.subheader("Model Files")
            files = os.listdir(model_dir)
            for file in files:
                if file.endswith('.pkl') or file.endswith('.png'):
                    st.write(f"File: {file}")
        else:
            st.warning(f"Model directory not found: {model_dir}")
        
        # Prediction capability
        st.subheader("Prediction Capability")
        st.markdown("""
        This model can predict pChEMBL values for new molecules targeting **{target}**.
        
        **Input**: SMILES string of a molecule
        **Output**: Predicted pChEMBL activity score
        
        **Usage**: Use the `predict.py` script with this target ID.
        """.format(target=selected_target))
        
    else:
        st.warning("QSAR performance data not found. Please run the QSAR modeling script first.") 

elif PAGES[page] == "qsar_predictions":
    st.header("QSAR Predictions Comparison")
    
    # Load prediction data
    summary_df, detailed_df, error_df = load_qsar_predictions_data()
    
    if summary_df is not None and detailed_df is not None:
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Targets", len(summary_df))
        with col2:
            st.metric("Total Predictions", len(detailed_df))
        with col3:
            st.metric("Average MAE", f"{detailed_df['absolute_error'].mean():.3f}")
        with col4:
            st.metric("Average R²", f"{summary_df['r2'].mean():.3f}")
        
        # Target selection
        st.subheader("Target Selection")
        target_options = ['All Targets'] + list(detailed_df['target_id'].unique())
        selected_target = st.selectbox("Choose Target for Analysis", target_options)
        
        if selected_target == 'All Targets':
            target_filter = detailed_df
            target_id = None
        else:
            target_filter = detailed_df[detailed_df['target_id'] == selected_target]
            target_id = selected_target
        
        # Tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs(["Actual vs Predicted", "Error Analysis", "Summary Table", "Outlier Analysis"])
        
        with tab1:
            st.subheader("Actual vs Predicted Values")
            fig = plot_actual_vs_predicted(target_filter, target_id)
            st.pyplot(fig)
            
            # Statistics
            if target_id:
                r2 = r2_score(target_filter['actual_pchembl'], target_filter['predicted_pchembl'])
                mae = mean_absolute_error(target_filter['actual_pchembl'], target_filter['predicted_pchembl'])
                rmse = np.sqrt(mean_squared_error(target_filter['actual_pchembl'], target_filter['predicted_pchembl']))
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("R² Score", f"{r2:.3f}")
                with col2:
                    st.metric("MAE", f"{mae:.3f}")
                with col3:
                    st.metric("RMSE", f"{rmse:.3f}")
        
        with tab2:
            st.subheader("Error Analysis")
            fig = plot_error_analysis(target_filter, target_id)
            st.pyplot(fig)
            
            # Error statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Absolute Error", f"{target_filter['absolute_error'].mean():.3f}")
            with col2:
                st.metric("Mean Percentage Error", f"{target_filter['percentage_error'].mean():.1f}%")
            with col3:
                st.metric("Max Error", f"{target_filter['absolute_error'].max():.3f}")
        
        with tab3:
            st.subheader("Detailed Comparison Table")
            
            # Filter options
            error_threshold = st.slider("Filter by Maximum Error", 0.0, 5.0, 2.0, 0.1)
            filtered_data = target_filter[target_filter['absolute_error'] <= error_threshold]
            
            st.write(f"Showing {len(filtered_data)} predictions with error ≤ {error_threshold}")
            st.dataframe(filtered_data.sort_values('absolute_error', ascending=False))
            
            # Download option
            csv = filtered_data.to_csv(index=False)
            st.download_button(
                label="Download Filtered Data",
                data=csv,
                file_name=f"qsar_predictions_{selected_target.replace(' ', '_')}.csv",
                mime="text/csv"
            )
        
        with tab4:
            st.subheader("Outlier Analysis")
            
            if error_df is not None:
                # Show worst predictions
                worst_predictions = error_df[error_df['error_category'] == 'Outlier']
                if target_id:
                    worst_predictions = worst_predictions[worst_predictions['target_id'] == target_id]
                
                st.write("**Worst Predictions (Outliers):**")
                st.dataframe(worst_predictions)
                
                # Show best predictions
                best_predictions = error_df[error_df['error_category'] == 'Excellent']
                if target_id:
                    best_predictions = best_predictions[best_predictions['target_id'] == target_id]
                
                st.write("**Best Predictions:**")
                st.dataframe(best_predictions)
            else:
                st.warning("Error analysis data not available. Run the extraction script first.")
        
        # Overall summary
        st.subheader("Overall Summary")
        
        # Performance distribution
        performance_dist = summary_df['r2'].apply(lambda x: 
            'Excellent' if x > 0.7 else 
            'Good' if x > 0.5 else 
            'Moderate' if x > 0.3 else 'Poor'
        ).value_counts()
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Performance Distribution:**")
            for perf, count in performance_dist.items():
                percentage = (count / len(summary_df)) * 100
                st.write(f"- {perf}: {count} targets ({percentage:.1f}%)")
        
        with col2:
            st.write("**Top 5 Best Performing Targets:**")
            top_targets = summary_df.nlargest(5, 'r2')[['target_id', 'r2', 'mae']]
            for _, row in top_targets.iterrows():
                st.write(f"- {row['target_id']}: R² = {row['r2']:.3f}, MAE = {row['mae']:.3f}")
    
    else:
        st.warning("QSAR predictions data not found. Please run the extraction script first.")
        st.info("To generate prediction comparison data, run: `python analyses/qsar_avoidome/extract_predictions.py`") 

 