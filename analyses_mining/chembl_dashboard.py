#! /usr/bin/env python3
"""
ChEMBL-mapped Analysis Dashboard

This dashboard provides analysis and visualization tools for ChEMBL-mapped bioactivity data.

Usage:
    streamlit run analyses_mining/chembl_dashboard.py

Features:
    - ChEMBL-mapped protein list exploration
    - Bioactivity points per protein visualization
    - Bioactivity type distribution analysis
    - Interactive exploration of ChEMBL-mapped bioactivity data

Requirements:
    - Python 3.8+
    - Streamlit
    - pandas, matplotlib, seaborn, numpy
"""

# --- Setup ---
import os
import sys
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add the parent directory to Python path to import functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions.data_loading import (
    load_chembl_mapped_bioactivity_profile,
)
from functions.plotting import (
    plot_protein_counts_bar,
    plot_bioactivity_type_pie_hist,
    plot_categorical_pie_bar,
    plot_numeric_histogram,
)

os.environ["STREAMLIT_SERVER_PORT"] = "8502"

st.title("ChEMBL-mapped Analysis Dashboard")

# --- Multipage structure ---
PAGES = {
    "Introduction": "intro",
    "ChEMBL-mapped Protein List": "chembl_list",
    "ChEMBL-mapped Bioactivity Points per Protein": "chembl_bioactivity_hist",
    "ChEMBL-mapped Bioactivity Type Distribution": "chembl_bioactivity_type",
    "Explore ChEMBL-mapped Bioactivity Data": "chembl_explore_bioactivity",
}

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", list(PAGES.keys()))

# --- Data loading helpers ---
@st.cache_data
def load_chembl_mapped_protein_list():
    """Load ChEMBL-mapped protein list data"""
    path = os.path.join("analyses_mining", "unique_proteins_mapped_with_chembl.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    st.error(f"File not found: {path}")
    return None

# --- Page logic ---
if PAGES[page] == "intro":
    st.header("ChEMBL-mapped Analysis Dashboard")
    st.markdown("""
    ## Welcome to the ChEMBL-mapped Analysis Dashboard
    
    This dashboard provides comprehensive analysis and visualization tools for ChEMBL-mapped bioactivity data.
    
    ### Available Pages:
    
    **1. ChEMBL-mapped Protein List**
    - Explore the complete list of proteins that have been successfully mapped to ChEMBL
    - View protein details and mapping statistics
    
    **2. ChEMBL-mapped Bioactivity Points per Protein**
    - Visualize the distribution of bioactivity data points across proteins
    - Identify proteins with the most comprehensive bioactivity data
    
    **3. ChEMBL-mapped Bioactivity Type Distribution**
    - Analyze the distribution of different bioactivity types
    - Explore specific bioactivity types in detail
    
    **4. Explore ChEMBL-mapped Bioactivity Data**
    - Interactive exploration of the complete bioactivity dataset
    - Custom visualizations for any column in the dataset
    
    ### Data Sources:
    - **Protein Mapping**: `analyses_mining/unique_proteins_mapped_with_chembl.csv`
    - **Bioactivity Data**: ChEMBL-mapped bioactivity profiles
    
    ### Navigation:
    Use the sidebar to navigate between different analysis pages.
    """)

elif PAGES[page] == "chembl_list":
    st.header("ChEMBL-mapped Protein List")
    df = load_chembl_mapped_protein_list()
    if df is not None:
        # Display basic information
        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Proteins", len(df))
        with col2:
            st.metric("Total Rows", df.shape[0])
        with col3:
            st.metric("Columns", df.shape[1])
        
        # Show the data
        st.subheader("Protein Data")
        st.dataframe(df)
        
        # Summary statistics
        st.subheader("Summary Statistics")
        st.write(df.describe(include='all'))
        
        # Column information
        st.subheader("Column Information")
        for col in df.columns:
            st.write(f"**{col}**: {df[col].dtype} - {df[col].nunique()} unique values")
            
        # Download option
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download ChEMBL-mapped Protein List",
            data=csv,
            file_name="chembl_mapped_proteins.csv",
            mime="text/csv"
        )
    else:
        st.warning("ChEMBL-mapped protein list data not found.")

elif PAGES[page] == "chembl_bioactivity_hist":
    st.header("ChEMBL-mapped: Bioactivity Points per Protein")
    df = load_chembl_mapped_bioactivity_profile()
    if df is not None:
        # Display basic information
        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Bioactivity Points", len(df))
        with col2:
            st.metric("Unique Proteins", df["Protein Name"].nunique())
        with col3:
            st.metric("Average Points per Protein", f"{len(df) / df['Protein Name'].nunique():.1f}")
        
        # Protein counts visualization
        protein_counts = df["Protein Name"].value_counts()
        
        # Top proteins
        st.subheader("Top Proteins by Bioactivity Points")
        top_n = st.slider("Number of top proteins to show", 10, 50, 30)
        plot_protein_counts_bar(protein_counts, top_n=top_n)
        
        # Distribution statistics
        st.subheader("Distribution Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Top 10 Proteins:**")
            for i, (protein, count) in enumerate(protein_counts.head(10).items(), 1):
                st.write(f"{i}. {protein}: {count} points")
        
        with col2:
            st.write("**Distribution Summary:**")
            st.write(f"Mean points per protein: {protein_counts.mean():.1f}")
            st.write(f"Median points per protein: {protein_counts.median():.1f}")
            st.write(f"Standard deviation: {protein_counts.std():.1f}")
            st.write(f"Min points: {protein_counts.min()}")
            st.write(f"Max points: {protein_counts.max()}")
        
        # Download option
        protein_summary = protein_counts.reset_index()
        protein_summary.columns = ['Protein Name', 'Bioactivity Points']
        csv = protein_summary.to_csv(index=False)
        st.download_button(
            label="Download Protein Bioactivity Summary",
            data=csv,
            file_name="chembl_protein_bioactivity_summary.csv",
            mime="text/csv"
        )
    else:
        st.warning("No ChEMBL-mapped bioactivity profile data found. Please run the corresponding analysis script.")

elif PAGES[page] == "chembl_bioactivity_type":
    st.header("ChEMBL-mapped: Bioactivity Type Distribution")
    df = load_chembl_mapped_bioactivity_profile()
    if df is not None:
        # Display basic information
        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Bioactivity Points", len(df))
        with col2:
            st.metric("Unique Bioactivity Types", df["Bioactivity Type"].nunique())
        with col3:
            st.metric("Unique Proteins", df["Protein Name"].nunique())
        
        # Bioactivity type analysis
        bioactivity_counts = df["Bioactivity Type"].value_counts()
        
        # Type selection
        st.subheader("Bioactivity Type Analysis")
        selected_group = st.selectbox("Select Bioactivity Type", bioactivity_counts.index, key="chembl_bioactivity_type_select")
        
        # Calculate percentage
        percent = bioactivity_counts[selected_group] / bioactivity_counts.sum() * 100
        
        # Filter data for selected type
        df_hist_numeric = df[(df["Bioactivity Type"] == selected_group) & pd.to_numeric(df["Value"], errors='coerce').notnull()]
        
        # Visualization options
        log_scale = st.checkbox("Log scale", value=False, key="chembl_bioactivity_type_log")
        
        # Create visualization
        plot_bioactivity_type_pie_hist(bioactivity_counts, percent, selected_group, df_hist_numeric, log_scale)
        
        # Detailed statistics
        st.subheader("Detailed Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Selected Type: {selected_group}**")
            st.write(f"Count: {bioactivity_counts[selected_group]}")
            st.write(f"Percentage: {percent:.1f}%")
            st.write(f"Numeric values: {len(df_hist_numeric)}")
        
        with col2:
            st.write("**All Bioactivity Types:**")
            for bio_type, count in bioactivity_counts.items():
                pct = (count / bioactivity_counts.sum()) * 100
                st.write(f"- {bio_type}: {count} ({pct:.1f}%)")
        
        # Download option
        type_summary = bioactivity_counts.reset_index()
        type_summary.columns = ['Bioactivity Type', 'Count']
        type_summary['Percentage'] = (type_summary['Count'] / type_summary['Count'].sum()) * 100
        csv = type_summary.to_csv(index=False)
        st.download_button(
            label="Download Bioactivity Type Summary",
            data=csv,
            file_name="chembl_bioactivity_type_summary.csv",
            mime="text/csv"
        )
    else:
        st.warning("No ChEMBL-mapped bioactivity profile data found.")

elif PAGES[page] == "chembl_explore_bioactivity":
    st.header("Explore ChEMBL-mapped Bioactivity Data")
    df = load_chembl_mapped_bioactivity_profile()
    if df is not None:
        # Display basic information
        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
        
        # Show the data
        st.subheader("Complete Dataset")
        st.dataframe(df)
        
        # Column information
        st.subheader("Column Information")
        st.write("Available columns:", list(df.columns))
        
        # Interactive exploration
        st.subheader("Interactive Column Exploration")
        selected_col = st.selectbox("Select column to explore", df.columns, key="chembl_explore_select")
        
        if pd.api.types.is_numeric_dtype(df[selected_col]):
            st.write(f"**Numeric Column: {selected_col}**")
            log_scale = st.checkbox("Log scale", value=False, key="chembl_explore_log")
            plot_numeric_histogram(df[selected_col].dropna(), selected_col, log_scale)
            
            # Numeric statistics
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Basic Statistics:**")
                st.write(f"Mean: {df[selected_col].mean():.3f}")
                st.write(f"Median: {df[selected_col].median():.3f}")
                st.write(f"Std: {df[selected_col].std():.3f}")
                st.write(f"Min: {df[selected_col].min()}")
                st.write(f"Max: {df[selected_col].max()}")
            
            with col2:
                st.write("**Percentiles:**")
                percentiles = [10, 25, 50, 75, 90, 95, 99]
                for p in percentiles:
                    value = df[selected_col].quantile(p/100)
                    st.write(f"{p}th percentile: {value:.3f}")
        else:
            st.write(f"**Categorical Column: {selected_col}**")
            value_counts = df[selected_col].value_counts()
            plot_categorical_pie_bar(value_counts, selected_col)
            
            # Categorical statistics
            st.write("**Value Counts:**")
            for value, count in value_counts.head(20).items():
                percentage = (count / len(df)) * 100
                st.write(f"- {value}: {count} ({percentage:.1f}%)")
        
        # Filtering options
        st.subheader("Data Filtering")
        if "Protein Name" in df.columns:
            protein_filter = st.multiselect("Filter by Protein", df["Protein Name"].unique())
            if protein_filter:
                df_filtered = df[df["Protein Name"].isin(protein_filter)]
                st.write(f"Filtered data: {len(df_filtered)} rows")
                st.dataframe(df_filtered)
        
        if "Bioactivity Type" in df.columns:
            type_filter = st.multiselect("Filter by Bioactivity Type", df["Bioactivity Type"].unique())
            if type_filter:
                df_filtered = df[df["Bioactivity Type"].isin(type_filter)]
                st.write(f"Filtered data: {len(df_filtered)} rows")
                st.dataframe(df_filtered)
        
        # Download options
        st.subheader("Download Data")
        col1, col2 = st.columns(2)
        
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Complete Dataset",
                data=csv,
                file_name="chembl_bioactivity_complete.csv",
                mime="text/csv"
            )
        
        with col2:
            if 'df_filtered' in locals():
                csv_filtered = df_filtered.to_csv(index=False)
                st.download_button(
                    label="Download Filtered Dataset",
                    data=csv_filtered,
                    file_name="chembl_bioactivity_filtered.csv",
                    mime="text/csv"
                )
    else:
        st.warning("No ChEMBL-mapped bioactivity profile data found.") 