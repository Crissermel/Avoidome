#!/usr/bin/env python3
"""
Test script to verify dashboard data loading
"""

import pandas as pd
import os
from pathlib import Path

def test_data_loading():
    """Test if all required data files can be loaded"""
    print("Testing QSAR Dashboard Data Loading")
    print("=" * 40)
    
    # Test Morgan summary
    morgan_path = "analyses/standardized_qsar_models/modeling_summary.csv"
    if os.path.exists(morgan_path):
        morgan_df = pd.read_csv(morgan_path)
        print(f"Morgan summary loaded: {len(morgan_df)} rows")
        print(f"   - Completed models: {len(morgan_df[morgan_df['status'] == 'completed'])}")
        print(f"   - Unique proteins: {morgan_df['protein_name'].nunique()}")
        print(f"   - Organisms: {morgan_df['organism'].unique()}")
    else:
        print(f"Morgan summary not found: {morgan_path}")
    
    # Test ESM+Morgan summary
    esm_path = "analyses/standardized_qsar_models/esm_morgan_modeling_summary.csv"
    if os.path.exists(esm_path):
        esm_df = pd.read_csv(esm_path)
        print(f"ESM+Morgan summary loaded: {len(esm_df)} rows")
        print(f"   - Completed models: {len(esm_df[esm_df['status'] == 'completed'])}")
        print(f"   - Unique proteins: {esm_df['protein_name'].nunique()}")
        print(f"   - Organisms: {esm_df['organism'].unique()}")
    else:
        print(f"ESM+Morgan summary not found: {esm_path}")
    
    # Test protein list
    protein_path = "processed_data/papyrus_protein_check_results.csv"
    if os.path.exists(protein_path):
        protein_df = pd.read_csv(protein_path)
        print(f"Protein list loaded: {len(protein_df)} rows")
    else:
        print(f"Protein list not found: {protein_path}")
    
    # Test model counts
    print("\nModel Counts by Type and Organism:")
    base_path = Path("analyses/standardized_qsar_models")
    
    for model_type in ["morgan_regression", "morgan_classification", "esm_morgan_regression", "esm_morgan_classification"]:
        print(f"\n{model_type}:")
        for organism in ["human", "mouse", "rat"]:
            model_path = base_path / model_type / organism
            if model_path.exists():
                model_files = list(model_path.glob("*/model.pkl"))
                print(f"  {organism}: {len(model_files)} models")
            else:
                print(f"  {organism}: 0 models (directory not found)")
    
    print("\n" + "=" * 40)
    print("Data loading test completed!")

if __name__ == "__main__":
    test_data_loading()