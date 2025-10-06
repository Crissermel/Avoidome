#!/usr/bin/env python3
"""
Comparison between AQSE Workflow and Standard QSAR Models

This script compares the results from:
1. AQSE workflow (04_2_qsar_model_creation.py) - both standard and similar proteins models
2. Standard QSAR models (qsar_papyrus_esm_emb) - all proteins

Author: AQSE Pipeline
Date: 2025
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any

def load_aqse_results(results_dir: str) -> pd.DataFrame:
    """Load AQSE workflow results"""
    results_dir = Path(results_dir)
    aqse_results = []
    
    # Load workflow summary
    summary_file = results_dir / "workflow_summary.json"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        print(f"AQSE Workflow Summary:")
        print(f"  Models created: {summary['models_created']}")
        print(f"  Proteins skipped: {summary['proteins_skipped']}")
        print(f"  Total proteins: {summary['total_proteins']}")
        print(f"  Morgan fingerprints: {summary['step_0_results']['valid_fingerprints']}")
        print()
    
    # Load individual model results
    for metrics_file in results_dir.glob("*_metrics.json"):
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        # Determine model type based on filename
        is_similar_model = "_similar" in metrics_file.stem
        
        if is_similar_model:
            aqse_results.append({
                'protein_name': metrics['target_name'],
                'uniprot_id': metrics['uniprot_id'],
                'method': 'AQSE_Similar',
                'r2': metrics.get('test_r2', np.nan),
                'q2': metrics.get('q2', np.nan),
                'rmse': metrics.get('test_rmse', np.nan),
                'mae': metrics.get('test_mae', np.nan),
                'n_train': metrics.get('n_train', np.nan),
                'n_test': metrics.get('n_test', np.nan),
                'n_features': metrics.get('n_features', np.nan),
                'similar_proteins': metrics.get('similar_proteins', []),
                'total_samples': metrics.get('n_train', 0) + metrics.get('n_test', 0)
            })
        else:
            aqse_results.append({
                'protein_name': metrics['target_name'],
                'uniprot_id': metrics['uniprot_id'],
                'method': 'AQSE_Standard',
                'r2': metrics.get('r2', np.nan),
                'q2': metrics.get('q2', np.nan),
                'rmse': metrics.get('rmse', np.nan),
                'mae': metrics.get('mae', np.nan),
                'n_train': np.nan,  # Not applicable for CV
                'n_test': np.nan,   # Not applicable for CV
                'n_features': metrics.get('n_features', np.nan),
                'similar_proteins': [],
                'total_samples': metrics.get('n_samples', 0)
            })
    
    return pd.DataFrame(aqse_results)

def load_standard_qsar_results(csv_file: str, mapping_file: str) -> pd.DataFrame:
    """Load standard QSAR results"""
    df = pd.read_csv(csv_file)
    
    # Load protein name to UniProt ID mapping
    mapping_df = pd.read_csv(mapping_file)
    protein_to_uniprot = dict(zip(mapping_df['protein_name'], mapping_df['human_uniprot_id']))
    
    # Filter for successful models only
    successful_models = df[df['status'] == 'success'].copy()
    
    # Convert to standard format
    standard_results = []
    for _, row in successful_models.iterrows():
        protein_name = row['protein']
        uniprot_id = protein_to_uniprot.get(protein_name, protein_name)  # Use mapping or fallback to protein name
        
        standard_results.append({
            'protein_name': protein_name,
            'uniprot_id': uniprot_id,
            'method': 'Standard_QSAR',
            'r2': row['avg_r2'],
            'q2': row['avg_r2'],  # Using avg_r2 as Q² approximation
            'rmse': row['avg_rmse'],
            'mae': row['avg_mae'],
            'n_train': np.nan,  # Not available in standard results
            'n_test': np.nan,   # Not available in standard results
            'n_features': np.nan,  # Not available in standard results
            'similar_proteins': [],
            'total_samples': row['n_samples']
        })
    
    return pd.DataFrame(standard_results)

def compare_results(aqse_df: pd.DataFrame, standard_df: pd.DataFrame) -> Dict[str, Any]:
    """Compare results between AQSE and Standard QSAR"""
    
    print("="*80)
    print("COMPARISON: AQSE vs Standard QSAR Models")
    print("="*80)
    
    # Basic statistics
    print(f"\n1. MODEL COUNTS:")
    print(f"   AQSE Standard models: {len(aqse_df[aqse_df['method'] == 'AQSE_Standard'])}")
    print(f"   AQSE Similar models: {len(aqse_df[aqse_df['method'] == 'AQSE_Similar'])}")
    print(f"   Total AQSE models: {len(aqse_df)}")
    print(f"   Standard QSAR models: {len(standard_df)}")
    
    # Find common proteins
    aqse_proteins = set(aqse_df['uniprot_id'])
    standard_proteins = set(standard_df['uniprot_id'])
    common_proteins = aqse_proteins & standard_proteins
    aqse_only = aqse_proteins - standard_proteins
    standard_only = standard_proteins - aqse_proteins
    
    print(f"\n2. PROTEIN OVERLAP:")
    print(f"   Common proteins: {len(common_proteins)}")
    print(f"   AQSE only: {len(aqse_only)}")
    print(f"   Standard only: {len(standard_only)}")
    
    if common_proteins:
        print(f"   Common proteins: {sorted(list(common_proteins))}")
    
    # Performance comparison for common proteins
    if common_proteins:
        print(f"\n3. PERFORMANCE COMPARISON (Common Proteins):")
        
        aqse_common = aqse_df[aqse_df['uniprot_id'].isin(common_proteins)].copy()
        standard_common = standard_df[standard_df['uniprot_id'].isin(common_proteins)].copy()
        
        # Merge on uniprot_id for direct comparison
        comparison_df = aqse_common.merge(
            standard_common[['uniprot_id', 'r2', 'rmse', 'mae', 'q2', 'total_samples']], 
            on='uniprot_id', 
            suffixes=('_aqse', '_standard')
        )
        
        # Rename method column to avoid confusion
        comparison_df = comparison_df.rename(columns={'method': 'method_aqse'})
        
        print(f"\n   Detailed Comparison:")
        print(f"   {'Protein':<15} {'Method':<12} {'AQSE_R2':<10} {'Std_R2':<10} {'AQSE_Q2':<10} {'Std_Q2':<10} {'AQSE_RMSE':<12} {'Std_RMSE':<12}")
        print(f"   {'-'*15} {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*12} {'-'*12}")
        
        for _, row in comparison_df.iterrows():
            method_type = "Similar" if row['method_aqse'] == 'AQSE_Similar' else "Standard"
            print(f"   {row['protein_name']:<15} {method_type:<12} {row['r2_aqse']:<10.3f} {row['r2_standard']:<10.3f} "
                  f"{row['q2_aqse']:<10.3f} {row['q2_standard']:<10.3f} {row['rmse_aqse']:<12.3f} {row['rmse_standard']:<12.3f}")
        
        # Statistical comparison
        print(f"\n   Statistical Summary:")
        print(f"   Metric                AQSE Mean    Standard Mean  Difference")
        print(f"   {'-'*20} {'-'*12} {'-'*12} {'-'*12}")
        
        r2_diff = comparison_df['r2_aqse'].mean() - comparison_df['r2_standard'].mean()
        q2_diff = comparison_df['q2_aqse'].mean() - comparison_df['q2_standard'].mean()
        rmse_diff = comparison_df['rmse_aqse'].mean() - comparison_df['rmse_standard'].mean()
        
        print(f"   R²                   {comparison_df['r2_aqse'].mean():<12.3f} {comparison_df['r2_standard'].mean():<12.3f} {r2_diff:<12.3f}")
        print(f"   Q²                   {comparison_df['q2_aqse'].mean():<12.3f} {comparison_df['q2_standard'].mean():<12.3f} {q2_diff:<12.3f}")
        print(f"   RMSE                 {comparison_df['rmse_aqse'].mean():<12.3f} {comparison_df['rmse_standard'].mean():<12.3f} {rmse_diff:<12.3f}")
        
        # Sample size comparison
        print(f"\n   Sample Size Comparison:")
        print(f"   AQSE total samples: {aqse_common['total_samples'].sum()}")
        print(f"   Standard total samples: {standard_common['total_samples'].sum()}")
        print(f"   Sample size difference: {aqse_common['total_samples'].sum() - standard_common['total_samples'].sum()}")
        
        return {
            'comparison_df': comparison_df,
            'aqse_common': aqse_common,
            'standard_common': standard_common,
            'common_proteins': common_proteins,
            'aqse_only': aqse_only,
            'standard_only': standard_only
        }
    
    return {}

def analyze_differences(comparison_results: Dict[str, Any]) -> None:
    """Analyze why there are differences between the approaches"""
    
    if not comparison_results:
        print("\n4. DIFFERENCE ANALYSIS:")
        print("   No common proteins to compare - approaches are completely different")
        return
    
    comparison_df = comparison_results['comparison_df']
    
    print(f"\n4. DIFFERENCE ANALYSIS:")
    print(f"   Why are the results different?")
    
    # Check for systematic differences
    r2_correlation = comparison_df['r2_aqse'].corr(comparison_df['r2_standard'])
    q2_correlation = comparison_df['q2_aqse'].corr(comparison_df['q2_standard'])
    
    print(f"\n   a) Correlation Analysis:")
    print(f"      R² correlation: {r2_correlation:.3f}")
    print(f"      Q² correlation: {q2_correlation:.3f}")
    
    # Identify which approach performs better
    aqse_better_r2 = (comparison_df['r2_aqse'] > comparison_df['r2_standard']).sum()
    standard_better_r2 = (comparison_df['r2_standard'] > comparison_df['r2_aqse']).sum()
    
    print(f"\n   b) Performance Comparison:")
    print(f"      AQSE better R²: {aqse_better_r2}/{len(comparison_df)} proteins")
    print(f"      Standard better R²: {standard_better_r2}/{len(comparison_df)} proteins")
    
    # Analyze by model type
    print(f"\n   c) Analysis by Model Type:")
    standard_models = comparison_df[comparison_df['method_aqse'] == 'AQSE_Standard']
    similar_models = comparison_df[comparison_df['method_aqse'] == 'AQSE_Similar']
    
    if len(standard_models) > 0:
        print(f"      Standard AQSE models ({len(standard_models)}):")
        print(f"        Mean R²: {standard_models['r2_aqse'].mean():.3f} vs {standard_models['r2_standard'].mean():.3f}")
        print(f"        Mean Q²: {standard_models['q2_aqse'].mean():.3f} vs {standard_models['q2_standard'].mean():.3f}")
    
    if len(similar_models) > 0:
        print(f"      Similar proteins AQSE models ({len(similar_models)}):")
        print(f"        Mean R²: {similar_models['r2_aqse'].mean():.3f} vs {similar_models['r2_standard'].mean():.3f}")
        print(f"        Mean Q²: {similar_models['q2_aqse'].mean():.3f} vs {similar_models['q2_standard'].mean():.3f}")
    
    # Analyze potential reasons
    print(f"\n   d) Potential Reasons for Differences:")
    
    # Check sample sizes
    sample_size_diff = comparison_df['total_samples_aqse'] - comparison_df['total_samples_standard']
    print(f"      1. Sample Size Differences:")
    print(f"         Average difference (AQSE - Standard): {sample_size_diff.mean():.1f} samples")
    print(f"         Range: {sample_size_diff.min():.1f} to {sample_size_diff.max():.1f} samples")
    
    # Check feature differences
    print(f"      2. Feature Differences:")
    print(f"         AQSE uses: Morgan fingerprints (2048) + Physicochemical (14) + ESM C (1280) = 3342 features")
    print(f"         Standard QSAR: Unknown feature count (not available in results)")
    
    # Check data filtering differences
    print(f"      3. Data Filtering Differences:")
    print(f"         AQSE Standard: Only proteins with NO similar proteins (filtered by similarity search)")
    print(f"         AQSE Similar: Proteins with similar proteins (train on similar, test on target)")
    print(f"         Standard: All proteins regardless of similarity")
    
    # Check model differences
    print(f"      4. Model Differences:")
    print(f"         AQSE Standard: Random Forest (100 trees) with 5-fold CV")
    print(f"         AQSE Similar: Random Forest (100 trees) with train/test split")
    print(f"         Standard: 5-fold cross-validation (exact method unknown)")
    
    # Check data source differences
    print(f"      5. Data Source Differences:")
    print(f"         AQSE: Direct from Papyrus database (latest, plusplus=True)")
    print(f"         Standard: Pre-processed data (exact source unknown)")

def create_visualizations(comparison_results: Dict[str, Any], output_dir: str) -> None:
    """Create comparison visualizations"""
    
    if not comparison_results:
        return
    
    comparison_df = comparison_results['comparison_df']
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # R² comparison
    axes[0, 0].scatter(comparison_df['r2_standard'], comparison_df['r2_aqse'], alpha=0.7)
    axes[0, 0].plot([0, 1], [0, 1], 'r--', alpha=0.5)
    axes[0, 0].set_xlabel('Standard QSAR R²')
    axes[0, 0].set_ylabel('AQSE R²')
    axes[0, 0].set_title('R² Comparison')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Q² comparison
    axes[0, 1].scatter(comparison_df['q2_standard'], comparison_df['q2_aqse'], alpha=0.7)
    axes[0, 1].plot([0, 1], [0, 1], 'r--', alpha=0.5)
    axes[0, 1].set_xlabel('Standard QSAR Q²')
    axes[0, 1].set_ylabel('AQSE Q²')
    axes[0, 1].set_title('Q² Comparison')
    axes[0, 1].grid(True, alpha=0.3)
    
    # RMSE comparison
    axes[1, 0].scatter(comparison_df['rmse_standard'], comparison_df['rmse_aqse'], alpha=0.7)
    axes[1, 0].plot([0, 2], [0, 2], 'r--', alpha=0.5)
    axes[1, 0].set_xlabel('Standard QSAR RMSE')
    axes[1, 0].set_ylabel('AQSE RMSE')
    axes[1, 0].set_title('RMSE Comparison')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Sample size comparison
    axes[1, 1].scatter(comparison_df['total_samples_standard'], comparison_df['total_samples_aqse'], alpha=0.7)
    axes[1, 1].plot([0, 5000], [0, 5000], 'r--', alpha=0.5)
    axes[1, 1].set_xlabel('Standard QSAR Sample Size')
    axes[1, 1].set_ylabel('AQSE Sample Size')
    axes[1, 1].set_title('Sample Size Comparison')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'aqse_vs_standard_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n5. VISUALIZATIONS:")
    print(f"   Comparison plots saved to: {output_dir / 'aqse_vs_standard_comparison.png'}")

def save_detailed_results(comparison_results: Dict[str, Any], output_dir: str) -> None:
    """Save detailed comparison results"""
    
    if not comparison_results:
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save detailed comparison
    comparison_df = comparison_results['comparison_df']
    comparison_df.to_csv(output_dir / 'detailed_comparison.csv', index=False)
    
    print(f"\n6. DETAILED RESULTS:")
    print(f"   Detailed comparison saved to: {output_dir / 'detailed_comparison.csv'}")

def main():
    """Main comparison function"""
    
    # File paths
    aqse_results_dir = "/home/serramelendezcsm/RA/Avoidome/analyses/avoidome_qsar_similarity_expansion/04_qsar_models_temp/results"
    standard_qsar_file = "/home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_esm_emb/esm_prediction_results.csv"
    mapping_file = "/home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_esm_emb/data_overview_results.csv"
    output_dir = "/home/serramelendezcsm/RA/Avoidome/analyses/avoidome_qsar_similarity_expansion/comparison_results"
    
    print("Loading AQSE workflow results...")
    aqse_df = load_aqse_results(aqse_results_dir)
    
    print("Loading standard QSAR results...")
    standard_df = load_standard_qsar_results(standard_qsar_file, mapping_file)
    
    # Compare results
    comparison_results = compare_results(aqse_df, standard_df)
    
    # Analyze differences
    analyze_differences(comparison_results)
    
    # Create visualizations
    create_visualizations(comparison_results, output_dir)
    
    # Save detailed results
    save_detailed_results(comparison_results, output_dir)
    
    print(f"\n{'='*80}")
    print("COMPARISON COMPLETED")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()