#!/usr/bin/env python3
"""
Compare model results between 04_qsar_model_creation.py and standardized QSAR models
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_04_qsar_results():
    """Load results from 04_qsar_model_creation.py"""
    try:
        results_file = "/home/serramelendezcsm/RA/Avoidome/analyses/avoidome_qsar_similarity_expansion/04_qsar_models/tables/target_performance_summary.csv"
        df = pd.read_csv(results_file)
        
        # Clean up the data
        df = df.rename(columns={
            'target_name': 'protein_name',
            'val_r2_mean': 'r2_04',
            'val_rmse_mean': 'rmse_04',
            'q2_mean': 'q2_04'
        })
        
        # Add source identifier
        df['source'] = '04_qsar_model_creation'
        
        logger.info(f"Loaded 04_qsar_model_creation.py results: {len(df)} proteins")
        return df
        
    except Exception as e:
        logger.error(f"Error loading 04_qsar_model_creation.py results: {e}")
        return pd.DataFrame()

def load_model_cases():
    """Load model case information"""
    try:
        cases_file = "/home/serramelendezcsm/RA/Avoidome/analyses/avoidome_qsar_similarity_expansion/model_cases.csv"
        df = pd.read_csv(cases_file)
        logger.info(f"Loaded model cases: {len(df)} proteins")
        return df
    except Exception as e:
        logger.error(f"Error loading model cases: {e}")
        return pd.DataFrame()

def load_standardized_qsar_results():
    """Load results from standardized QSAR models"""
    try:
        results_file = "/home/serramelendezcsm/RA/Avoidome/analyses/standardized_qsar_models/esm_morgan_modeling_summary.csv"
        df = pd.read_csv(results_file)
        
        # Filter for human proteins only and completed models
        df = df[(df['organism'] == 'human') & (df['status'] == 'completed')]
        
        # Clean up the data
        df = df.rename(columns={
            'regression_r2': 'r2_standardized',
            'regression_rmse': 'rmse_standardized'
        })
        
        # Add source identifier
        df['source'] = 'standardized_qsar'
        
        logger.info(f"Loaded standardized QSAR results: {len(df)} proteins")
        return df
        
    except Exception as e:
        logger.error(f"Error loading standardized QSAR results: {e}")
        return pd.DataFrame()

def create_protein_mapping():
    """Create mapping between protein names and UniProt IDs"""
    mapping = {
        'CYP1A2': 'P05177',
        'CYP2B6': 'P20813', 
        'CYP2C9': 'P11712',
        'CYP2C19': 'P33261',
        'CYP2D6': 'P10635',
        'CYP3A4': 'P08684',
        'XDH': 'P47989',
        'MAOA': 'P21397',
        'MAOB': 'P27338',
        'ALDH1A1': 'P00352',
        'HSD11B1': 'P28845',
        'NR1I3': 'Q14994',
        'NR1I2': 'O75469',
        'HTR2B': 'P41595',
        'ADRB1': 'P08588',
        'ADRB2': 'P07550',
        'ADRA1A': 'P35348',
        'ADRA2A': 'P08913',
        'CHRM1': 'P11229',
        'CHRM2': 'P08172',
        'CHRM3': 'P20309',
        'CHRNA7': 'P36544',
        'KCNH2': 'Q12809',
        'SCN5A': 'Q14524',
        'SLC6A2': 'P23975',
        'SLC6A3': 'P23975',
        'SLC6A4': 'P31645',
        'SLCO2B1': 'O94956',
        'CNR2': 'P34972',
        'HRH1': 'P35367'
    }
    return mapping

def compare_results():
    """Compare results between both approaches"""
    logger.info("Loading results from both approaches...")
    
    # Load results
    df_04 = load_04_qsar_results()
    df_std = load_standardized_qsar_results()
    df_cases = load_model_cases()
    
    if df_04.empty or df_std.empty:
        logger.error("Failed to load results from one or both approaches")
        return
    
    # Create protein mapping
    protein_mapping = create_protein_mapping()
    
    # Add UniProt ID to 04_qsar results
    df_04['uniprot_id'] = df_04['protein_name'].map(protein_mapping)
    
    # Merge results on protein name or UniProt ID
    comparison_df = pd.merge(
        df_04[['protein_name', 'uniprot_id', 'r2_04', 'rmse_04', 'q2_04', 'n_train_mean', 'n_val_mean']],
        df_std[['protein_name', 'uniprot_id', 'r2_standardized', 'rmse_standardized', 'n_samples']],
        on='protein_name',
        how='inner'
    )
    
    # Add case information
    if not df_cases.empty:
        comparison_df = pd.merge(
            comparison_df,
            df_cases[['protein_name', 'case', 'target_samples', 'unique_proteins', 'reason']],
            on='protein_name',
            how='left'
        )
    else:
        comparison_df['case'] = 'Unknown'
        comparison_df['target_samples'] = 0
        comparison_df['unique_proteins'] = 0
        comparison_df['reason'] = 'Case data not available'
    
    logger.info(f"Found {len(comparison_df)} proteins with results from both approaches")
    
    # Calculate differences
    comparison_df['r2_diff'] = comparison_df['r2_04'] - comparison_df['r2_standardized']
    comparison_df['rmse_diff'] = comparison_df['rmse_04'] - comparison_df['rmse_standardized']
    
    # Sort by R² difference (worst performing 04_qsar first)
    comparison_df = comparison_df.sort_values('r2_diff')
    
    return comparison_df

def analyze_comparison(comparison_df):
    """Analyze the comparison results"""
    logger.info("\n" + "="*80)
    logger.info("MODEL PERFORMANCE COMPARISON RESULTS")
    logger.info("="*80)
    
    # Display detailed results
    logger.info(f"{'Protein':<12} {'Case':<6} {'Samples':<8} {'04 R²':<8} {'Std R²':<8} {'R² Diff':<8} {'Status'}")
    logger.info("-" * 80)
    
    for _, row in comparison_df.iterrows():
        status = "BETTER" if row['r2_diff'] > 0.01 else "WORSE" if row['r2_diff'] < -0.01 else "SAME"
        samples = f"{int(row['n_train_mean'])}/{int(row['n_val_mean'])}" if pd.notna(row['n_train_mean']) else f"{int(row['n_samples'])}"
        case = row.get('case', 'Unknown')
        logger.info(f"{row['protein_name']:<12} {case:<6} {samples:<8} {row['r2_04']:<8.3f} {row['r2_standardized']:<8.3f} {row['r2_diff']:<8.3f} {status}")
    
    # Summary statistics
    better_count = len(comparison_df[comparison_df['r2_diff'] > 0.01])
    worse_count = len(comparison_df[comparison_df['r2_diff'] < -0.01])
    same_count = len(comparison_df[abs(comparison_df['r2_diff']) <= 0.01])
    
    logger.info(f"\nSummary Statistics:")
    logger.info(f"  Total proteins compared: {len(comparison_df)}")
    logger.info(f"  04_qsar_model_creation.py BETTER: {better_count} proteins ({better_count/len(comparison_df)*100:.1f}%)")
    logger.info(f"  04_qsar_model_creation.py WORSE: {worse_count} proteins ({worse_count/len(comparison_df)*100:.1f}%)")
    logger.info(f"  Performance SAME: {same_count} proteins ({same_count/len(comparison_df)*100:.1f}%)")
    
    # Performance metrics
    mean_r2_04 = comparison_df['r2_04'].mean()
    mean_r2_std = comparison_df['r2_standardized'].mean()
    mean_diff = comparison_df['r2_diff'].mean()
    
    logger.info(f"\nAverage Performance:")
    logger.info(f"  04_qsar_model_creation.py average R²: {mean_r2_04:.3f}")
    logger.info(f"  Standardized QSAR average R²: {mean_r2_std:.3f}")
    logger.info(f"  Average difference: {mean_diff:.3f}")
    
    # Show worst cases
    worst_cases = comparison_df.head(10)
    logger.info(f"\nTop 10 worst performing cases (04_qsar_model_creation.py worse):")
    for _, row in worst_cases.iterrows():
        if row['r2_diff'] < -0.01:
            logger.info(f"  {row['protein_name']}: {row['r2_04']:.3f} vs {row['r2_standardized']:.3f} (diff: {row['r2_diff']:.3f})")
    
    # Show best cases
    best_cases = comparison_df.tail(10)
    logger.info(f"\nTop 10 best performing cases (04_qsar_model_creation.py better):")
    for _, row in best_cases.iterrows():
        if row['r2_diff'] > 0.01:
            logger.info(f"  {row['protein_name']}: {row['r2_04']:.3f} vs {row['r2_standardized']:.3f} (diff: {row['r2_diff']:.3f})")
    
    # Case-specific analysis
    logger.info(f"\nCase-specific Analysis:")
    for case in ['Case 1', 'Case 2', 'Case 3']:
        case_data = comparison_df[comparison_df['case'] == case]
        if not case_data.empty:
            case_better = len(case_data[case_data['r2_diff'] > 0.01])
            case_worse = len(case_data[case_data['r2_diff'] < -0.01])
            case_same = len(case_data[abs(case_data['r2_diff']) <= 0.01])
            avg_r2_04 = case_data['r2_04'].mean()
            avg_r2_std = case_data['r2_standardized'].mean()
            avg_diff = case_data['r2_diff'].mean()
            
            logger.info(f"  {case}: {len(case_data)} proteins")
            logger.info(f"    Better: {case_better}, Worse: {case_worse}, Same: {case_same}")
            logger.info(f"    Avg R² - 04: {avg_r2_04:.3f}, Std: {avg_r2_std:.3f}, Diff: {avg_diff:.3f}")
    
    # Statistical significance
    from scipy import stats
    try:
        t_stat, p_value = stats.ttest_rel(comparison_df['r2_04'], comparison_df['r2_standardized'])
        logger.info(f"\nStatistical Analysis:")
        logger.info(f"  Paired t-test p-value: {p_value:.4f}")
        logger.info(f"  Statistically significant: {'Yes' if p_value < 0.05 else 'No'}")
    except ImportError:
        logger.info(f"\nStatistical Analysis: scipy not available")
    
    return comparison_df

def main():
    """Main comparison function"""
    logger.info("Starting model results comparison...")
    
    # Compare results
    comparison_df = compare_results()
    
    if comparison_df.empty:
        logger.error("No comparison data available")
        return
    
    # Analyze comparison
    comparison_df = analyze_comparison(comparison_df)
    
    # Save results
    output_file = "/home/serramelendezcsm/RA/Avoidome/analyses/avoidome_qsar_similarity_expansion/model_comparison_results.csv"
    comparison_df.to_csv(output_file, index=False)
    logger.info(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
