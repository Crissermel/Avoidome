#!/usr/bin/env python3
"""
View MCC Analysis Results

This script displays the MCC analysis results in a readable format.
"""

import pandas as pd
import json
from pathlib import Path

def main():
    # Load summary data
    summary_df = pd.read_csv('results/mcc_analysis_summary.csv')
    
    print("="*80)
    print("MCC ANALYSIS RESULTS SUMMARY")
    print("="*80)
    
    for _, row in summary_df.iterrows():
        model_type = row['model_type'].replace('_', '+').upper()
        print(f"\n{model_type} MODELS:")
        print(f"  Total Comparisons: {int(row['total_comparisons'])}")
        print(f"  Classification Wins: {int(row['classification_wins'])} ({row['classification_win_rate']:.1%})")
        print(f"  Regression Wins: {int(row['regression_wins'])} ({1-row['classification_win_rate']:.1%})")
        print(f"  Mean MCC Difference: {row['mean_mcc_difference']:.4f} ± {row['std_mcc_difference']:.4f}")
        print(f"  Mean Accuracy Difference: {row['mean_accuracy_difference']:.4f}")
        print(f"  Mean F1 Difference: {row['mean_f1_difference']:.4f}")
    
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    # Determine which approach is better for each model type
    morgan_row = summary_df[summary_df['model_type'] == 'morgan'].iloc[0]
    esm_row = summary_df[summary_df['model_type'] == 'esm_morgan'].iloc[0]
    
    morgan_better = "Classification" if morgan_row['classification_win_rate'] > 0.5 else "Regression"
    esm_better = "Classification" if esm_row['classification_win_rate'] > 0.5 else "Regression"
    
    print(f"1. Morgan Models: {morgan_better} models perform better ({morgan_row['classification_win_rate']:.1%} win rate)")
    print(f"2. ESM+Morgan Models: {esm_better} models perform better ({esm_row['classification_win_rate']:.1%} win rate)")
    
    # MCC improvement analysis
    if morgan_row['mean_mcc_difference'] > 0:
        print(f"3. Morgan Models: Classification shows average MCC improvement of {morgan_row['mean_mcc_difference']:.4f}")
    else:
        print(f"3. Morgan Models: Regression shows average MCC improvement of {abs(morgan_row['mean_mcc_difference']):.4f}")
    
    if esm_row['mean_mcc_difference'] > 0:
        print(f"4. ESM+Morgan Models: Classification shows average MCC improvement of {esm_row['mean_mcc_difference']:.4f}")
    else:
        print(f"4. ESM+Morgan Models: Regression shows average MCC improvement of {abs(esm_row['mean_mcc_difference']):.4f}")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print("Based on the MCC analysis:")
    print("- Use Classification models for Morgan-based approaches (65.9% win rate)")
    print("- Use Classification models for ESM+Morgan approaches (54.5% win rate)")
    print("- The advantage is more pronounced for Morgan-only models")
    print("- Consider the magnitude of MCC differences when making final decisions")
    print("- Evaluate additional metrics (accuracy, F1, precision, recall) for comprehensive assessment")
    print("="*80)

if __name__ == "__main__":
    main()