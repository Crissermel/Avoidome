# MCC Analysis Report: Regression vs Classification QSAR Models

**Generated on:** 2025-09-11 15:50:49
**Analysis Directory:** /home/serramelendezcsm/RA/Avoidome/analyses/mcc_comparison

## Overview

This report presents a comprehensive comparison between regression and classification QSAR models using the Matthews Correlation Coefficient (MCC) as the primary metric. For regression models, predictions were binarized using a threshold of 7.0 to enable fair comparison with classification models.

## Analysis Results

### Morgan Models Comparison

- **Total Comparisons:** 44
- **Classification Wins:** 29 (65.9%)
- **Regression Wins:** 15 (34.1%)
- **Mean MCC Difference:** 0.0343 ± 0.0951
- **Mean Accuracy Difference:** -0.0013
- **Mean F1 Difference:** 0.0406

### ESM+Morgan Models Comparison

- **Total Comparisons:** 44
- **Classification Wins:** 24 (54.5%)
- **Regression Wins:** 20 (45.5%)
- **Mean MCC Difference:** 0.0236 ± 0.1086
- **Mean Accuracy Difference:** -0.0063
- **Mean F1 Difference:** 0.0350

## Conclusions

### Key Findings

1. **Morgan Models:** Classification models perform better (65.9% win rate)
2. **ESM+Morgan Models:** Classification models perform better (54.5% win rate)
3. **Morgan Models:** Classification shows average MCC improvement of 0.0343
4. **ESM+Morgan Models:** Classification shows average MCC improvement of 0.0236

### Recommendations

Based on the MCC analysis:
- Use the model type that shows higher win rate for each specific approach
- Consider the magnitude of MCC differences when making final decisions
- Evaluate additional metrics (accuracy, F1, precision, recall) for comprehensive assessment
- Consider the specific protein targets and their characteristics when choosing between regression and classification

## Files Generated

- `mcc_analysis_results.json`: Complete analysis results
- `morgan_comparison_detailed.csv`: Detailed Morgan model comparisons
- `esm_morgan_comparison_detailed.csv`: Detailed ESM+Morgan model comparisons
- `mcc_analysis_summary.csv`: Summary statistics
- `plots/`: Visualization files
  - `mcc_difference_distributions.png`: Distribution of MCC differences
  - `mcc_scatter_comparison.png`: Scatter plots of regression vs classification MCC
  - `win_rate_comparison.png`: Win rate comparison between model types

