# Unified Avoidome QSAR Dashboard

A comprehensive Streamlit dashboard for QSAR modeling analysis with hierarchical organization.

## Overview

This dashboard provides a unified interface for analyzing QSAR modeling results across different approaches:
- **Data Overview**: 55 proteins with bioactivity data
- **QSAR Models**: Four different modeling approaches
- **Model Comparison**: Comprehensive comparison across all models

## Navigation Structure

### Data Overview
- **Protein Overview**: 55 proteins with organism UniProt IDs (Human, Mouse, Rat)
- **Bioactivity Overview**: Bioactivity data points for each protein

### QSAR Models
- **Morgan Regression**: Regression models with Morgan fingerprints
  - Model performance results (R², RMSE, MAE)
  - Comparison of different models (RF, SVM, etc.)
  - Top performing models
- **Morgan Classification**: Classification models with Morgan fingerprints
  - Classification performance results (F1, Accuracy, AUC)
  - Threshold optimization
  - Top performing classification models
- **ESM+Morgan Regression**: Combined ESM and Morgan regression
  - Combined descriptor performance
  - ESM-only regression (sub-analysis)
  - Descriptor combination analysis
- **ESM+Morgan Classification**: Combined ESM and Morgan classification
  - Combined classification performance
  - ESM-only classification (sub-analysis)
  - Ensemble classification results

### Model Comparison
- **Proteins and Model Performances**: 4-model performance matrix
  - Morgan Regression performance
  - Morgan Classification performance
  - ESM+Morgan Regression performance
  - ESM+Morgan Classification performance
- **Descriptor Analysis**: Morgan vs ESM descriptor comparison
- **Task Type Analysis**: Regression vs Classification analysis
- **Performance Summary**: Overall model performance assessment

## Usage

### Running the Dashboard

```bash
# Run the unified dashboard
streamlit run unified_dashboard.py

# The dashboard will be available at http://localhost:8501
```

### Data Requirements

The dashboard requires the following data files:
- `analyses/qsar_papyrus_modelling/multi_organism_results.csv` - Protein and bioactivity data
- `analyses/qsar_papyrus_modelling/prediction_results.csv` - Morgan regression results
- `analyses/qsar_papyrus_modelling/classification_results.csv` - Morgan classification results
- `analyses/qsar_papyrus_esm_emb_classification/esm_classification_results.csv` - ESM classification results

## Features

### Data Overview
- **Protein Database**: Complete list of 55 proteins with UniProt IDs
- **Bioactivity Distribution**: Interactive visualization of bioactivity data points
- **Data Quality Assessment**: Sample size and data completeness analysis
- **Summary Statistics**: Key metrics for data coverage and quality

### QSAR Models
- **Performance Metrics**: R², RMSE, MAE for regression; F1, Accuracy, AUC for classification
- **Performance Distributions**: Histograms showing model performance across proteins
- **Top Performers**: Ranking of best performing models
- **Model Comparison**: Justification for model selection (RF vs SVM, etc.)

### Model Comparison
- **4-Model Performance Matrix**: Side-by-side comparison of all model types
- **Visual Comparisons**: Box plots and histograms for performance comparison
- **Statistical Analysis**: Performance ranking and significance testing
- **Recommendations**: Best model selection per protein

## Architecture

### Data Sources
- **Papyrus++ Database**: Multi-organism bioactivity data
- **Morgan Fingerprints**: Traditional molecular descriptors
- **ESM Embeddings**: Protein language model descriptors
- **Cross-Validation**: 5-fold CV for model evaluation

### Model Types
1. **Morgan Regression**: Random Forest with Morgan fingerprints
2. **Morgan Classification**: Random Forest classification with Morgan fingerprints
3. **ESM+Morgan Regression**: Combined ESM and Morgan regression
4. **ESM+Morgan Classification**: Combined ESM and Morgan classification

### Performance Metrics
- **Regression**: R² Score, RMSE, MAE
- **Classification**: F1 Score, Accuracy, AUC, Precision, Recall

## Key Benefits

1. **Unified Interface**: Single dashboard for all QSAR analyses
2. **Hierarchical Organization**: Logical flow from data to models to comparison
3. **Comprehensive Comparison**: 4-model performance matrix for each protein
4. **Interactive Visualizations**: Plotly charts for better user experience
5. **Data-Driven Insights**: Performance distributions and statistical analysis

## Technical Details

### Dependencies
- Streamlit
- Pandas
- Matplotlib
- Seaborn
- Plotly
- NumPy

### File Structure
```
unified_dashboard.py          # Main dashboard file
UNIFIED_DASHBOARD_README.md  # This documentation
analyses/                    # Data directory
├── qsar_papyrus_modelling/  # Morgan model results
├── qsar_papyrus_esm_emb_classification/  # ESM model results
└── qsar_avoidome/          # Additional model results
```

## Integration

This unified dashboard replaces the previous separate dashboards:
- `streamlit_dashboard.py` (main dashboard)
- `papyrus_qsar_dashboard.py` (QSAR-specific dashboard)

The new dashboard provides a more organized and comprehensive view of all QSAR modeling results in a single interface.

## Future Enhancements

- Model retraining capabilities
- Interactive model selection
- Export functionality for model files
- Real-time model performance updates
- Advanced statistical analysis tools 