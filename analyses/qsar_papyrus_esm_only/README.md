# ESM-Only QSAR Modeling Pipeline

This directory contains the QSAR modeling pipeline that uses **only ESM protein embeddings** for bioactivity prediction, without Morgan fingerprints.

## Overview

The ESM-only approach uses protein sequence embeddings from the ESM-1b model to predict bioactivity values. Unlike the combined approach (Morgan + ESM), this method relies solely on protein information.

### Key Features

- **Protein-only modeling**: Uses only ESM embeddings (1280 dimensions)
- **No molecular descriptors**: No Morgan fingerprints or compound information
- **Protein-centric approach**: Focuses on protein sequence information
- **Random Forest regression**: 5-fold cross-validation
- **Comprehensive analysis**: Data overview and model performance evaluation

## Architecture

### Feature Engineering
- **Input**: ESM protein embeddings (1280 dimensions)
- **Processing**: Same ESM embedding repeated for all compounds of a protein
- **Output**: Feature matrix of shape (n_compounds, 1280)

### Model Training
- **Algorithm**: RandomForestRegressor
- **Cross-validation**: 5-fold with shuffling
- **Metrics**: R², RMSE, MAE
- **Minimum data**: 10 samples per protein

## Files

### Core Scripts
- `minimal_papyrus_esm_only_prediction.py`: Main QSAR modeling script
- `esm_only_data_overview.py`: Data analysis and feasibility testing

### Output Files
- `esm_only_prediction_results.csv`: Model performance results
- `esm_only_data_overview_results.csv`: Data availability analysis
- `esm_only_data_overview.png`: Visualization plots
- `esm_only_data_overview_report.txt`: Detailed text report

## Usage

### 1. Data Overview
```bash
cd analyses/qsar_papyrus_esm_only
python esm_only_data_overview.py
```

### 2. QSAR Modeling
```bash
cd analyses/qsar_papyrus_esm_only
python minimal_papyrus_esm_only_prediction.py
```

## Data Requirements

### Input Files
- `embeddings.npy`: ESM protein embeddings (from ESM-1b model)
- `targets_w_sequences.csv`: Protein sequence information
- `papyrus_protein_check_results.csv`: Protein availability data

### Output Files
- Model performance results
- Data overview analysis
- Visualization plots
- Detailed reports

## Model Performance

The ESM-only approach provides:
- **Protein-specific insights**: Each protein has unique ESM embeddings
- **Sequence-based predictions**: Relies on protein sequence information
- **Comparative analysis**: Can be compared with Morgan-only and combined approaches

## Comparison with Other Approaches

| Approach | Features | Dimensions | Information Type |
|----------|----------|------------|------------------|
| Morgan-only | Morgan fingerprints | 2048 | Molecular structure |
| ESM-only | ESM embeddings | 1280 | Protein sequence |
| Combined | Morgan + ESM | 3328 | Molecular + Protein |

## Expected Results

The ESM-only approach is expected to:
- Show lower performance than combined approaches (due to lack of molecular information)
- Provide insights into protein-specific modeling capabilities
- Demonstrate the value of protein sequence information in QSAR modeling
- Serve as a baseline for protein-centric modeling approaches

## Dependencies

See `requirements.txt` for the complete list of Python packages required.

## Logging

All scripts include comprehensive logging to track:
- Data loading progress
- Model training status
- Performance metrics
- Error handling

## Future Enhancements

Potential improvements for the ESM-only approach:
1. **Attention mechanisms**: Focus on relevant protein regions
2. **Graph neural networks**: Protein structure graphs
3. **Multi-task learning**: Multiple bioactivity endpoints
4. **Interpretability**: Understanding ESM feature importance
5. **Ensemble methods**: Combining multiple ESM models 