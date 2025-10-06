# Standardized QSAR Models with Morgan Fingerprints and Physicochemical Descriptors

This directory contains QSAR models built using Morgan fingerprints and physicochemical descriptors for proteins in the Avoidome dataset using Papyrus bioactivity data.

## Overview

- **Total Proteins Processed**: 52
- **Successfully Modeled**: 27 proteins
- **Skipped (Insufficient Data)**: 24 proteins  
- **Errors**: 1 protein

## Model Performance Summary

### Regression Models (R²)
- **Average R²**: 0.557
- **Best Performing**: XDH (0.736), CHRM3 (0.730), ADRB2 (0.697)
- **Range**: 0.341 - 0.736

### Classification Models (Accuracy)
- **Average Accuracy**: 0.853
- **Best Performing**: SCN5A (0.988), KCNH2 (0.945), CYP3A4 (0.963)
- **Range**: 0.739 - 0.988

### Classification Models (F1 Score)
- **Average F1**: 0.709
- **Best Performing**: ADRB2 (0.895), CHRM3 (0.876), HSD11B1 (0.859)
- **Range**: 0.333 - 0.895

## Directory Structure

```
standardized_qsar_models/
├── morgan_regression/          # Regression models
│   ├── CYP1A2_P05177/         # Individual protein directories
│   │   ├── model.pkl          # Trained model
│   │   ├── results.json       # Detailed results
│   │   ├── predictions.csv    # Cross-validation predictions
│   │   └── feature_importance.csv
│   └── ...
├── morgan_classification/      # Classification models
│   ├── CYP1A2_P05177/         # Individual protein directories
│   │   ├── model.pkl          # Trained model
│   │   ├── results.json       # Detailed results
│   │   ├── predictions.csv    # Cross-validation predictions
│   │   └── feature_importance.csv
│   └── ...
├── modeling_summary.csv        # Summary of all models
└── morgan_qsar.log            # Detailed execution log
```

## Model Details

### Features
- **Morgan Fingerprints**: 2048 bits (radius=2)
- **Physicochemical Descriptors**: 14 features
  - ALogP, Molecular Weight, H-bond donors/acceptors
  - Rotatable bonds, atoms, rings, surface areas
  - LogS, formal charge, etc.
- **Total Features**: 2062

### Model Configuration
- **Algorithm**: Random Forest
- **Cross-Validation**: 5-fold CV
- **Minimum Samples**: 50 per protein
- **Classification Threshold**: pchembl_value > 7.0

### Data Source
- **Dataset**: Papyrus (version='latest', plusplus=True)
- **Target Variable**: pchembl_value_Mean (averaged bioactivity values)
- **SMILES**: Canonical SMILES from Papyrus

## Successfully Modeled Proteins

| Protein | UniProt ID | Samples | R² | Accuracy | F1 | AUC |
|---------|------------|---------|----|---------|----|----|
| CYP1A2 | P05177 | 329 | 0.510 | 0.863 | 0.634 | 0.883 |
| CYP2C9 | P11712 | 466 | 0.535 | 0.957 | 0.412 | 0.947 |
| CYP2D6 | P10635 | 607 | 0.414 | 0.909 | 0.409 | 0.770 |
| CYP3A4 | P08684 | 1387 | 0.572 | 0.963 | 0.633 | 0.887 |
| XDH | P47989 | 98 | 0.736 | 0.867 | 0.735 | 0.928 |
| MAOA | P21397 | 1130 | 0.560 | 0.925 | 0.632 | 0.897 |
| MAOB | P27338 | 1519 | 0.586 | 0.841 | 0.743 | 0.901 |
| ALDH1A1 | P00352 | 150 | 0.490 | 0.833 | 0.699 | 0.889 |
| HSD11B1 | P28845 | 2046 | 0.630 | 0.825 | 0.859 | 0.903 |
| NR1I3 | Q14994 | 59 | 0.562 | 0.881 | 0.788 | 0.943 |
| NR1I2 | O75469 | 344 | 0.595 | 0.890 | 0.648 | 0.894 |
| KCNH2 | Q12809 | 4492 | 0.501 | 0.945 | 0.643 | 0.920 |
| SCN5A | Q14524 | 329 | 0.436 | 0.988 | 0.333 | 0.744 |
| HTR2B | P41595 | 1180 | 0.341 | 0.739 | 0.614 | 0.786 |
| SLC6A4 | P31645 | 2958 | 0.579 | 0.807 | 0.823 | 0.888 |
| SLC6A3 | Q01959 | 1646 | 0.623 | 0.832 | 0.743 | 0.890 |
| SLC6A2 | P23975 | 2021 | 0.577 | 0.798 | 0.785 | 0.869 |
| ADRA1A | P25100 | 566 | 0.520 | 0.809 | 0.860 | 0.882 |
| ADRA2A | P08913 | 526 | 0.489 | 0.810 | 0.776 | 0.887 |
| ADRB1 | P08588 | 614 | 0.545 | 0.816 | 0.784 | 0.884 |
| ADRB2 | P07550 | 860 | 0.697 | 0.857 | 0.895 | 0.917 |
| CNR2 | P34972 | 4295 | 0.498 | 0.778 | 0.785 | 0.854 |
| CHRM1 | P11229 | 1103 | 0.488 | 0.796 | 0.702 | 0.853 |
| CHRM2 | P08172 | 721 | 0.600 | 0.809 | 0.805 | 0.892 |
| CHRM3 | P20309 | 1209 | 0.730 | 0.847 | 0.876 | 0.931 |
| CHRNA7 | P36544 | 595 | 0.578 | 0.839 | 0.742 | 0.894 |
| HRH1 | P35367 | 945 | 0.642 | 0.812 | 0.788 | 0.881 |

## Usage

### Loading a Model
```python
import pickle
import pandas as pd

# Load regression model
with open('morgan_regression/CYP1A2_P05177/model.pkl', 'rb') as f:
    imputer, model = pickle.load(f)

# Load results
results = pd.read_json('morgan_regression/CYP1A2_P05177/results.json')
```

### Making Predictions
```python
# Prepare features (Morgan + physicochemical descriptors)
# Use the same feature calculation as in morgan_qsar_modeling.py

# Impute missing values
features_imputed = imputer.transform(features)

# Make predictions
predictions = model.predict(features_imputed)
```

## Notes

- Models are saved with imputers to handle missing values
- Feature importance is available for each model
- Cross-validation predictions are saved for analysis
- All models use the same feature set for consistency
- Classification threshold is pchembl_value > 7.0 (active/inactive)

## Files Generated

- **model.pkl**: Trained Random Forest model with imputer
- **results.json**: Complete model results and metrics
- **predictions.csv**: Cross-validation predictions
- **feature_importance.csv**: Feature importance rankings
- **modeling_summary.csv**: Summary of all models
- **morgan_qsar.log**: Detailed execution log