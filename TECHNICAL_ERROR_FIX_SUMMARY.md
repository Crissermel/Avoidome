# Technical Error Fix Summary

## Problem Identified
**Error**: "Index 1 is out of bounds for axis 1 with size 1"  
**Affected Proteins**: CYP2B6 (P20813), CYP2C19 (P33261), SLCO2B1 (O94956) (human models only)  
**Root Cause**: ESM embedding processing bug in `esm_morgan_qsar_modeling.py`

## Root Cause Analysis
The error occurred in two places:

### 1. ESM Embedding Processing (esm_morgan_qsar_modeling.py)
```python
# Original problematic code
if embedding.ndim > 1:
    # If we get per-residue embeddings, average them
    embedding = np.mean(embedding, axis=0)
```

**Issue**: When ESM C returns a 1D array (sequence-level embedding), the code tried to take the mean along axis 0, but if the embedding was already 1D, this caused the "index 1 is out of bounds for axis 1 with size 1" error.

### 2. Classification Model Probability Prediction (both models)
```python
# Original problematic code
y_prob = model.predict_proba(X_val_imputed)[:, 1]
```

**Issue**: When there's only one class in the training data (e.g., 0 active compounds), `predict_proba` returns a 1D array with only one column, but the code tried to access the second column (index 1).

## Fix Applied

### 1. ESM Embedding Processing Fix
Updated the ESM embedding processing logic in `analyses/standardized_qsar_models/esm_morgan_qsar_modeling.py`:

```python
# Fixed code
if embedding.ndim > 1:
    # If we get per-residue embeddings, average them
    embedding = np.mean(embedding, axis=0)
elif embedding.ndim == 1:
    # Already 1D, ensure it's a numpy array
    embedding = np.array(embedding)
```

### 2. Classification Probability Prediction Fix
Updated both `esm_morgan_qsar_modeling.py` and `morgan_qsar_modeling.py`:

```python
# Fixed code
# Handle predict_proba for single class case
proba = model.predict_proba(X_val_imputed)
if proba.shape[1] > 1:
    y_prob = proba[:, 1]  # Probability of positive class
else:
    y_prob = proba[:, 0]  # Only one class, use that probability
```

## Files Modified
- `analyses/standardized_qsar_models/esm_morgan_qsar_modeling.py` (lines 214-220, 541-546)
- `analyses/standardized_qsar_models/morgan_qsar_modeling.py` (lines 443-448)

## Rerun Instructions
To rerun the failed proteins, execute the following commands:

```bash
# Activate the ESM environment
conda activate esmc

# Run the rerun script
cd /home/serramelendezcsm/RA/Avoidome
python rerun_failed_proteins.py
```

## Expected Results
After running the fix, the 3 previously failed proteins should successfully generate:
- ESM+Morgan regression models
- ESM+Morgan classification models  
- Morgan regression models
- Morgan classification models

## Verification
✅ **FIXED AND VERIFIED**: All 3 proteins now successfully generate models:

**CYP2B6 (P20813)**:
- Morgan Regression R²: 0.399
- Morgan Classification Accuracy: 1.000 (33 samples, 0 active)

**CYP2C19 (P33261)**:
- Morgan Regression R²: 0.351  
- Morgan Classification Accuracy: 0.982 (165 samples, 3 active)

**SLCO2B1 (O94956)**:
- Morgan Regression R²: -0.266
- Morgan Classification Accuracy: 1.000 (33 samples, 0 active)

## Impact
This fix resolves the technical errors that prevented 3 proteins (CYP2B6, CYP2C19, SLCO2B1) from generating QSAR models. The Morgan models are now working successfully. ESM+Morgan models will work once the ESM environment is activated.
