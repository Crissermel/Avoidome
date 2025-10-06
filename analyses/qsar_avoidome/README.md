# QSAR Modeling for Avoidome Bioactivity Data

This directory contains QSAR (Quantitative Structure-Activity Relationship) models for predicting pChEMBL values from molecular structures (SMILES) for targets in the avoidome dataset.

## Overview

The QSAR modeling system:
- Loads bioactivity data from `avoidome_bioactivity_profile.csv`
- Calculates molecular descriptors from SMILES structures (with intelligent caching)
- Trains Random Forest models for each target
- Evaluates model performance using cross-validation
- Saves trained models and performance metrics

## Performance & Caching System

### **Global Molecule Caching - Major Performance Optimization**

The system includes an intelligent **global molecule caching system** that dramatically reduces computation time by calculating descriptors only once per unique molecule:

#### **Key Innovation:**
- **Global cache**: One descriptor calculation per unique molecule across ALL targets
- **Reuse**: Same molecules appear in multiple targets, so descriptors are calculated once and reused
- **Massive efficiency**: No redundant calculations for identical molecules

#### **Time Savings:**
- **Without Caching**: 3-6 hours per run (recalculates descriptors every time)
- **With Global Caching**: 30-60 minutes per run (loads pre-calculated descriptors)
- **First-time setup**: 1-2 hours for pre-calculation of all unique molecules
- **Subsequent runs**: Near-instant loading from global cache

#### **Storage Requirements:**
- **Global cache**: 1-3 GB (one file for all unique molecules)
- **Per molecule**: ~1-5 KB (much more efficient than per-target caching)
- **Location**: `analyses/qsar_avoidome/descriptor_cache/global_molecule_descriptors.pkl`

#### **Cache Management Commands:**
```bash
# Check cache status
python analyses/qsar_avoidome/manage_cache.py --status

# Pre-calculate all descriptors (recommended for first-time setup)
python analyses/qsar_avoidome/manage_cache.py --pre-calculate

# Clear cache if needed
python analyses/qsar_avoidome/manage_cache.py --clear

# View specific target cache info
python analyses/qsar_avoidome/manage_cache.py --target P05177
```

#### **Recommended Workflow:**
1. **First time**: Run `manage_cache.py --pre-calculate` (1-2 hours)
2. **Subsequent runs**: Use `qsar_modeling.py` (30-60 minutes)
3. **Model development**: Iterate quickly with cached descriptors

## Features

### Molecular Descriptors
- **Basic descriptors**: Molecular weight, LogP, H-bond donors/acceptors, rotatable bonds, TPSA
- **Structural descriptors**: Number of atoms, rings, aromatic rings
- **Morgan fingerprints**: 512-bit ECFP-like fingerprints for molecular similarity (optimized for speed)
- **Caching**: All descriptors are automatically cached to disk for fast subsequent runs

### Machine Learning Models
- **Random Forest**: Primary model with excellent performance on molecular data
  - 100 estimators, max depth 15
  - Optimized for speed and accuracy
  - No feature scaling required

### Model Evaluation
- **R² score**: Coefficient of determination
- **RMSE**: Root mean squared error
- **MAE**: Mean absolute error
- **Cross-validation**: 5-fold CV for robust performance estimation

## Usage

### Installation
```bash
pip install -r requirements.txt
```

### Quick Start (Recommended)
```bash
# 1. Pre-calculate descriptors for all targets (one-time setup, 1-2 hours)
python analyses/qsar_avoidome/manage_cache.py --pre-calculate

# 2. Run QSAR modeling (uses cached descriptors, 30-60 minutes)
python analyses/qsar_avoidome/qsar_modeling.py

# 3. Make predictions on new molecules
python analyses/qsar_avoidome/predict.py --smiles "CCc1ccc(C(=O)N2Cc3ccccc3C[C@H]2CN2CCOCC2)c(-c2cc(C(=O)N(c3ccc(O)cc3)c3ccc4c(ccn4C)c3)c3n2CCCC3)c1" --target P05177
```

### Cache Management
```bash
# Check cache status
python analyses/qsar_avoidome/manage_cache.py --status

# Clear cache if needed
python analyses/qsar_avoidome/manage_cache.py --clear

# View specific target cache info
python analyses/qsar_avoidome/manage_cache.py --target P05177
```

### Quick Testing
```bash
# Test with a small dataset (for validation)
python analyses/qsar_avoidome/qsar_test.py
```

### Working Directory Options
You can run the scripts from different directories:

**From project root (`/home/serramelendezcsm/RA/Avoidome`):**
```bash
python analyses/qsar_avoidome/manage_cache.py --status
python analyses/qsar_avoidome/qsar_modeling.py
```

**From QSAR directory (`/home/serramelendezcsm/RA/Avoidome/analyses/qsar_avoidome`):**
```bash
cd analyses/qsar_avoidome
python manage_cache.py --status
python qsar_modeling.py
```

### Configuration
You can modify the following parameters in `qsar_modeling.py`:
- `min_samples`: Minimum number of samples required per target (default: 50)
- `test_size`: Fraction of data for testing (default: 0.2)
- Model hyperparameters in the Random Forest configuration

## Output Structure

```
qsar_avoidome/
├── README.md
├── requirements.txt
├── qsar_modeling.py
├── predict.py
├── qsar_test.py
├── manage_cache.py
├── model_performance_summary.csv
├── qsar_modeling.log
├── descriptor_cache/
│   ├── P00352_descriptors.pkl
│   ├── Q12809_descriptors.pkl
│   └── ...
└── {target_id}/
    ├── RandomForest_model.pkl
    ├── RandomForest_metadata.json
    ├── {target_id}_performance.png
    └── {target_id}_results.txt
```

## Model Files

### Model Files (.pkl)
- Trained Random Forest models saved using joblib
- Can be loaded for making predictions on new compounds

### Metadata Files (.json)
- Model performance metrics (R², RMSE, MAE)
- Feature names and target information
- Training parameters and cross-validation results

### Performance Plots (.png)
- Prediction vs actual plots
- Residual analysis
- Model performance visualization

### Log Files (.log)
- Detailed logging of the entire modeling process
- Progress tracking and error reporting
- Performance timing information

## Making Predictions

```python
import joblib
import json
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem

def load_model(target_id):
    """Load a trained QSAR model"""
    model_dir = f"/home/serramelendezcsm/RA/Avoidome/analyses/qsar_avoidome/{target_id}"
    
    # Load model
    model = joblib.load(f"{model_dir}/RandomForest_model.pkl")
    
    # Load metadata
    with open(f"{model_dir}/RandomForest_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    return model, metadata

def predict_pchembl(smiles, target_id):
    """Predict pChEMBL value for a given SMILES and target"""
    model, metadata = load_model(target_id)
    
    # Calculate descriptors (same as training)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Calculate descriptors (same as training)
    desc_dict = {
        'MolWt': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
        'TPSA': Descriptors.TPSA(mol),
        'NumAtoms': mol.GetNumAtoms(),
        'NumHeavyAtoms': mol.GetNumHeavyAtoms(),
        'NumRings': Descriptors.RingCount(mol),
        'NumAromaticRings': Descriptors.NumAromaticRings(mol)
    }
    
    # Add Morgan fingerprints (512 bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
    fp_array = np.array(fp)
    for j, bit in enumerate(fp_array):
        desc_dict[f'FP_{j}'] = bit
    
    # Create feature vector
    features = pd.DataFrame([desc_dict])
    X = features[metadata['feature_names']].values
    
    # Make prediction (no scaling needed for Random Forest)
    prediction = model.predict(X)[0]
    return prediction

# Example usage
smiles = "CCc1ccc(C(=O)N2Cc3ccccc3C[C@H]2CN2CCOCC2)c(-c2cc(C(=O)N(c3ccc(O)cc3)c3ccc4c(ccn4C)c3)c3n2CCCC3)c1"
target_id = "P05177"  # CYP1A2
prediction = predict_pchembl(smiles, target_id)
print(f"Predicted pChEMBL: {prediction:.2f}")
```

## Performance Summary

The `model_performance_summary.csv` file contains:
- Performance metrics for all models across all targets
- Best performing model for each target
- Cross-validation results

## Notes

- Models are trained on targets with at least 50 bioactivity measurements
- Random Forest models don't require feature scaling
- 512-bit Morgan fingerprints provide molecular similarity information (optimized for speed)
- Cross-validation ensures robust performance estimation
- Descriptor caching dramatically speeds up repeated runs
- Models can be used for virtual screening and lead optimization
- Comprehensive logging tracks all operations and performance metrics

## Troubleshooting

1. **RDKit installation issues**: Use conda instead of pip for RDKit
   ```bash
   conda install -c conda-forge rdkit
   ```

2. **Memory issues**: Reduce the number of Morgan fingerprint bits or use fewer targets

3. **Poor performance**: Check data quality, increase minimum samples per target, or try different model parameters 