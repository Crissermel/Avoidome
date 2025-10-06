# QSAR Model Documentation

## Overview

This document describes the QSAR (Quantitative Structure-Activity Relationship) models developed for predicting pChEMBL values from molecular structures in the Avoidome dataset.

## Model Performance Summary

### **Overall Statistics:**
- **Total Models Trained**: 33
- **Average R²**: 0.563
- **Average RMSE**: 0.688
- **Best R²**: 0.802 (P31645)
- **Worst R²**: 0.197 (P00352)

### **Performance Distribution:**
- **Excellent (R² > 0.7)**: 6 models (18%)
- **Good (R² 0.5-0.7)**: 15 models (45%)
- **Moderate (R² 0.3-0.5)**: 10 models (30%)
- **Poor (R² < 0.3)**: 2 models (6%)

## Model Architecture

### **Model Type: Random Forest Regressor**

**Why Random Forest?**
- **Robust**: Handles non-linear relationships well
- **Feature Importance**: Provides insights into which molecular features matter most
- **No Overfitting**: Ensemble method reduces overfitting
- **No Scaling Required**: Works well with mixed feature types
- **Interpretable**: Can analyze feature importance

### **Architecture Components:**

#### **1. Input Layer: Molecular Descriptors (522 features)**

**Physicochemical Properties:**
- Molecular weight (MolWt)
- Octanol-water partition coefficient (LogP)
- Hydrogen bond donors (HBD)
- Hydrogen bond acceptors (HBA)
- Topological polar surface area (TPSA)
- Rotatable bonds count
- Aromatic rings count

**Morgan Fingerprints (512-bit):**
- Extended Connectivity Fingerprints (ECFP-like)
- Captures molecular substructures
- Binary representation of molecular features
- 512-bit vectors for computational efficiency

**Structural Descriptors:**
- Ring counts (aliphatic, aromatic)
- Atom counts (C, N, O, S, P, F, Cl, Br, I)
- Bond counts (single, double, triple, aromatic)
- Connectivity indices

#### **2. Ensemble Layer: Multiple Decision Trees**

**Bootstrap Sampling:**
- Each tree trained on random subset of data (with replacement)
- Creates diversity among trees
- Reduces overfitting

**Feature Randomization:**
- Each split considers random subset of features
- Further increases tree diversity
- Improves generalization

**Voting/Averaging:**
- Final prediction is average of all tree predictions
- Reduces variance in predictions
- Provides more stable results

#### **3. Output Layer: pChEMBL Prediction**

**Continuous Value:**
- Predicted pChEMBL activity score
- Range: typically 4-10 (log scale)
- Higher values indicate stronger binding

**Confidence Estimation:**
- Model provides uncertainty estimates
- Based on tree prediction variance
- Useful for risk assessment

## Training Process

### **Data Preparation:**

1. **SMILES Input**: Canonical SMILES strings for each molecule
2. **Descriptor Calculation**: RDKit generates 522 molecular descriptors
3. **Target Selection**: Separate model for each protein target
4. **Data Split**: 80% training, 20% testing
5. **Global Caching**: Descriptors calculated once per unique molecule

### **Model Training:**

1. **Cross-Validation**: 5-fold CV for robust evaluation
2. **Hyperparameter Optimization**: Grid search for optimal parameters
3. **Feature Selection**: Automatic feature importance ranking
4. **Model Validation**: Multiple metrics (R², RMSE, MAE)

### **Hyperparameters:**

```python
RandomForestRegressor(
    n_estimators=100,      # Number of trees
    max_depth=None,        # Maximum tree depth
    min_samples_split=2,   # Minimum samples to split
    min_samples_leaf=1,    # Minimum samples per leaf
    max_features='sqrt',   # Features per split
    random_state=42        # Reproducibility
)
```

## Performance Metrics

### **R² (Coefficient of Determination):**
- **Range**: 0 to 1
- **Interpretation**: Proportion of variance explained by model
- **Formula**: R² = 1 - (SS_res / SS_tot)

### **RMSE (Root Mean Square Error):**
- **Range**: 0 to ∞
- **Interpretation**: Average prediction error
- **Formula**: RMSE = √(Σ(y_pred - y_true)² / n)

### **MAE (Mean Absolute Error):**
- **Range**: 0 to ∞
- **Interpretation**: Average absolute prediction error
- **Formula**: MAE = Σ|y_pred - y_true| / n

### **CV R² (Cross-Validation R²):**
- **Range**: 0 to 1
- **Interpretation**: More robust performance estimate
- **Method**: 5-fold cross-validation

## Performance Interpretation

### **R² Score Guidelines:**
- **R² > 0.7**: Excellent predictive performance
- **R² 0.5-0.7**: Good predictive performance
- **R² 0.3-0.5**: Moderate predictive performance
- **R² < 0.3**: Poor predictive performance

### **RMSE Guidelines:**
- **RMSE < 0.5**: Excellent accuracy
- **RMSE 0.5-0.8**: Good accuracy
- **RMSE 0.8-1.2**: Moderate accuracy
- **RMSE > 1.2**: Poor accuracy

## Key Innovations

### **1. Global Molecule Caching:**
- Descriptors calculated once per unique molecule
- Massive time savings (90% reduction in computation)
- Reusable across multiple targets
- Storage: ~1-3 GB for all unique molecules

### **2. Single-Target Models:**
- Specialized models for each protein target
- Captures target-specific structure-activity relationships
- Better performance than multi-target models
- 33 individual models for 33 targets

### **3. Comprehensive Descriptors:**
- 522 features capturing molecular properties
- Mix of physicochemical and structural descriptors
- Morgan fingerprints for substructure recognition
- Balanced representation of molecular features

### **4. Robust Validation:**
- 5-fold cross-validation for reliable performance estimates
- Multiple metrics for comprehensive evaluation
- Out-of-sample testing for generalization assessment

## Top Performing Models

### **Best Models (R² > 0.75):**
1. **P31645**: R² = 0.802, RMSE = 0.551
2. **P23975**: R² = 0.782, RMSE = 0.534
3. **P20309**: R² = 0.779, RMSE = 0.815
4. **Q9GZZ6**: R² = 0.757, RMSE = 0.753
5. **Q12809**: R² = 0.743, RMSE = 0.471

### **Worst Models (R² < 0.3):**
1. **P00352**: R² = 0.197, RMSE = 0.422
2. **Q14994**: R² = 0.201, RMSE = 1.011

## Model Analysis

### **Feature Importance:**
- Morgan fingerprints typically most important
- LogP and molecular weight often significant
- H-bond descriptors important for binding prediction
- Structural features contribute to specificity

### **Target Characteristics:**
- **High-performing targets**: Often have well-defined binding sites
- **Low-performing targets**: May have flexible binding modes
- **Data quality**: More samples generally lead to better models
- **Chemical diversity**: Broader chemical space improves generalization

## Usage

### **Making Predictions:**
```bash
python predict.py --smiles "CCc1ccc(C(=O)N2Cc3ccccc3C[C@H]2CN2CCOCC2)c(-c2cc(C(=O)N(c3ccc(O)cc3)c3ccc4c(ccn4C)c3)c3n2CCCC3)c1" --target P31645
```

### **Model Files:**
Each target has its own directory containing:
- `model.pkl`: Trained Random Forest model
- `performance_plot.png`: Training vs test performance
- `feature_importance.png`: Feature importance plot
- `predictions.csv`: Test set predictions

### **Cache Management:**
```bash
# Check cache status
python manage_cache.py --status

# Pre-calculate descriptors
python manage_cache.py --pre-calculate

# Clear cache
python manage_cache.py --clear
```

## Technical Details

### **Software Stack:**
- **Python**: 3.8+
- **Scikit-learn**: Random Forest implementation
- **RDKit**: Molecular descriptor calculation
- **Pandas**: Data manipulation
- **Joblib**: Model serialization
- **Matplotlib/Seaborn**: Visualization

### **Computational Requirements:**
- **Memory**: 8-16 GB RAM recommended
- **Storage**: 5-10 GB for models and cache
- **CPU**: Multi-core for parallel processing
- **Time**: 30-60 minutes for full pipeline

### **Data Requirements:**
- **Minimum samples**: 50 per target
- **SMILES format**: Canonical SMILES strings
- **pChEMBL values**: Standardized activity scores
- **Target IDs**: UniProt identifiers




---

*This documentation was generated automatically from the QSAR modeling pipeline. For questions or issues, please refer to the main README.md file.* 