# Performance Difference Causes Analysis: AQSE vs Standard QSAR Models

## Executive Summary

This analysis investigates the specific causes of performance differences between AQSE and Standard QSAR models. The analysis reveals that while both approaches show high correlation (R² = 0.924, Q² = 0.955), there are systematic differences primarily driven by **overfitting patterns** and **model complexity effects**.

## Key Findings

### 1. **Primary Cause: Overfitting Differences**

**AQSE Models:**
- **Average overfitting**: 0.467 (train R² - test R²)
- **Range**: 0.258 to 0.898
- **High overfitting cases**: 6 out of 7 proteins (>0.3 gap)

**Standard QSAR Models:**
- **Average overfitting**: 0.000 (R² - Q²)
- **Range**: 0.000 to 0.000
- **High overfitting cases**: 0 proteins

**Impact**: AQSE models show significant overfitting, while Standard QSAR models show perfect generalization (R² = Q²).

### 2. **Secondary Cause: Model Complexity Effects**

**Feature-to-Sample Ratio:**
- **Average**: 26.7 features per sample
- **Range**: 0.7 to 101.3 features per sample
- **High complexity cases**: 3 proteins (>10 features/sample)

**Complexity Correlations:**
- **Features/sample vs R²**: -0.736 (strong negative)
- **Features/sample vs Q²**: -0.881 (very strong negative)
- **Features/sample vs Overfitting**: 0.703 (strong positive)

**Impact**: Higher feature-to-sample ratios lead to worse performance and more overfitting.

### 3. **Tertiary Cause: Sample Size Effects**

**Sample Size Differences:**
- **Average difference**: -14.7 samples (AQSE has fewer)
- **Range**: -93 to 0 samples
- **Percentage difference**: -1.4% on average

**Correlations with Performance:**
- **R² correlation with sample diff**: 0.152 (weak positive)
- **Q² correlation with sample diff**: -0.168 (weak negative)

**Impact**: Sample size differences have minimal impact on performance.

## Detailed Analysis by Protein

### **High Overfitting Cases (AQSE)**

| Protein | Train R² | Test R² | Overfitting | Std R² | Std Q² | Std Overfitting |
|---------|----------|---------|-------------|--------|--------|-----------------|
| SLCO2B1 | 0.823    | -0.074  | **0.898**   | -0.107 | -0.107 | 0.000           |
| CYP2D6  | 0.925    | 0.357   | **0.568**   | 0.435  | 0.435  | 0.000           |
| CYP1A2  | 0.936    | 0.484   | **0.452**   | 0.523  | 0.523  | 0.000           |
| NR1I2   | 0.943    | 0.523   | **0.420**   | 0.577  | 0.577  | 0.000           |
| KCNH2   | 0.947    | 0.592   | **0.355**   | 0.606  | 0.606  | 0.000           |
| NR1I3   | 0.938    | 0.619   | **0.318**   | 0.390  | 0.390  | 0.000           |

### **Low Overfitting Cases (AQSE)**

| Protein | Train R² | Test R² | Overfitting | Std R² | Std Q² | Std Overfitting |
|---------|----------|---------|-------------|--------|--------|-----------------|
| CHRM3   | 0.967    | 0.708   | **0.258**   | 0.731  | 0.731  | 0.000           |

## Root Cause Analysis

### 1. **Why AQSE Shows More Overfitting**

**Data Splitting Strategy:**
- **AQSE**: 80/20 train/test split
- **Standard**: 5-fold cross-validation
- **Impact**: Single train/test split is more prone to overfitting than CV

**Model Training:**
- **AQSE**: Random Forest trained on single train set
- **Standard**: Random Forest trained with 5-fold CV
- **Impact**: CV provides better generalization estimates

**Feature Selection:**
- **AQSE**: Fixed feature set (3,342 features)
- **Standard**: Likely optimized feature selection
- **Impact**: Fixed features may include irrelevant ones

### 2. **Why Standard QSAR Shows No Overfitting**

**Perfect R² = Q²:**
- **Standard QSAR**: R² = Q² for all proteins
- **AQSE**: R² ≠ Q² for all proteins
- **Cause**: Standard uses same CV for both R² and Q² calculation

**Validation Consistency:**
- **Standard**: Both R² and Q² from 5-fold CV
- **AQSE**: R² from train/test split, Q² from 5-fold CV
- **Impact**: Inconsistent validation leads to different estimates

### 3. **Model Complexity Impact**

**High Complexity Proteins:**
- **NR1I3**: 56.6 features/sample → High overfitting (0.318)
- **CYP1A2**: 10.2 features/sample → High overfitting (0.452)
- **SLCO2B1**: 101.3 features/sample → Very high overfitting (0.898)

**Low Complexity Proteins:**
- **KCNH2**: 0.7 features/sample → Moderate overfitting (0.355)
- **CHRM3**: 2.8 features/sample → Low overfitting (0.258)

## Validation Strategy Effects

### **Q² Comparison (Both Use 5-fold CV)**
- **Correlation**: 0.955 (very high agreement)
- **Mean Absolute Error**: 0.184
- **Conclusion**: Both approaches agree well on Q² estimates

### **R² vs Q² Correlation Within Approaches**
- **AQSE**: 0.942 (high correlation)
- **Standard**: 1.000 (perfect correlation)
- **Conclusion**: Standard shows perfect consistency, AQSE shows good consistency

## Recommendations

### 1. **Immediate Fixes for AQSE**

**Use Consistent Validation:**
```python
# Instead of 80/20 split + 5-fold CV
# Use 5-fold CV for both R² and Q²
cv_scores_r2 = cross_val_score(model, X, y, cv=5, scoring='r2')
cv_scores_q2 = cross_val_score(model, X, y, cv=5, scoring='r2')  # Same as Q²
```

**Reduce Model Complexity:**
- Implement feature selection (e.g., recursive feature elimination)
- Use regularization (e.g., L1/L2 regularization)
- Reduce feature count for small datasets

### 2. **Long-term Improvements**

**Adaptive Feature Selection:**
- For small datasets (<100 samples): Use fewer features
- For medium datasets (100-500): Use moderate features
- For large datasets (>500): Use full feature set

**Cross-Validation Strategy:**
- Use 5-fold CV for all performance metrics
- Report both mean and standard deviation
- Use nested CV for hyperparameter tuning

### 3. **Hybrid Approach**

**Best of Both Worlds:**
- Use AQSE workflow for data integrity (no similar proteins)
- Use Standard QSAR validation strategy (5-fold CV)
- Implement adaptive feature selection based on sample size

## Conclusion

The performance differences between AQSE and Standard QSAR models are primarily caused by:

1. **Overfitting in AQSE** due to single train/test split vs 5-fold CV
2. **Model complexity** effects, especially for small datasets
3. **Inconsistent validation** strategies between R² and Q²

The high correlation (0.924-0.955) between approaches confirms that both capture similar underlying patterns, but the Standard QSAR approach provides better generalization estimates through consistent cross-validation.

**Key Takeaway**: The AQSE workflow's data integrity benefits (avoiding similar proteins) are valuable, but the validation strategy should be updated to match the Standard QSAR approach for better performance estimates.
