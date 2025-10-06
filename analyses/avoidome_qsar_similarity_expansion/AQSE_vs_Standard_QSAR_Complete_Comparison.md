# AQSE vs Standard QSAR Models - Complete Comparison Report

## Executive Summary

This report compares the performance of the AQSE (Avoidome QSAR Similarity Expansion) workflow against the standard QSAR models. The AQSE workflow now includes two distinct approaches:
1. **Standard AQSE models**: For proteins with no similar proteins (using 5-fold CV)
2. **Similar proteins AQSE models**: For proteins with similar proteins (train on similar proteins, test on target protein)

## Key Findings

### Model Distribution
- **AQSE Standard models**: 7 proteins (proteins with no similar proteins)
- **AQSE Similar models**: 18 proteins (proteins with similar proteins)
- **Total AQSE models**: 25 out of 55 avoidome proteins
- **Standard QSAR models**: 33 proteins
- **Common proteins**: 24 proteins for direct comparison

### Performance Comparison

#### Overall Performance
- **AQSE Mean R²**: 0.163 (significantly lower than standard)
- **Standard Mean R²**: 0.554
- **Performance Gap**: -0.391 R² units

#### Performance by Model Type

**Standard AQSE Models (7 proteins):**
- Mean R²: 0.455 vs 0.451 (Standard)
- Mean Q²: 0.455 vs 0.451 (Standard)
- **Result**: Nearly identical performance to standard models

**Similar Proteins AQSE Models (17 proteins):**
- Mean R²: 0.042 vs 0.596 (Standard)
- Mean Q²: 0.545 vs 0.596 (Standard)
- **Result**: Significantly lower R² but similar Q²

### Detailed Protein-by-Protein Analysis

| Protein | Method | AQSE R² | Standard R² | AQSE Q² | Standard Q² | Performance Gap |
|---------|--------|---------|-------------|---------|-------------|-----------------|
| **Standard AQSE Models** | | | | | | |
| CHRM3 | Standard | 0.750 | 0.731 | 0.750 | 0.731 | +0.019 |
| KCNH2 | Standard | 0.618 | 0.606 | 0.618 | 0.606 | +0.012 |
| NR1I2 | Standard | 0.607 | 0.577 | 0.607 | 0.577 | +0.030 |
| CYP1A2 | Standard | 0.495 | 0.523 | 0.495 | 0.523 | -0.028 |
| CYP2D6 | Standard | 0.463 | 0.435 | 0.463 | 0.435 | +0.028 |
| NR1I3 | Standard | 0.431 | 0.390 | 0.431 | 0.390 | +0.041 |
| SLCO2B1 | Standard | -0.176 | -0.107 | -0.176 | -0.107 | -0.069 |
| **Similar Proteins AQSE Models** | | | | | | |
| CYP2C19 | Similar | 0.373 | 0.289 | 0.524 | 0.289 | +0.084 |
| CHRM2 | Similar | 0.256 | 0.657 | 0.646 | 0.657 | -0.401 |
| SLC6A4 | Similar | 0.221 | 0.669 | 0.703 | 0.669 | -0.448 |
| CHRNA7 | Similar | 0.162 | 0.652 | 0.653 | 0.652 | -0.490 |
| HSD11B1 | Similar | 0.115 | 0.512 | 0.471 | 0.512 | -0.397 |
| CHRM1 | Similar | 0.116 | 0.632 | 0.752 | 0.632 | -0.516 |
| SLC6A3 | Similar | 0.112 | 0.700 | 0.719 | 0.700 | -0.588 |
| CNR2 | Similar | 0.106 | 0.616 | 0.562 | 0.616 | -0.510 |
| XDH | Similar | 0.305 | 0.679 | 0.766 | 0.679 | -0.374 |
| MAOB | Similar | 0.137 | 0.642 | 0.465 | 0.642 | -0.505 |
| CYP2B6 | Similar | 0.069 | 0.376 | 0.185 | 0.376 | -0.307 |
| CYP2C9 | Similar | 0.060 | 0.538 | 0.280 | 0.538 | -0.478 |
| ADRB2 | Similar | -0.063 | 0.708 | 0.618 | 0.708 | -0.771 |
| ADRB1 | Similar | -0.046 | 0.627 | 0.491 | 0.627 | -0.673 |
| SLC6A2 | Similar | -0.044 | 0.635 | 0.388 | 0.635 | -0.679 |
| ADRA2A | Similar | -0.202 | 0.544 | 0.440 | 0.544 | -0.746 |
| MAOA | Similar | -0.958 | 0.655 | 0.593 | 0.655 | -1.613 |

## Analysis of Performance Differences

### 1. Standard AQSE Models vs Standard QSAR
- **Performance**: Nearly identical (R² correlation: 0.838)
- **Conclusion**: When using the same validation approach (5-fold CV), AQSE performs as well as standard QSAR
- **Key insight**: The AQSE approach works well for proteins without similar proteins

### 2. Similar Proteins AQSE Models vs Standard QSAR
- **Performance**: Significantly lower R² (0.042 vs 0.596)
- **Q² Performance**: Similar (0.545 vs 0.596)
- **Conclusion**: The similar proteins approach shows promise but needs refinement

### 3. Key Differences Between Approaches

#### Data Usage
- **Standard AQSE**: Uses all available data for the target protein with 5-fold CV
- **Similar Proteins AQSE**: Trains on similar proteins, tests on target protein
- **Standard QSAR**: Uses all available data for each protein with 5-fold CV

#### Sample Sizes
- **AQSE total samples**: 36,669
- **Standard total samples**: 34,522
- **Difference**: +2,147 samples (AQSE has more data)

#### Feature Engineering
- **AQSE**: Morgan fingerprints (2048) + Physicochemical (14) + ESM C (1280) = 3,342 features
- **Standard QSAR**: Unknown feature count (not available in results)

## Recommendations

### 1. For Standard AQSE Models
- **Status**: ✅ **Working well**
- **Action**: Continue using this approach for proteins without similar proteins
- **Performance**: Comparable to standard QSAR models

### 2. For Similar Proteins AQSE Models
- **Status**: ⚠️ **Needs improvement**
- **Issues**:
  - Low R² values (mean: 0.042)
  - High RMSE values (mean: 0.948)
  - Poor generalization from similar proteins to target protein

### 3. Potential Improvements for Similar Proteins Models

#### A. Data Quality Issues
- **Problem**: Similar proteins may have different binding patterns
- **Solution**: Implement stricter similarity thresholds or protein family clustering

#### B. Feature Alignment
- **Problem**: ESM C descriptors are calculated for target protein only
- **Solution**: Calculate ESM C descriptors for each similar protein and use ensemble features

#### C. Training Strategy
- **Problem**: Simple train/test split may not be optimal
- **Solution**: Implement weighted training based on protein similarity scores

#### D. Model Architecture
- **Problem**: Random Forest may not be optimal for cross-protein prediction
- **Solution**: Try more sophisticated models (e.g., neural networks, ensemble methods)

## Technical Implementation Notes

### AQSE Workflow Features
1. **Descriptor Caching**: Successfully implemented to speed up subsequent runs
2. **5-fold Cross-Validation**: Properly implemented for standard models
3. **Similar Proteins Detection**: Working correctly (18 proteins identified)
4. **Morgan Fingerprints**: 398,709 compounds processed successfully

### Data Sources
- **AQSE**: Direct from Papyrus database (latest, plusplus=True)
- **Standard QSAR**: Pre-processed data (exact source unknown)

## Conclusion

The AQSE workflow shows **mixed results**:

1. **✅ Standard AQSE models** (proteins without similar proteins) perform comparably to standard QSAR models
2. **⚠️ Similar proteins AQSE models** show promise but need significant refinement

The **similar proteins approach** is conceptually sound but requires optimization in:
- Protein similarity thresholds
- Feature engineering strategies
- Model training approaches
- Cross-protein generalization techniques

**Next Steps**:
1. Investigate why similar proteins models perform poorly
2. Implement improved feature engineering for cross-protein prediction
3. Test alternative model architectures
4. Refine protein similarity criteria

---

*Report generated on: 2025-09-22*  
*AQSE Workflow Version: 2.0 (with similar proteins support)*  
*Total models analyzed: 58 (25 AQSE + 33 Standard)*
