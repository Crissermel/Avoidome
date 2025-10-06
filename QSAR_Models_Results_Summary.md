# QSAR Models Results Summary - Avoidome Project

**Generated on:** 29 September 2025
**Project:** Avoidome QSAR Modeling Analysis

## Executive Summary

This document provides a comprehensive summary of the current results from both **Standard QSAR Models** and **AQSE (Avoidome QSAR Similarity Expansion) Models** developed for the Avoidome project. The analysis covers multiple model architectures, organism-specific approaches, and similarity-based expansion strategies.

---

## 1. Standard QSAR Models

### 1.1 Model Architecture Overview

The Standard QSAR Models use a **organism-specific architecture** where separate models are created for each protein-organism combination, rather than selecting the best organism per protein.

**Total Potential Models:** 624 (4 model types × 52 proteins × 3 organisms)  
**Minimum Data Threshold:** 30 samples per protein-organism combination  
**Total Models Generated:** 182 models

### 1.2 Model Types and Performance

#### **Morgan Models** (182 total models)
- **Architecture:** Morgan fingerprints + physicochemical descriptors
- **Fingerprint Details:** 2048-bit Morgan fingerprints (radius=2) + 14 physicochemical descriptors
- **Organism Distribution:**
  - Human: 114 models (62.6%)
  - Mouse: 12 models (6.6%)
  - Rat: 56 models (30.8%)

**Performance Metrics:**
- **Average Regression R²:** 0.571
- **Average Classification Accuracy:** 0.839
- **Average Classification F1:** 0.747
- **Best Regression R²:** 0.736 (XDH)
- **Best Classification Accuracy:** 0.988 (SCN5A)

#### **ESM+Morgan Models** (91 total models)
- **Architecture:** ESM C protein embeddings + Morgan fingerprints + physicochemical descriptors
- **Feature Details:** 1280 ESM C dimensions + 2048 Morgan bits + 14 physicochemical descriptors
- **Organism Distribution:**
  - Human: 47 models
  - Mouse: 12 models
  - Rat: 32 models

**Performance Metrics:**
- **Average Regression R²:** 0.571
- **Average Classification Accuracy:** 0.833
- **Average Classification F1:** 0.747
- **ESM Addition Impact on R²:** +0.000 (minimal improvement)
- **ESM Addition Impact on Accuracy:** -0.003 (slight decrease)

#### **AlphaFold+Morgan Models** (Human-specific)
- **Architecture:** AlphaFold protein embeddings + Morgan fingerprints + physicochemical descriptors
- **Feature Details:** AlphaFold structural embeddings + 2048 Morgan bits + 14 physicochemical descriptors
- **Scope:** Human proteins only (AlphaFold database limitation)

**Performance Metrics:**
- **Average Regression R²:** Available in dashboard (human-specific results)
- **Average Classification Accuracy:** Available in dashboard
- **Note:** Results integrated into unified dashboard for visualization

### 1.3 Model Distribution by Organism

| Organism | Total Models | Morgan Models | ESM+Morgan Models | AlphaFold+Morgan |
|----------|--------------|---------------|-------------------|------------------|
| **Human** | 114 (62.6%) | 94 | 47 | Available |
| **Mouse** | 12 (6.6%) | 12 | 12 | N/A |
| **Rat** | 56 (30.8%) | 56 | 32 | N/A |

### 1.4 Top Performing Proteins

#### **Regression Models (R²)**
1. **XDH (P47989):** 0.736
2. **CHRM3 (P20309):** 0.730
3. **ADRB2 (P07550):** 0.697
4. **CYP3A4 (P08684):** 0.572
5. **CYP2C9 (P11712):** 0.535

#### **Classification Models (Accuracy)**
1. **SCN5A (Q14524):** 0.988
2. **KCNH2 (Q12809):** 0.945
3. **CYP3A4 (P08684):** 0.963
4. **CYP2C9 (P11712):** 0.957
5. **CHRM3 (P20309):** 0.863

---

## 2. AQSE Models (Avoidome QSAR Similarity Expansion)

### 2.1 AQSE Workflow Overview

The AQSE pipeline implements a **similarity-based expansion approach** that enriches the chemical space of avoidome proteins through protein similarity search using the Papyrus database.

**Key Innovation:** Uses similar proteins to expand training data for proteins with limited bioactivity data.

### 2.2 Model Types and Distribution

#### **Standard AQSE Models** (Proteins without similar proteins)
- **Status:** ⚠️ **OPTIMIZED OUT** - Data identical to standard QSAR models
- **Reason:** No computational benefit since input data would be identical
- **Recommendation:** Use existing standard QSAR model results instead
- **Proteins Affected:** 7 proteins (proteins with no similar proteins found)

#### **Similar Proteins AQSE Models** (Proteins with similar proteins)
- **Status:** ✅ **ACTIVE** - Creates threshold-specific models
- **Approach:** Train on similar proteins + target protein data, test on target protein holdout
- **Proteins Affected:** 18 proteins (proteins with similar proteins found)
- **Total AQSE Models:** 25 out of 55 avoidome proteins

#### **Threshold Models** (Similar proteins only)
- **High Similarity (≥70% identity):** Most conservative, highest confidence
- **Medium Similarity (50-70% identity):** Balanced expansion vs. confidence  
- **Low Similarity (30-50% identity):** Maximum expansion, lower confidence

### 2.3 AQSE Performance Results

#### **Overall Performance Comparison**
- **AQSE Mean R²:** 0.163 (significantly lower than standard)
- **Standard Mean R²:** 0.554
- **Performance Gap:** -0.391 R² units

#### **Performance by Model Type**

**Standard AQSE Models (7 proteins):**
- **Mean R²:** 0.455 vs 0.451 (Standard)
- **Mean Q²:** 0.455 vs 0.451 (Standard)
- **Result:** Nearly identical performance to standard models
- **Status:** Skipped in current implementation (optimization)

**Similar Proteins AQSE Models (17 proteins):**
- **Mean R²:** 0.042 vs 0.596 (Standard)
- **Mean Q²:** 0.545 vs 0.596 (Standard)
- **Result:** Significantly lower R² but similar Q²
- **Status:** Active and being refined

### 2.4 Top Performing AQSE Models

#### **Best AQSE Models (by R²)**
1. **NR1I3 (medium threshold):** 0.603
2. **CNR2 (all thresholds):** 0.594
3. **SLC6A3 (low threshold):** 0.591
4. **HSD11B1 (all thresholds):** 0.591
5. **CYP2C19 (all thresholds):** 0.520

#### **Model Quality Distribution (AQSE)**
- **Poor (R² < 0.3):** 33 models (36.7%)
- **Fair (0.3 ≤ R² < 0.5):** 32 models (35.6%)
- **Good (0.5 ≤ R² < 0.7):** 25 models (27.8%)

### 2.5 AQSE vs Standard QSAR Direct Comparison

| Protein | Method | AQSE R² | Standard R² | AQSE Q² | Standard Q² | Performance Gap |
|---------|--------|---------|-------------|---------|-------------|-----------------|
| **Standard AQSE Models (Skipped)** | | | | | | |
| CHRM3 | Standard | 0.750 | 0.731 | 0.750 | 0.731 | +0.019 |
| KCNH2 | Standard | 0.618 | 0.606 | 0.618 | 0.606 | +0.012 |
| NR1I2 | Standard | 0.607 | 0.577 | 0.607 | 0.577 | +0.030 |
| **Similar Proteins AQSE Models** | | | | | | |
| CYP2C19 | Similar | 0.373 | 0.289 | 0.524 | 0.289 | +0.084 |
| CHRM2 | Similar | 0.256 | 0.657 | 0.646 | 0.657 | -0.401 |
| SLC6A4 | Similar | 0.221 | 0.669 | 0.703 | 0.669 | -0.448 |

---

## 3. Key Findings and Insights

### 3.1 Standard QSAR Models

#### **Strengths:**
- **Consistent Performance:** Average R² of 0.571 across all model types
- **High Classification Accuracy:** 83.9% average accuracy
- **Organism Coverage:** Comprehensive coverage across human, mouse, and rat
- **Robust Architecture:** Morgan fingerprints + physicochemical descriptors provide reliable performance

#### **Limitations:**
- **ESM Addition Impact:** Minimal improvement (+0.000 R²) from ESM C embeddings
- **Data Requirements:** 30+ samples per protein-organism combination required
- **Limited Expansion:** No mechanism to leverage similar proteins for data augmentation

### 3.2 AQSE Models

#### **Strengths:**
- **Data Expansion:** Successfully leverages similar proteins to increase training data
- **Conceptual Innovation:** Novel approach to address data scarcity in avoidome proteins
- **Threshold Flexibility:** Multiple similarity thresholds allow for different confidence levels

#### **Challenges:**
- **Performance Issues:** Similar proteins models show significantly lower R² (0.042 vs 0.596)
- **Overfitting:** High overfitting in AQSE models (0.467 average gap)
- **Validation Strategy:** Inconsistent validation between R² and Q² calculations
- **Model Complexity:** High feature-to-sample ratios lead to worse performance

### 3.3 Optimization Implementations

#### **Standard AQSE Models Optimization:**
- **Status:** ✅ **IMPLEMENTED**
- **Change:** Skip model creation for proteins without similar proteins
- **Benefit:** Saves computational resources, avoids redundant work
- **Reasoning:** Data would be identical to standard QSAR models

#### **Similar Proteins AQSE Models:**
- **Status:** ⚠️ **NEEDS REFINEMENT**
- **Issues:** Low R² values, high overfitting, poor cross-protein generalization
- **Recommendations:**
  - Implement stricter similarity thresholds
  - Use consistent 5-fold CV for validation
  - Implement adaptive feature selection
  - Try alternative model architectures

---

## 4. Technical Specifications

### 4.1 Feature Engineering

#### **Molecular Descriptors:**
- **Morgan Fingerprints:** 2048-bit (radius=2, ECFP4-like)
- **Physicochemical Descriptors:** 14 features (MW, LogP, TPSA, HBD, HBA, etc.)
- **Total Molecular Features:** 2,062 per compound

#### **Protein Descriptors:**
- **ESM C Embeddings:** 1,280 dimensions
- **AlphaFold Embeddings:** Variable dimensions (structure-based)
- **Total Features per Model:** 3,342 (Morgan + Physicochemical + ESM C)

### 4.2 Model Architecture

#### **Algorithm:** Random Forest Regressor
- **Cross-validation:** 5-fold CV
- **Hyperparameters:** n_estimators=100, random_state=42
- **Feature Scaling:** Not required (Random Forest handles mixed types)

#### **Validation Strategy:**
- **Standard QSAR:** 5-fold CV for both R² and Q²
- **AQSE Similar Proteins:** 80/20 train/test split + 5-fold CV
- **AQSE Standard:** 5-fold CV (now skipped)

---

## 5. Recommendations

### 5.1 For Standard QSAR Models
- ✅ **Continue current approach** - performing well with consistent results
- ✅ **Use Morgan + Physicochemical** as primary architecture
- ⚠️ **Consider ESM C addition** only for specific high-value targets
- ✅ **Maintain organism-specific modeling** for comprehensive coverage

### 5.2 For AQSE Models
- ✅ **Skip Standard AQSE models** - optimization implemented
- ⚠️ **Refine Similar Proteins approach:**
  - Implement consistent 5-fold CV validation
  - Use adaptive feature selection based on sample size
  - Try stricter similarity thresholds (≥80% identity)
  - Consider ensemble methods for cross-protein prediction
- 🔄 **Investigate alternative architectures:**
  - Neural networks for cross-protein learning
  - Transfer learning approaches
  - Weighted training based on similarity scores

### 5.3 For Future Development
- **Hybrid Approach:** Combine best of both approaches
- **Feature Engineering:** Investigate protein family-specific descriptors
- **Validation:** Standardize validation strategies across all approaches
- **Performance Monitoring:** Implement automated performance tracking

---

## 6. Data Sources and Availability

### 6.1 Standard QSAR Models
- **Location:** `/analyses/standardized_qsar_models/`
- **Results Files:** `modeling_summary.csv`, individual model directories
- **Dashboard:** Available in `qsar_modeling_dashboard.py`

### 6.2 AQSE Models
- **Location:** `/analyses/avoidome_qsar_similarity_expansion/04_qsar_models_temp/`
- **Results Files:** `aqse_model_results.csv`, `workflow_summary.json`
- **Comparison:** `AQSE_vs_Standard_QSAR_Complete_Comparison.md`

### 6.3 Visualization and Analysis
- **Unified Dashboard:** `unified_dashboard.py` (comprehensive analysis)
- **QSAR Dashboard:** `qsar_modeling_dashboard.py` (focused QSAR analysis)
- **MCC Analysis:** `/analyses/mcc_comparison/` (regression vs classification)

---

## 7. Conclusion

The Avoidome QSAR modeling project has successfully implemented multiple modeling approaches with varying degrees of success:

1. **Standard QSAR Models** provide reliable, consistent performance across organisms and model types
2. **AQSE Models** show promise for data expansion but require significant refinement for similar proteins approach
3. **Optimization strategies** have been implemented to improve computational efficiency
4. **Comprehensive evaluation** reveals clear performance patterns and areas for improvement

The project demonstrates the value of both traditional QSAR approaches and innovative similarity-based expansion methods, while highlighting the importance of proper validation strategies and model complexity management.

---

