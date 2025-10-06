# QSAR Classification with ESM Embeddings

This directory contains scripts for performing binary classification of protein bioactivity using Morgan fingerprints combined with ESM protein embeddings.

## Overview

The classification pipeline uses:
- **Morgan fingerprints** for molecular representation (2048 bits, radius 2)
- **ESM embeddings** for protein representation (from ESM-2 model)
- **Random Forest classifier** with 5-fold cross-validation
- **Activity threshold** to convert continuous bioactivity values to binary classes

## Files

### Core Scripts

1. **`minimal_papyrus_esm_classification.py`**
   - Main classification script
   - Implements `PapyrusESMQSARClassifier` class
   - Uses activity threshold (default: 6.0) to convert continuous values to binary classes
   - Trains Random Forest classifier with balanced class weights
   - Reports accuracy, precision, recall, F1 score, and AUC

2. **`analyze_classification_results.py`**
   - Analyzes classification results
   - Creates performance visualizations
   - Generates detailed reports
   - Identifies top and bottom performing proteins

3. **`test_classification.py`**
   - Tests the classification pipeline with a subset of proteins
   - Verifies functionality before full analysis

4. **`run_classification.py`**
   - Simple interface to run the classification pipeline
   - Supports different modes: classify, analyze, full, test

## Usage

### Basic Classification

```bash
cd /home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_esm_emb_classification

# Run classification with default threshold (6.0)
python run_classification.py --mode classify

# Run with custom threshold
python run_classification.py --mode classify --threshold 7.0
```

### Test the Pipeline

```bash
# Test the pipeline with a subset of proteins
python run_classification.py --mode test
```

### Analyze Results

```bash
# Analyze existing results and create visualizations
python run_classification.py --mode analyze
```

### Full Pipeline

```bash
# Run classification and analysis together
python run_classification.py --mode full
```

## Configuration

### Activity Threshold

The activity threshold determines how continuous bioactivity values are converted to binary classes:

- **Active (1)**: pChEMBL value ≥ threshold
- **Inactive (0)**: pChEMBL value < threshold

Default threshold: **6.0** (standard threshold for balanced classification)

### Model Parameters

- **Random Forest**: 100 estimators, balanced class weights
- **Cross-validation**: 5-fold with shuffling
- **Morgan fingerprints**: 2048 bits, radius 2
- **ESM embeddings**: 1280 dimensions

## Output Files

### Results Files

- `esm_classification_results.csv`: Main results for all proteins
- `test_results.csv`: Test results (when running test mode)

### Log Files

- `papyrus_esm_classification.log`: Main classification log
- `run_classification.log`: Runner script log
- `test_classification.log`: Test log
- `analysis.log`: Analysis log

### Reports and Visualizations

- `detailed_report.txt`: Comprehensive text report
- `plots/`: Directory containing performance visualizations
  - `performance_distributions.png`: Distribution of metrics
  - `performance_vs_samples.png`: Performance vs sample size
  - `top_proteins_f1.png`: Top performing proteins
  - `correlation_matrix.png`: Metric correlations

## Performance Metrics

The classification pipeline reports:

- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **AUC**: Area under ROC curve

## Class Distribution

The script handles class imbalance by:
- Using balanced class weights in Random Forest
- Reporting class distribution for each protein
- Skipping proteins with only one class
- Logging class balance statistics

## Data Sources

- **Protein data**: `processed_data/papyrus_protein_check_results.csv`
- **ESM embeddings**: `analyses/qsar_papyrus_esm_emb/embeddings.npy`
- **Target sequences**: `analyses/qsar_papyrus_esm_emb/targets_w_sequences.csv`
- **Bioactivity data**: Papyrus dataset (latest version)

## Dependencies

- pandas
- numpy
- scikit-learn
- rdkit
- matplotlib
- seaborn
- papyrus-scripts

## Example Results

Typical performance for a well-performing protein:
- Accuracy: 0.75-0.85
- F1 Score: 0.70-0.80
- AUC: 0.75-0.85

## Troubleshooting

### Common Issues

1. **Insufficient data**: Proteins with < 10 samples are skipped
2. **Single class**: Proteins with only active or inactive compounds are skipped
3. **Missing embeddings**: Proteins without ESM embeddings are skipped
4. **Invalid SMILES**: Compounds with invalid SMILES are filtered out

### Log Files

Check the log files for detailed error messages and processing information.

## Comparison with Regression

This classification approach complements the regression analysis by:
- Providing binary predictions (active/inactive)
- Using different evaluation metrics
- Handling class imbalance
- Offering interpretable thresholds

The classification results can be compared with regression results to understand both continuous and binary prediction performance. 