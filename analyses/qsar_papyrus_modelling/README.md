# Papyrus QSAR Modeling

This directory contains a minimal implementation of QSAR (Quantitative Structure-Activity Relationship) modeling using the Papyrus database for avoidome proteins.

## Architecture

The prediction architecture follows these steps:

1. **Data Retrieval**: For each protein (each row of `papyrus_protein_check_results.csv`), retrieve bioactivity points for human, mouse, and rat UniProt IDs from Papyrus
2. **Data Pooling**: Pool all bioactivity data from the three organisms for each protein
3. **Data Shuffling**: Shuffle the combined bioactivity points
4. **Fingerprinting**: Create Morgan fingerprints (radius=2, 2048 bits) from SMILES strings
5. **Modeling**: Train a Random Forest model with 5-fold cross-validation
6. **Results**: Create a table with results from all folds

## Files

- `minimal_papyrus_prediction.py`: Main script implementing the complete pipeline
- `test_first_protein.py`: Test script to verify implementation with the first protein only
- `requirements.txt`: Python dependencies
- `README.md`: This documentation file

## Usage

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Test with First Protein

To test the implementation with only the first protein:

```bash
python test_first_protein.py
```

This will:
- Load the first protein from `papyrus_protein_check_results.csv`
- Retrieve bioactivity data for human, mouse, and rat UniProt IDs
- Pool and process the data
- Train a 5-fold CV Random Forest model
- Save results to `test_first_protein_results.csv`

### Run Complete Pipeline

To run the complete pipeline for all proteins:

```bash
python minimal_papyrus_prediction.py
```

This will:
- Process all proteins in the dataset
- Train models for each protein with sufficient data
- Save comprehensive results to `prediction_results.csv`

## Data Sources

- **Protein List**: `/home/serramelendezcsm/RA/Avoidome/processed_data/papyrus_protein_check_results.csv`
- **Bioactivity Data**: Papyrus database (accessed via papyrus Python package)

## Output Files

- `prediction_results.csv`: Results for all proteins and folds
- `test_first_protein_results.csv`: Results for the first protein test
- `papyrus_prediction.log`: Main execution log
- `test_first_protein.log`: Test execution log

## Model Details

- **Algorithm**: Random Forest Regressor
- **Cross-validation**: 5-fold
- **Fingerprints**: Morgan fingerprints (radius=2, 2048 bits)
- **Metrics**: MSE, RMSE, MAE, R²

## Notes

- Only proteins with sufficient bioactivity data (≥10 samples) are modeled
- Data from human, mouse, and rat organisms are pooled together
- Invalid SMILES strings are filtered out
- Duplicate entries are removed
- Results include fold-wise performance metrics

## Future Enhancements

- Hyperparameter optimization
- Additional molecular descriptors
- Ensemble methods
- Model interpretation tools
- Visualization of results 