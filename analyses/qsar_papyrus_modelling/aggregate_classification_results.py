#!/usr/bin/env python3
"""
Aggregate Classification Results

This script aggregates the raw fold-level classification results from Morgan fingerprints
into a format expected by the dashboard with averaged metrics and status information.

Date: 2025-01-30
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_modelling/aggregation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def aggregate_classification_results(input_path: str, output_path: str):
    """
    Aggregate raw fold-level classification results into dashboard format
    
    Args:
        input_path: Path to raw classification results CSV
        output_path: Path to save aggregated results
    """
    try:
        # Load raw results
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} fold-level results from {input_path}")
        
        # Group by protein and aggregate
        aggregated_results = []
        
        for protein in df['protein'].unique():
            protein_data = df[df['protein'] == protein]
            
            # Check if protein has any valid results
            valid_folds = protein_data.dropna(subset=['accuracy', 'precision', 'recall', 'f1_score'])
            
            if len(valid_folds) == 0:
                # No valid results - mark as failed
                result = {
                    'protein': protein,
                    'n_samples': protein_data['n_samples'].iloc[0] if len(protein_data) > 0 else 0,
                    'status': 'failed',
                    'cv_results': '[]',
                    'avg_accuracy': np.nan,
                    'avg_precision': np.nan,
                    'avg_recall': np.nan,
                    'avg_f1': np.nan,
                    'avg_auc': np.nan
                }
            else:
                # Calculate averages across folds
                avg_accuracy = valid_folds['accuracy'].mean()
                avg_precision = valid_folds['precision'].mean()
                avg_recall = valid_folds['recall'].mean()
                avg_f1 = valid_folds['f1_score'].mean()
                
                # Create CV results as JSON string
                cv_results = []
                for _, fold_data in valid_folds.iterrows():
                    fold_result = {
                        'fold': int(fold_data['fold']),
                        'accuracy': fold_data['accuracy'],
                        'precision': fold_data['precision'],
                        'recall': fold_data['recall'],
                        'f1': fold_data['f1_score'],
                        'auc': np.nan,  # AUC not available in Morgan results
                        'n_train': fold_data['n_train'],
                        'n_test': fold_data['n_test'],
                        'n_train_active': fold_data['n_train_active'],
                        'n_train_inactive': fold_data['n_train_inactive'],
                        'n_test_active': fold_data['n_test_active'],
                        'n_test_inactive': fold_data['n_test_inactive'],
                        'confusion_matrix': [[0, 0], [0, 0]]  # Placeholder
                    }
                    cv_results.append(fold_result)
                
                result = {
                    'protein': protein,
                    'n_samples': protein_data['n_samples'].iloc[0],
                    'status': 'success',
                    'cv_results': json.dumps(cv_results),
                    'avg_accuracy': avg_accuracy,
                    'avg_precision': avg_precision,
                    'avg_recall': avg_recall,
                    'avg_f1': avg_f1,
                    'avg_auc': np.nan  # AUC not available in Morgan results
                }
            
            aggregated_results.append(result)
        
        # Create DataFrame and save
        aggregated_df = pd.DataFrame(aggregated_results)
        aggregated_df.to_csv(output_path, index=False)
        
        # Log summary
        successful = len(aggregated_df[aggregated_df['status'] == 'success'])
        failed = len(aggregated_df[aggregated_df['status'] == 'failed'])
        
        logger.info(f"Aggregation complete:")
        logger.info(f"  Total proteins: {len(aggregated_df)}")
        logger.info(f"  Successful models: {successful}")
        logger.info(f"  Failed models: {failed}")
        logger.info(f"  Results saved to: {output_path}")
        
        return aggregated_df
        
    except Exception as e:
        logger.error(f"Error aggregating results: {e}")
        raise

def main():
    """Main function to run aggregation"""
    input_path = "/home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_modelling/classification_results.csv"
    output_path = "/home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_modelling/aggregated_classification_results.csv"
    
    logger.info("Starting classification results aggregation...")
    
    # Check if input file exists
    if not Path(input_path).exists():
        logger.error(f"Input file not found: {input_path}")
        return
    
    # Aggregate results
    aggregated_df = aggregate_classification_results(input_path, output_path)
    
    # Print summary statistics
    if len(aggregated_df) > 0:
        successful_df = aggregated_df[aggregated_df['status'] == 'success']
        
        if len(successful_df) > 0:
            logger.info("\nPerformance Summary (successful models):")
            logger.info(f"  Average Accuracy: {successful_df['avg_accuracy'].mean():.3f}")
            logger.info(f"  Average Precision: {successful_df['avg_precision'].mean():.3f}")
            logger.info(f"  Average Recall: {successful_df['avg_recall'].mean():.3f}")
            logger.info(f"  Average F1-Score: {successful_df['avg_f1'].mean():.3f}")
            
            # Top performing proteins
            top_proteins = successful_df.nlargest(5, 'avg_f1')[['protein', 'avg_accuracy', 'avg_precision', 'avg_recall', 'avg_f1']]
            logger.info("\nTop 5 performing proteins (by F1-score):")
            for _, protein in top_proteins.iterrows():
                logger.info(f"  {protein['protein']}: F1={protein['avg_f1']:.3f}, Accuracy={protein['avg_accuracy']:.3f}")

if __name__ == "__main__":
    main() 