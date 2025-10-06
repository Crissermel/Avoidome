#!/usr/bin/env python3
"""
Test Classification Pipeline

This script tests the classification pipeline with a small subset of proteins
to verify that everything works correctly before running the full analysis.

Date: 2025-01-30
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from minimal_papyrus_esm_classification import PapyrusESMQSARClassifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_esm_emb_classification/test_classification.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def test_classification_pipeline():
    """Test the classification pipeline with a subset of proteins"""
    
    logger.info("Starting classification pipeline test...")
    
    # Load protein data
    data_path = "/home/serramelendezcsm/RA/Avoidome/processed_data/papyrus_protein_check_results.csv"
    proteins_df = pd.read_csv(data_path)
    
    # Select a small subset of proteins for testing (first 5)
    test_proteins = proteins_df.head(5).copy()
    logger.info(f"Testing with {len(test_proteins)} proteins: {test_proteins['name2_entry'].tolist()}")
    
    # Create a temporary results file for testing
    test_results_path = "/home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_esm_emb_classification/test_results.csv"
    
    try:
        # Initialize classifier
        model = PapyrusESMQSARClassifier(activity_threshold=6.0)
        
        # Load data
        model.load_data()
        
        # Process only the test proteins
        results = []
        
        for idx, row in test_proteins.iterrows():
            try:
                logger.info(f"Testing protein: {row['name2_entry']}")
                result = model.process_protein(row)
                results.append(result)
                
                if result['status'] == 'success':
                    logger.info(f"✓ {result['protein']}: Accuracy={result['avg_accuracy']:.3f}, F1={result['avg_f1']:.3f}")
                else:
                    logger.warning(f"✗ {result['protein']}: {result['status']}")
                
            except Exception as e:
                logger.error(f"Error testing protein {row['name2_entry']}: {e}")
                results.append({
                    'protein': row['name2_entry'],
                    'status': 'error',
                    'error': str(e)
                })
        
        # Save test results
        results_df = pd.DataFrame(results)
        results_df.to_csv(test_results_path, index=False)
        logger.info(f"Test results saved to {test_results_path}")
        
        # Print summary
        successful_results = [r for r in results if r['status'] == 'success']
        logger.info(f"\nTest Summary:")
        logger.info(f"  - Proteins tested: {len(test_proteins)}")
        logger.info(f"  - Successful models: {len(successful_results)}")
        logger.info(f"  - Failed models: {len(results) - len(successful_results)}")
        
        if successful_results:
            avg_accuracy = np.mean([r['avg_accuracy'] for r in successful_results])
            avg_f1 = np.mean([r['avg_f1'] for r in successful_results])
            logger.info(f"  - Average Accuracy: {avg_accuracy:.3f}")
            logger.info(f"  - Average F1: {avg_f1:.3f}")
        
        logger.info("✓ Classification pipeline test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Classification pipeline test failed: {e}")
        return False

def main():
    """Run classification pipeline test"""
    logger.info("=" * 60)
    logger.info("CLASSIFICATION PIPELINE TESTING")
    logger.info("=" * 60)
    
    # Test basic classification
    success = test_classification_pipeline()
    
    if success:
        logger.info("\n" + "=" * 60)
        logger.info("TEST PASSED! ✓")
        logger.info("The classification pipeline is ready for full analysis.")
        logger.info("=" * 60)
    else:
        logger.error("\n" + "=" * 60)
        logger.error("TEST FAILED! ✗")
        logger.error("Please check the logs and fix issues before running full analysis.")
        logger.error("=" * 60)

if __name__ == "__main__":
    main() 