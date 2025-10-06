#!/usr/bin/env python3
"""
Test script for ESM+Morgan QSAR modeling

This script tests the ESM+Morgan QSAR modeling implementation with a single protein
to ensure everything works correctly before running the full dataset.
"""

import sys
import logging
from pathlib import Path

# Add the analyses directory to the path
sys.path.append('/home/serramelendezcsm/RA/Avoidome/analyses')

from esm_morgan_qsar_modeling import ESMMorganQSARModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_single_protein():
    """Test the ESM+Morgan QSAR modeling with a single protein"""
    logger.info("Testing ESM+Morgan QSAR modeling with single protein")
    
    # Initialize model
    model = ESMMorganQSARModel()
    
    # Load data
    model.load_data()
    
    # Test with CYP1A2 (should have good data and sequence)
    test_protein = model.proteins_df[model.proteins_df['name2_entry'] == 'CYP1A2'].iloc[0]
    
    logger.info(f"Testing with protein: {test_protein['name2_entry']}")
    
    # Process the protein
    result = model.process_protein(test_protein)
    
    logger.info("Test results:")
    logger.info(f"Status: {result['status']}")
    if result['status'] == 'completed':
        logger.info(f"R²: {result['regression_r2']:.3f}")
        logger.info(f"Accuracy: {result['classification_accuracy']:.3f}")
        logger.info(f"F1: {result['classification_f1']:.3f}")
        logger.info(f"AUC: {result['classification_auc']:.3f}")
    
    return result

if __name__ == "__main__":
    test_single_protein()