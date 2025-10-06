#!/usr/bin/env python3
"""
Test script for Morgan QSAR modeling with 30-sample threshold
"""

import sys
import os
sys.path.append('/home/serramelendezcsm/RA/Avoidome/analyses/standardized_qsar_models')

from morgan_qsar_modeling import MorganQSARModeler

def test_morgan_30():
    """Test Morgan QSAR modeling with 30-sample threshold"""
    print("Testing Morgan QSAR modeling with 30-sample threshold...")
    
    # Initialize modeler
    modeler = MorganQSARModeler()
    
    # Test with a protein that has between 30-50 samples
    # Let's try CYP2B6 which had 33 samples
    result = modeler.process_protein("CYP2B6")
    
    print(f"Test result for CYP2B6:")
    print(f"Status: {result['status']}")
    print(f"Reason: {result.get('reason', 'N/A')}")
    print(f"Samples: {result.get('n_samples', 'N/A')}")
    
    if result['status'] == 'completed':
        print(f"R²: {result.get('regression_r2', 'N/A'):.3f}")
        print(f"Accuracy: {result.get('classification_accuracy', 'N/A'):.3f}")
        print("SUCCESS: CYP2B6 now processes with 30-sample threshold!")
    else:
        print("FAILED: CYP2B6 still not processing")

if __name__ == "__main__":
    test_morgan_30()