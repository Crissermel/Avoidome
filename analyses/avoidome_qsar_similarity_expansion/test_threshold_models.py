#!/usr/bin/env python3
"""
Test script for threshold-specific AQSE models

This script demonstrates how the new threshold-specific models work
by creating models for different similarity thresholds.
"""

import sys
sys.path.append('/home/serramelendezcsm/RA/Avoidome/analyses/avoidome_qsar_similarity_expansion')

from 04_2_qsar_model_creation import AQSEQSARModelCreation

def test_threshold_models():
    """Test threshold-specific model creation"""
    
    # Initialize AQSE workflow
    aqse = AQSEQSARModelCreation(
        output_dir="/home/serramelendezcsm/RA/Avoidome/analyses/avoidome_qsar_similarity_expansion/04_qsar_models_temp",
        avoidome_file="/home/serramelendezcsm/RA/Avoidome/primary_data/avoidome_prot_list.csv",
        similarity_file="/home/serramelendezcsm/RA/Avoidome/analyses/avoidome_qsar_similarity_expansion/02_similarity_search/similarity_search_summary.csv"
    )
    
    # Test with CYP2B6 (P20813) which has similar proteins at all thresholds
    target_name = "CYP2B6"
    uniprot_id = "P20813"
    
    print(f"Testing threshold-specific models for {target_name} ({uniprot_id})")
    print("="*80)
    
    # Check what similar proteins exist at each threshold
    for threshold in ['high', 'medium', 'low']:
        similar_proteins = aqse.get_similar_proteins_for_threshold(uniprot_id, threshold)
        print(f"{threshold.upper()} threshold: {len(similar_proteins)} similar proteins")
        if similar_proteins:
            print(f"  Proteins: {similar_proteins}")
        print()
    
    # Create models for each threshold
    for threshold in ['high', 'medium', 'low']:
        similar_proteins = aqse.get_similar_proteins_for_threshold(uniprot_id, threshold)
        if len(similar_proteins) > 0:
            print(f"Creating {threshold.upper()} threshold model...")
            results = aqse.create_qsar_model_with_threshold(target_name, uniprot_id, threshold)
            if results:
                print(f"✅ {threshold.upper()} model created successfully")
                print(f"   Test R²: {results['test_r2']:.3f}")
                print(f"   Q²: {results['q2']:.3f}")
                print(f"   Training samples: {results['n_train']}")
                print(f"   Test samples: {results['n_test']}")
            else:
                print(f"❌ {threshold.upper()} model failed")
            print()
        else:
            print(f"⚠️  No similar proteins at {threshold.upper()} threshold")
            print()

if __name__ == "__main__":
    test_threshold_models()
