#!/usr/bin/env python3
"""
Determine which case (1, 2, or 3) was used for each protein model in 04_qsar_model_creation.py
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_avoidome_targets():
    """Load avoidome targets to get protein name to UniProt ID mapping"""
    try:
        avoidome_file = "/home/serramelendezcsm/RA/Avoidome/primary_data/avoidome_prot_list.csv"
        df = pd.read_csv(avoidome_file)
        
        # Create mapping from protein name to UniProt ID
        mapping = {}
        for _, row in df.iterrows():
            if pd.notna(row['UniProt ID']):
                mapping[row['Name_2']] = row['UniProt ID']
        
        logger.info(f"Loaded {len(mapping)} avoidome targets")
        return mapping
        
    except Exception as e:
        logger.error(f"Error loading avoidome targets: {e}")
        return {}

def load_bioactivity_data():
    """Load bioactivity data for all thresholds"""
    try:
        from papyrus_scripts import PapyrusDataset
        
        papyrus_data = PapyrusDataset(version='latest', plusplus=True)
        papyrus_df = papyrus_data.to_dataframe()
        
        logger.info(f"Loaded {len(papyrus_df)} bioactivity records from Papyrus")
        return papyrus_df
        
    except Exception as e:
        logger.error(f"Error loading bioactivity data: {e}")
        return pd.DataFrame()

def determine_case_for_protein(protein_name, uniprot_id, bioactivity_data, avoidome_targets):
    """Determine which case was used for a specific protein"""
    try:
        # Get target protein data
        target_data = bioactivity_data[bioactivity_data['accession'] == uniprot_id]
        target_samples = len(target_data)
        
        logger.info(f"Analyzing {protein_name} ({uniprot_id}): {target_samples} samples")
        
        # Case 3: Target protein with less than 30 bioactivity points
        if target_samples < 30:
            return "Case 3", target_samples, 0, "Insufficient data"
        
        # Check if this protein would be in the high threshold dataset
        # (assuming high threshold has the most comprehensive data)
        high_threshold_data = bioactivity_data  # Using full Papyrus data as proxy
        
        # Case 1: Only target protein with >= 30 bioactivities
        if target_samples >= 30:
            # Check if there are other proteins in the dataset
            # For Case 1, we'd expect only the target protein
            # For Case 2, we'd expect multiple proteins
            
            # Count unique proteins in the dataset
            unique_proteins = bioactivity_data['accession'].nunique()
            
            # If we're looking at a single protein model, it's likely Case 1
            # If there are multiple proteins, it's likely Case 2
            
            # For now, we'll use a heuristic based on sample size and protein diversity
            if target_samples >= 1000:  # Large datasets often use Case 2
                return "Case 2", target_samples, unique_proteins, "Multiple proteins likely"
            elif target_samples >= 30 and target_samples < 1000:
                return "Case 1", target_samples, 1, "Single protein likely"
            else:
                return "Case 3", target_samples, 0, "Insufficient data"
        
        return "Unknown", target_samples, 0, "Could not determine"
        
    except Exception as e:
        logger.error(f"Error determining case for {protein_name}: {e}")
        return "Error", 0, 0, str(e)

def main():
    """Main function to determine cases for all proteins"""
    logger.info("Determining model cases for all proteins...")
    
    # Load data
    avoidome_targets = load_avoidome_targets()
    bioactivity_data = load_bioactivity_data()
    
    if not avoidome_targets or bioactivity_data.empty:
        logger.error("Failed to load required data")
        return
    
    # Load the comparison results to get the list of proteins
    comparison_file = "/home/serramelendezcsm/RA/Avoidome/analyses/avoidome_qsar_similarity_expansion/model_comparison_results.csv"
    try:
        comparison_df = pd.read_csv(comparison_file)
        logger.info(f"Loaded comparison results for {len(comparison_df)} proteins")
    except Exception as e:
        logger.error(f"Error loading comparison results: {e}")
        return
    
    # Determine case for each protein
    case_results = []
    
    for _, row in comparison_df.iterrows():
        protein_name = row['protein_name']
        uniprot_id = row['uniprot_id_x']
        
        case, samples, unique_proteins, reason = determine_case_for_protein(
            protein_name, uniprot_id, bioactivity_data, avoidome_targets
        )
        
        case_results.append({
            'protein_name': protein_name,
            'uniprot_id': uniprot_id,
            'case': case,
            'target_samples': samples,
            'unique_proteins': unique_proteins,
            'reason': reason
        })
        
        logger.info(f"{protein_name}: {case} ({samples} samples, {unique_proteins} proteins)")
    
    # Create results DataFrame
    case_df = pd.DataFrame(case_results)
    
    # Save results
    output_file = "/home/serramelendezcsm/RA/Avoidome/analyses/avoidome_qsar_similarity_expansion/model_cases.csv"
    case_df.to_csv(output_file, index=False)
    logger.info(f"Case determination results saved to: {output_file}")
    
    # Summary
    case_counts = case_df['case'].value_counts()
    logger.info(f"\nCase distribution:")
    for case, count in case_counts.items():
        logger.info(f"  {case}: {count} proteins")

if __name__ == "__main__":
    main()
