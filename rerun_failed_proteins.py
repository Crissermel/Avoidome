#!/usr/bin/env python3
"""
Script to rerun QSAR models for the 3 proteins that failed due to technical errors
Run this script in the esmc conda environment: conda activate esmc
"""

import sys
import os
sys.path.append('/home/serramelendezcsm/RA/Avoidome/analyses/standardized_qsar_models')

from esm_morgan_qsar_modeling import ESMMorganQSARModel
from morgan_qsar_modeling import MorganQSARModel
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/serramelendezcsm/RA/Avoidome/rerun_failed_proteins.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Rerun models for the 3 failed proteins"""
    
    # Failed proteins that had technical errors
    failed_proteins = [
        ('CYP2B6', 'P20813', 'human'),
        ('CYP2C19', 'P33261', 'human'), 
        ('SLCO2B1', 'O94956', 'human')  # Corrected UniProt ID
    ]
    
    logger.info(f"Rerunning models for {len(failed_proteins)} failed proteins")
    logger.info("Make sure you are in the 'esmc' conda environment: conda activate esmc")
    
    # Initialize ESM+Morgan model
    logger.info("Initializing ESM+Morgan model...")
    esm_model = ESMMorganQSARModel()
    esm_model.load_data()
    
    # Initialize Morgan model  
    logger.info("Initializing Morgan model...")
    morgan_model = MorganQSARModel()
    morgan_model.load_data()
    
    for protein_name, uniprot_id, organism in failed_proteins:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {protein_name} ({uniprot_id}) - {organism}")
        logger.info(f"{'='*60}")
        
        try:
            # Get protein activities
            activities = esm_model.get_protein_activities(uniprot_id)
            if activities is None or len(activities) == 0:
                logger.warning(f"No activities found for {protein_name}")
                continue
                
            logger.info(f"Found {len(activities)} activities for {protein_name}")
            
            # Get protein sequence
            sequence = esm_model.get_protein_sequence(uniprot_id)
            if sequence is None:
                logger.warning(f"No sequence found for {protein_name}")
                continue
                
            logger.info(f"Sequence length: {len(sequence)}")
            
            # Train ESM+Morgan models
            logger.info(f"Training ESM+Morgan models for {protein_name}")
            esm_result = esm_model.process_organism(protein_name, organism, uniprot_id)
            
            if esm_result and esm_result.get('status') == 'completed':
                logger.info(f"✓ ESM+Morgan models completed for {protein_name}")
                logger.info(f"  - Regression R²: {esm_result.get('regression_r2', 'N/A'):.3f}")
                logger.info(f"  - Classification Accuracy: {esm_result.get('classification_accuracy', 'N/A'):.3f}")
            else:
                logger.error(f"✗ ESM+Morgan models failed for {protein_name}")
                if esm_result:
                    logger.error(f"Error: {esm_result.get('error', 'Unknown error')}")
            
            # Train Morgan models
            logger.info(f"Training Morgan models for {protein_name}")
            morgan_result = morgan_model.process_organism(protein_name, organism, uniprot_id)
            
            if morgan_result and morgan_result.get('status') == 'completed':
                logger.info(f"✓ Morgan models completed for {protein_name}")
                logger.info(f"  - Regression R²: {morgan_result.get('regression_r2', 'N/A'):.3f}")
                logger.info(f"  - Classification Accuracy: {morgan_result.get('classification_accuracy', 'N/A'):.3f}")
            else:
                logger.error(f"✗ Morgan models failed for {protein_name}")
                if morgan_result:
                    logger.error(f"Error: {morgan_result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Error processing {protein_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    logger.info(f"\n{'='*60}")
    logger.info("Completed rerunning failed proteins")
    logger.info(f"{'='*60}")

if __name__ == "__main__":
    main()
