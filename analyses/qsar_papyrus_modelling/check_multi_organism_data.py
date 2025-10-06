#!/usr/bin/env python3
"""
Script to check for proteins with bioactivity data from multiple organisms
"""

import pandas as pd
from papyrus_scripts import PapyrusDataset
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_modelling/multi_organism_check.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_multi_organism_data():
    """Check all proteins for multi-organism bioactivity data"""
    
    # Load protein data
    data_path = "/home/serramelendezcsm/RA/Avoidome/processed_data/papyrus_protein_check_results.csv"
    proteins_df = pd.read_csv(data_path)
    
    logger.info(f"Checking {len(proteins_df)} proteins for multi-organism data...")
    
    # Initialize Papyrus dataset
    logger.info("Initializing Papyrus dataset...")
    papyrus_data = PapyrusDataset(version='latest', plusplus=True)
    logger.info("Papyrus dataset initialized successfully")
    
    # Load full dataset into DataFrame for efficient filtering
    logger.info("Loading full Papyrus dataset into DataFrame...")
    papyrus_df = papyrus_data.to_dataframe()
    logger.info(f"Loaded {len(papyrus_df)} total activities from Papyrus")
    
    # Results storage
    multi_organism_proteins = []
    single_organism_proteins = []
    no_data_proteins = []
    
    for idx, row in proteins_df.iterrows():
        protein_name = row['name2_entry']
        human_id = row['human_uniprot_id']
        mouse_id = row['mouse_uniprot_id']
        rat_id = row['rat_uniprot_id']
        
        logger.info(f"Checking protein {idx+1}/{len(proteins_df)}: {protein_name}")
        
        # Collect data for each organism
        organism_data = {}
        
        for organism, uniprot_id in [('human', human_id), ('mouse', mouse_id), ('rat', rat_id)]:
            if pd.notna(uniprot_id) and uniprot_id:
                try:
                    # Filter bioactivity data for the specific protein using pandas
                    activities_df = papyrus_df[papyrus_df['accession'] == uniprot_id]
                    organism_data[organism] = len(activities_df)
                    logger.info(f"  {organism.capitalize()} ({uniprot_id}): {len(activities_df)} activities")
                except Exception as e:
                    organism_data[organism] = 0
                    logger.error(f"  {organism.capitalize()} ({uniprot_id}): Error - {e}")
            else:
                organism_data[organism] = 0
                logger.info(f"  {organism.capitalize()}: No UniProt ID provided")
        
        # Analyze results
        total_activities = sum(organism_data.values())
        organisms_with_data = [org for org, count in organism_data.items() if count > 0]
        
        result = {
            'protein_name': protein_name,
            'human_id': human_id,
            'mouse_id': mouse_id,
            'rat_id': rat_id,
            'human_activities': organism_data.get('human', 0),
            'mouse_activities': organism_data.get('mouse', 0),
            'rat_activities': organism_data.get('rat', 0),
            'total_activities': total_activities,
            'organisms_with_data': organisms_with_data,
            'num_organisms': len(organisms_with_data)
        }
        
        if total_activities == 0:
            no_data_proteins.append(result)
            logger.warning(f"  Result: NO DATA")
        elif len(organisms_with_data) > 1:
            multi_organism_proteins.append(result)
            logger.info(f"  Result: MULTI-ORGANISM ({len(organisms_with_data)} organisms)")
        else:
            single_organism_proteins.append(result)
            logger.info(f"  Result: SINGLE-ORGANISM ({organisms_with_data[0]})")
    
    # Create summary
    logger.info("\n" + "="*50)
    logger.info("SUMMARY")
    logger.info("="*50)
    
    logger.info(f"Total proteins checked: {len(proteins_df)}")
    logger.info(f"Proteins with multi-organism data: {len(multi_organism_proteins)}")
    logger.info(f"Proteins with single-organism data: {len(single_organism_proteins)}")
    logger.info(f"Proteins with no data: {len(no_data_proteins)}")
    
    # Save detailed results
    results_df = pd.DataFrame(multi_organism_proteins + single_organism_proteins + no_data_proteins)
    output_path = "/home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_modelling/multi_organism_results.csv"
    results_df.to_csv(output_path, index=False)
    logger.info(f"Detailed results saved to: {output_path}")
    
    # Show multi-organism proteins
    if multi_organism_proteins:
        logger.info("\nMULTI-ORGANISM PROTEINS:")
        logger.info("-" * 30)
        for protein in multi_organism_proteins:
            logger.info(f"{protein['protein_name']}: {protein['organisms_with_data']} "
                       f"(H:{protein['human_activities']}, M:{protein['mouse_activities']}, R:{protein['rat_activities']})")
    else:
        logger.info("\nNo proteins found with multi-organism data.")
    
    # Show top single-organism proteins
    if single_organism_proteins:
        logger.info("\nTOP SINGLE-ORGANISM PROTEINS (by activity count):")
        logger.info("-" * 40)
        sorted_single = sorted(single_organism_proteins, key=lambda x: x['total_activities'], reverse=True)
        for protein in sorted_single[:10]:
            logger.info(f"{protein['protein_name']}: {protein['organisms_with_data'][0]} "
                       f"({protein['total_activities']} activities)")
    
    return results_df

if __name__ == "__main__":
    check_multi_organism_data() 