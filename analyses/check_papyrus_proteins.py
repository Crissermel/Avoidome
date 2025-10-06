#!/usr/bin/env python3
"""
Script to check if UniProt ID column entries from avoidome_prot_list.csv exist in Papyrus database.

This script:
1. Loads the avoidome_prot_list.csv file
2. Extracts the UniProt ID column entries
3. Maps Name_2 entries to UniProt IDs using UniProt API
4. Queries the Papyrus database using papyrus-scripts
5. Reports which proteins are found and which are not
6. Saves results to a CSV file with additional "Human" column

Date: 2025
"""

import pandas as pd
import os
import sys
import requests
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

# Add the project root to the path to import custom modules
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_uniprot_id_from_entry_name(entry_name: str) -> Optional[str]:
    """
    Get UniProt ID from entry name using UniProt API.
    
    Args:
        entry_name: Entry name (e.g., 'CYP1A2_HUMAN', 'CYP1A2_MOUSE')
        
    Returns:
        UniProt ID if found, None otherwise
    """
    try:
        url = f"https://rest.uniprot.org/uniprotkb/search"
        
        # Extract organism from entry name
        if entry_name.endswith('_HUMAN'):
            organism = 'organism_id:9606'
        elif entry_name.endswith('_MOUSE'):
            organism = 'organism_id:10090'
        elif entry_name.endswith('_RAT'):
            organism = 'organism_id:10116'
        else:
            organism = 'organism_id:9606'  # Default to human
            
        # Extract protein name without organism suffix
        protein_name = entry_name.replace('_HUMAN', '').replace('_MOUSE', '').replace('_RAT', '')
        
        params = {
            "query": f"gene:{protein_name} AND {organism}",
            "format": "tsv",
            "fields": "accession,gene_names"
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        lines = response.text.strip().split('\n')
        if len(lines) > 1:
            # Parse the result line
            parts = lines[1].strip().split('\t')
            if len(parts) >= 2:
                uniprot_id = parts[0].strip()
                gene_names = parts[1].strip() if len(parts) > 1 else ""
                
                # Verify it's the correct gene
                if uniprot_id and uniprot_id != "Entry":
                    # Check if the gene name matches our protein
                    if protein_name.upper() in gene_names.upper() or protein_name.upper() in uniprot_id:
                        logger.info(f"Found UniProt ID {uniprot_id} for {entry_name}")
                        return uniprot_id
                    else:
                        logger.warning(f"Gene name mismatch for {entry_name}: expected {protein_name}, got {gene_names}")
                        return None
                else:
                    logger.warning(f"Empty or invalid result for {entry_name}")
                    return None
        else:
            logger.warning(f"No mapping found for {entry_name}")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error querying UniProt API for {entry_name}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error for {entry_name}: {e}")
        return None

def map_name2_to_multiple_organisms(csv_path: str) -> Dict[str, Dict[str, str]]:
    """
    Map Name_2 entries to UniProt IDs for multiple organisms (Human, Mouse, Rat).
    
    Args:
        csv_path: Path to the avoidome_prot_list.csv file
        
    Returns:
        Dictionary mapping Name_2 entries to UniProt IDs for different organisms
    """
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} entries from {csv_path}")
        
        # Extract Name_2 column and remove any NaN values
        name2_entries = df['Name_2'].dropna().unique().tolist()
        logger.info(f"Found {len(name2_entries)} unique Name_2 entries")
        
        # Map each Name_2 entry to UniProt IDs for different organisms
        organism_mapping = {}
        organisms = {
            'human': '_HUMAN',
            'mouse': '_MOUSE', 
            'rat': '_RAT'
        }
        
        for name2 in name2_entries:
            organism_mapping[name2] = {}
            
            for organism, suffix in organisms.items():
                entry_name = f"{name2}{suffix}"
                logger.info(f"Mapping {name2} -> {entry_name} ({organism})")
                
                uniprot_id = get_uniprot_id_from_entry_name(entry_name)
                if uniprot_id:
                    organism_mapping[name2][organism] = uniprot_id
                    logger.info(f"✓ Mapped {name2} -> {uniprot_id} ({organism})")
                else:
                    logger.warning(f"✗ No mapping found for {name2} ({organism})")
                    organism_mapping[name2][organism] = None
        
        # Count successful mappings
        total_mappings = len(name2_entries) * len(organisms)
        successful_mappings = sum(
            1 for entry in organism_mapping.values() 
            for uniprot_id in entry.values() 
            if uniprot_id is not None
        )
        
        logger.info(f"Successfully mapped {successful_mappings}/{total_mappings} organism entries")
        return organism_mapping
        
    except FileNotFoundError:
        logger.error(f"File not found: {csv_path}")
        raise
    except Exception as e:
        logger.error(f"Error mapping Name_2 to multiple organisms: {e}")
        raise

def check_proteins_in_papyrus(uniprot_ids: List[str]) -> Dict[str, bool]:
    """
    Check which UniProt IDs from the list exist in Papyrus database.
    
    Args:
        uniprot_ids: List of UniProt IDs to check
        
    Returns:
        Dictionary mapping UniProt IDs to boolean indicating if found in Papyrus
    """
    try:
        # Import papyrus-scripts
        from papyrus_scripts import PapyrusDataset
        
        logger.info("Initializing Papyrus dataset...")
        
        # Initialize Papyrus dataset (regular, not ++)
        papyrus_data = PapyrusDataset(version='latest', plusplus=True) # If ++ wanted, change to True
        
        # Get protein data - this returns a PapyrusProteinSet object
        protein_data = papyrus_data.proteins()
        
        logger.info("Papyrus dataset loaded successfully")
        
        # Convert PapyrusProteinSet to DataFrame for easier handling
        # We need to iterate through the protein data to get the actual data
        protein_df = None
        try:
            # Try to convert to DataFrame if possible
            protein_df = protein_data.to_dataframe()
            logger.info(f"Papyrus dataset contains {len(protein_df)} protein entries")
        except AttributeError:
            # If to_dataframe() doesn't exist, try to iterate
            protein_list = []
            for protein in protein_data:
                protein_list.append(protein)
            protein_df = pd.DataFrame(protein_list)
            logger.info(f"Papyrus dataset contains {len(protein_df)} protein entries")
        
        # Create sets for different types of identifiers
        papyrus_uniprot_ids = set()
        target_ids = set()
        
        # Extract UniProt IDs (remove the _WT suffix from target_id)
        if 'UniProtID' in protein_df.columns:
            papyrus_uniprot_ids.update(protein_df['UniProtID'].dropna().str.upper())
            logger.info(f"Found {len(papyrus_uniprot_ids)} unique UniProt IDs")
        
        if 'target_id' in protein_df.columns:
            # Extract UniProt IDs from target_id (format: P47747_WT)
            target_uniprot_ids = protein_df['target_id'].str.replace('_WT', '').str.upper()
            papyrus_uniprot_ids.update(target_uniprot_ids.dropna())
            logger.info(f"After adding target_id UniProt IDs: {len(papyrus_uniprot_ids)} unique UniProt IDs")
        
        # Check each UniProt ID from avoidome list
        results = {}
        for uniprot_id in uniprot_ids:
            # Convert to uppercase for case-insensitive comparison
            uniprot_id_upper = uniprot_id.upper()
            found = uniprot_id_upper in papyrus_uniprot_ids
            results[uniprot_id] = found
            
            if found:
                logger.info(f"✓ Found: {uniprot_id}")
            else:
                logger.warning(f"✗ Not found: {uniprot_id}")
        
        return results
        
    except ImportError:
        logger.error("papyrus-scripts package not found. Please install it with: pip install papyrus-scripts")
        raise
    except Exception as e:
        logger.error(f"Error checking proteins in Papyrus: {e}")
        raise

def save_results_with_multiple_organisms(results: Dict[str, bool], organism_mapping: Dict[str, Dict[str, str]], output_path: str):
    """
    Save the results to a CSV file with additional columns for multiple organisms.
    
    Args:
        results: Dictionary mapping UniProt IDs to boolean indicating if found
        organism_mapping: Dictionary mapping Name_2 entries to UniProt IDs for different organisms
        output_path: Path to save the results CSV file
    """
    # Create a reverse mapping from UniProt ID to Name_2 entry and organism
    uniprot_to_info = {}
    for name2, organisms in organism_mapping.items():
        for organism, uniprot_id in organisms.items():
            if uniprot_id is not None:
                uniprot_to_info[uniprot_id] = {'name2': name2, 'organism': organism}
    
    # Create DataFrame from results
    df_results = pd.DataFrame([
        {
            'name2_entry': uniprot_to_info.get(uniprot_id, {}).get('name2', ''),
            'organism': uniprot_to_info.get(uniprot_id, {}).get('organism', ''),
            'uniprot_id': uniprot_id, 
            'found_in_papyrus': found
        }
        for uniprot_id, found in results.items()
    ])
    
    # Create a consolidated DataFrame with one row per protein
    consolidated_data = []
    for name2, organisms in organism_mapping.items():
        row = {
            'name2_entry': name2,
            'human_uniprot_id': organisms.get('human'),
            'mouse_uniprot_id': organisms.get('mouse'),
            'rat_uniprot_id': organisms.get('rat')
        }
        
        # Add found_in_papyrus columns for each organism
        for organism, uniprot_id in organisms.items():
            if uniprot_id is not None:
                found = results.get(uniprot_id, False)
                row[f'{organism}_found_in_papyrus'] = found
            else:
                row[f'{organism}_found_in_papyrus'] = None
        
        consolidated_data.append(row)
    
    df_final = pd.DataFrame(consolidated_data)
    
    # Add summary statistics
    total_proteins = len(results)
    found_proteins = sum(results.values())
    not_found_proteins = total_proteins - found_proteins
    
    # Count successful mappings per organism
    human_mappings = sum(1 for entry in organism_mapping.values() if entry.get('human') is not None)
    mouse_mappings = sum(1 for entry in organism_mapping.values() if entry.get('mouse') is not None)
    rat_mappings = sum(1 for entry in organism_mapping.values() if entry.get('rat') is not None)
    
    # Save to CSV
    df_final.to_csv(output_path, index=False)
    
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"Summary: {found_proteins}/{total_proteins} mapped UniProt IDs found in Papyrus ({found_proteins/total_proteins*100:.1f}%)")
    logger.info(f"Organism mappings - Human: {human_mappings}, Mouse: {mouse_mappings}, Rat: {rat_mappings}")
    
    # Print summary
    print(f"\n=== SUMMARY ===")
    print(f"Total mapped UniProt IDs checked: {total_proteins}")
    print(f"Found in Papyrus: {found_proteins}")
    print(f"Not found in Papyrus: {not_found_proteins}")
    print(f"Success rate: {found_proteins/total_proteins*100:.1f}%")
    print(f"Organism mappings:")
    print(f"  - Human: {human_mappings}/{len(organism_mapping)}")
    print(f"  - Mouse: {mouse_mappings}/{len(organism_mapping)}")
    print(f"  - Rat: {rat_mappings}/{len(organism_mapping)}")
    
    # Print not found proteins
    if not_found_proteins > 0:
        print(f"\nMapped UniProt IDs NOT found in Papyrus:")
        for uniprot_id, found in results.items():
            if not found:
                info = uniprot_to_info.get(uniprot_id, {})
                name2 = info.get('name2', 'Unknown')
                organism = info.get('organism', 'Unknown')
                print(f"  - {name2} ({organism}) -> {uniprot_id}")

def main():
    """
    Main function to execute the protein checking workflow.
    """
    # Define file paths
    avoidome_csv_path = "/home/serramelendezcsm/RA/Avoidome/primary_data/avoidome_prot_list.csv"
    output_path = "/home/serramelendezcsm/RA/Avoidome/processed_data/papyrus_protein_check_results.csv"
    
    logger.info("Starting Papyrus UniProt ID check for avoidome proteins...")
    
    try:
        # Map Name_2 entries to UniProt IDs using UniProt API
        logger.info("Mapping Name_2 entries to UniProt IDs...")
        name2_mapping = map_name2_to_multiple_organisms(avoidome_csv_path)
        
        # Use only the mapped UniProt IDs from Name_2 entries
        mapped_uniprot_ids = [uniprot_id for entry_name in name2_mapping.keys() for uniprot_id in name2_mapping[entry_name].values() if uniprot_id is not None]
        logger.info(f"Using {len(mapped_uniprot_ids)} mapped UniProt IDs from Name_2 entries for Papyrus checking")
        
        # Check mapped UniProt IDs in Papyrus
        results = check_proteins_in_papyrus(mapped_uniprot_ids)
        
        # Save results with human mapping
        save_results_with_multiple_organisms(results, name2_mapping, output_path)
        
        logger.info("UniProt ID checking completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 