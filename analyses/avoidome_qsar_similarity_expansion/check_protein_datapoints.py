#!/usr/bin/env python3
"""
Simple script to check how many datapoints a protein has in the Papyrus dataset.

Usage:
    python check_protein_datapoints.py <protein_id>

Example:
    python check_protein_datapoints.py P12345
"""

import sys
import argparse
from papyrus_scripts import PapyrusDataset


def check_protein_datapoints(protein_id):
    """
    Check how many datapoints a protein has in the Papyrus dataset.
    
    Args:
        protein_id (str): UniProt accession ID of the protein
        
    Returns:
        int: Number of datapoints for the protein
    """
    try:
        # Initialize Papyrus dataset
        print(f"Loading Papyrus dataset...")
        papyrus_data = PapyrusDataset(version='latest', plusplus=True)
        
        # Load data into DataFrame
        print(f"Loading bioactivity data...")
        bioactivity_data = papyrus_data.to_dataframe()
        
        # Filter data for the specific protein
        protein_data = bioactivity_data[bioactivity_data['accession'] == protein_id]
        datapoint_count = len(protein_data)
        
        return datapoint_count
        
    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    """Main function to handle command line arguments and execute the check."""
    parser = argparse.ArgumentParser(
        description="Check how many datapoints a protein has in the Papyrus dataset"
    )
    parser.add_argument(
        'protein_id', 
        help='UniProt accession ID of the protein (e.g., P12345)'
    )
    
    args = parser.parse_args()
    
    # Check datapoints for the protein
    datapoint_count = check_protein_datapoints(args.protein_id)
    
    if datapoint_count is not None:
        print(f"Protein {args.protein_id} has {datapoint_count} datapoints")
    else:
        print(f"Failed to check datapoints for protein {args.protein_id}")
        sys.exit(1)


if __name__ == "__main__":
    main()


