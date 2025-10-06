#!/usr/bin/env python3
"""
Script to check for overlapping UniProt IDs between avoidome list and pcmol_targets.txt

This script compares UniProt IDs from:
- /home/serramelendezcsm/RA/Avoidome/primary_data/prot_orgs.csv (avoidome list)
- /home/serramelendezcsm/RA/Avoidome/analyses/run_esm_embeddings/pcmol_targets.txt

The script extracts all UniProt IDs from both files and finds matches.
"""

import pandas as pd
import os
from pathlib import Path

def load_avoidome_uniprot_ids(csv_file):
    """
    Load UniProt IDs from the avoidome CSV file.
    Returns a set of all unique UniProt IDs from human, mouse, and rat columns.
    """
    print(f"Loading avoidome data from: {csv_file}")
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Extract all UniProt IDs from the three columns
    uniprot_ids = set()
    
    # Human UniProt IDs
    human_ids = df['human_uniprot_id'].dropna().tolist()
    uniprot_ids.update(human_ids)
    
    # Mouse UniProt IDs  
    mouse_ids = df['mouse_uniprot_id'].dropna().tolist()
    uniprot_ids.update(mouse_ids)
    
    # Rat UniProt IDs
    rat_ids = df['rat_uniprot_id'].dropna().tolist()
    uniprot_ids.update(rat_ids)
    
    # Remove empty strings if any
    uniprot_ids = {id for id in uniprot_ids if id.strip()}
    
    print(f"Found {len(uniprot_ids)} unique UniProt IDs in avoidome list")
    print(f"  - Human: {len(human_ids)} IDs")
    print(f"  - Mouse: {len(mouse_ids)} IDs") 
    print(f"  - Rat: {len(rat_ids)} IDs")
    
    return uniprot_ids

def load_pcmol_uniprot_ids(txt_file):
    """
    Load UniProt IDs from the pcmol_targets.txt file.
    Returns a set of all unique UniProt IDs.
    """
    print(f"\nLoading pcmol targets from: {txt_file}")
    
    uniprot_ids = set()
    
    with open(txt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                uniprot_ids.add(line)
    
    print(f"Found {len(uniprot_ids)} unique UniProt IDs in pcmol_targets.txt")
    
    return uniprot_ids

def find_overlaps(avoidome_ids, pcmol_ids):
    """
    Find overlapping UniProt IDs between the two sets.
    """
    print(f"\nFinding overlaps...")
    
    # Find intersections
    overlaps = avoidome_ids.intersection(pcmol_ids)
    
    print(f"Found {len(overlaps)} overlapping UniProt IDs:")
    
    if overlaps:
        print("\nOverlapping IDs:")
        for i, uniprot_id in enumerate(sorted(overlaps), 1):
            print(f"  {i:3d}. {uniprot_id}")
    else:
        print("No overlapping UniProt IDs found.")
    
    return overlaps

def get_gene_info_for_overlaps(overlaps, csv_file):
    """
    Get gene information for overlapping UniProt IDs from the avoidome CSV.
    """
    if not overlaps:
        return
    
    print(f"\nGene information for overlapping UniProt IDs:")
    print("=" * 80)
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    for uniprot_id in sorted(overlaps):
        # Find rows where this UniProt ID appears
        mask = (df['human_uniprot_id'] == uniprot_id) | \
               (df['mouse_uniprot_id'] == uniprot_id) | \
               (df['rat_uniprot_id'] == uniprot_id)
        
        matching_rows = df[mask]
        
        for _, row in matching_rows.iterrows():
            print(f"\nUniProt ID: {uniprot_id}")
            print(f"  Gene name: {row['name2_entry']}")
            print(f"  Human UniProt: {row['human_uniprot_id']}")
            print(f"  Mouse UniProt: {row['mouse_uniprot_id']}")
            print(f"  Rat UniProt: {row['rat_uniprot_id']}")

def main():
    """
    Main function to run the comparison.
    """
    print("UniProt ID Overlap Checker")
    print("=" * 50)
    
    # File paths
    avoidome_file = "/home/serramelendezcsm/RA/Avoidome/primary_data/prot_orgs.csv"
    pcmol_file = "/home/serramelendezcsm/RA/Avoidome/analyses/run_esm_embeddings/pcmol_targets.txt"
    
    # Check if files exist
    if not os.path.exists(avoidome_file):
        print(f"Error: Avoidome file not found: {avoidome_file}")
        return
    
    if not os.path.exists(pcmol_file):
        print(f"Error: PCMol targets file not found: {pcmol_file}")
        return
    
    # Load UniProt IDs from both files
    avoidome_ids = load_avoidome_uniprot_ids(avoidome_file)
    pcmol_ids = load_pcmol_uniprot_ids(pcmol_file)
    
    # Find overlaps
    overlaps = find_overlaps(avoidome_ids, pcmol_ids)
    
    # Get gene information for overlaps
    if overlaps:
        get_gene_info_for_overlaps(overlaps, avoidome_file)
    
    # Summary
    print(f"\n" + "=" * 50)
    print("SUMMARY:")
    print(f"  Avoidome UniProt IDs: {len(avoidome_ids)}")
    print(f"  PCMol target IDs: {len(pcmol_ids)}")
    print(f"  Overlapping IDs: {len(overlaps)}")
    print(f"  Overlap percentage: {len(overlaps)/len(avoidome_ids)*100:.2f}% of avoidome IDs")

if __name__ == "__main__":
    main()





