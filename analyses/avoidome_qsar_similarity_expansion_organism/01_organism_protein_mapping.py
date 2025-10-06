#!/usr/bin/env python3
"""
AQSE-org Workflow - Step 1: Organism-Specific Protein Mapping

This script maps avoidome proteins to their organism-specific UniProt IDs
(human, mouse, rat) and prepares them for the AQSE-org workflow.

Author: AQSE-org Pipeline
Date: 2025
"""

import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import requests
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OrganismProteinMapping:
    """Handles organism-specific protein mapping for AQSE-org workflow"""
    
    def __init__(self, avoidome_file: str, organism_mapping_file: str, output_dir: str):
        """
        Initialize the organism protein mapping class
        
        Args:
            avoidome_file: Path to avoidome_prot_list.csv
            organism_mapping_file: Path to prot_orgs_extended.csv
            output_dir: Output directory for prepared files
        """
        self.avoidome_file = avoidome_file
        self.organism_mapping_file = organism_mapping_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.fasta_dir = self.output_dir / "fasta_files"
        self.fasta_dir.mkdir(exist_ok=True)
        
        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        
        # Organisms to process
        self.organisms = ['human', 'mouse', 'rat']
        self.organism_columns = {
            'human': 'human_uniprot_id',
            'mouse': 'mouse_uniprot_id', 
            'rat': 'rat_uniprot_id'
        }
    
    def load_avoidome_proteins(self) -> pd.DataFrame:
        """
        Load and clean avoidome protein list
        
        Returns:
            DataFrame with avoidome proteins
        """
        logger.info(f"Loading avoidome proteins from {self.avoidome_file}")
        
        try:
            df = pd.read_csv(self.avoidome_file)
            logger.info(f"Loaded {len(df)} avoidome proteins")
            
            # Clean the data
            df = df.dropna(subset=['UniProt ID'])
            df = df[df['UniProt ID'] != 'no info']
            
            # Remove duplicates based on UniProt ID
            df = df.drop_duplicates(subset=['UniProt ID'])
            
            logger.info(f"After cleaning: {len(df)} unique proteins with UniProt IDs")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading avoidome proteins: {e}")
            raise
    
    def load_organism_mapping(self) -> pd.DataFrame:
        """
        Load organism-specific protein mapping
        
        Returns:
            DataFrame with organism mappings
        """
        logger.info(f"Loading organism mapping from {self.organism_mapping_file}")
        
        try:
            df = pd.read_csv(self.organism_mapping_file)
            logger.info(f"Loaded {len(df)} organism mappings")
            
            # Clean the data - remove rows where all organism IDs are NaN
            organism_cols = list(self.organism_columns.values())
            df = df.dropna(subset=organism_cols, how='all')
            
            logger.info(f"After cleaning: {len(df)} proteins with at least one organism mapping")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading organism mapping: {e}")
            raise
    
    def create_organism_protein_mapping(self, avoidome_df: pd.DataFrame, 
                                      organism_df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """
        Create mapping between avoidome proteins and organism-specific UniProt IDs
        
        Args:
            avoidome_df: Avoidome proteins DataFrame
            organism_df: Organism mapping DataFrame
            
        Returns:
            Dictionary mapping organism to list of protein mappings
        """
        logger.info("Creating organism-specific protein mappings")
        
        organism_mappings = {org: [] for org in self.organisms}
        
        # Create mapping based on Name_2 from avoidome and name2_entry from organism mapping
        for _, avoidome_row in avoidome_df.iterrows():
            avoidome_name = avoidome_row['Name_2']
            avoidome_uniprot = avoidome_row['UniProt ID']
            
            # Find matching entry in organism mapping
            matching_org_rows = organism_df[organism_df['name2_entry'] == avoidome_name]
            
            if not matching_org_rows.empty:
                org_row = matching_org_rows.iloc[0]  # Take first match
                
                for organism in self.organisms:
                    org_column = self.organism_columns[organism]
                    org_uniprot_id = org_row[org_column]
                    
                    if pd.notna(org_uniprot_id) and org_uniprot_id.strip():
                        protein_mapping = {
                            'avoidome_name': avoidome_name,
                            'avoidome_uniprot_id': avoidome_uniprot,
                            'organism': organism,
                            'organism_uniprot_id': org_uniprot_id.strip(),
                            'protein_name': avoidome_row['Name'],
                            'alternative_names': avoidome_row.get('Alternative Names', '')
                        }
                        organism_mappings[organism].append(protein_mapping)
                        logger.info(f"Mapped {avoidome_name} -> {organism}: {org_uniprot_id}")
            else:
                logger.warning(f"No organism mapping found for {avoidome_name}")
        
        # Log summary
        for organism in self.organisms:
            logger.info(f"{organism.capitalize()}: {len(organism_mappings[organism])} proteins mapped")
        
        return organism_mappings
    
    def fetch_protein_sequences(self, uniprot_ids: List[str], organism: str) -> Dict[str, str]:
        """
        Fetch protein sequences from UniProt for a specific organism
        
        Args:
            uniprot_ids: List of UniProt IDs
            organism: Organism name for logging
            
        Returns:
            Dictionary mapping UniProt ID to sequence
        """
        logger.info(f"Fetching sequences for {len(uniprot_ids)} {organism} proteins")
        
        sequences = {}
        failed_ids = []
        
        for i, uniprot_id in enumerate(uniprot_ids):
            try:
                # UniProt REST API
                url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
                response = requests.get(url, timeout=30)
                
                if response.status_code == 200:
                    # Parse FASTA
                    fasta_text = response.text
                    if fasta_text.startswith('>'):
                        # Extract sequence
                        lines = fasta_text.strip().split('\n')
                        sequence = ''.join(lines[1:])
                        sequences[uniprot_id] = sequence
                        logger.info(f"Fetched {organism} sequence for {uniprot_id} ({len(sequence)} aa)")
                    else:
                        logger.warning(f"No sequence found for {organism} {uniprot_id}")
                        failed_ids.append(uniprot_id)
                else:
                    logger.warning(f"Failed to fetch {organism} {uniprot_id}: HTTP {response.status_code}")
                    failed_ids.append(uniprot_id)
                    
            except Exception as e:
                logger.warning(f"Error fetching {organism} {uniprot_id}: {e}")
                failed_ids.append(uniprot_id)
            
            # Add delay to avoid overwhelming UniProt
            if i % 10 == 0 and i > 0:
                import time
                time.sleep(1)
        
        logger.info(f"Successfully fetched {len(sequences)} {organism} sequences")
        logger.info(f"Failed to fetch {len(failed_ids)} {organism} sequences: {failed_ids}")
        
        return sequences, failed_ids
    
    def create_fasta_files(self, organism_mappings: Dict[str, List[Dict]], 
                          sequences: Dict[str, Dict[str, str]]) -> Dict[str, List[str]]:
        """
        Create FASTA files for each organism
        
        Args:
            organism_mappings: Dictionary of organism mappings
            sequences: Dictionary of sequences by organism
            
        Returns:
            Dictionary of FASTA files by organism
        """
        logger.info("Creating FASTA files for each organism")
        
        all_fasta_files = {}
        
        for organism in self.organisms:
            if organism not in organism_mappings or not organism_mappings[organism]:
                logger.warning(f"No mappings for {organism}, skipping")
                continue
                
            organism_fasta_files = []
            
            # Create individual FASTA files for each protein
            for protein_mapping in organism_mappings[organism]:
                uniprot_id = protein_mapping['organism_uniprot_id']
                protein_name = protein_mapping['avoidome_name']
                
                if organism in sequences and uniprot_id in sequences[organism]:
                    # Create FASTA record
                    seq_record = SeqRecord(
                        Seq(sequences[organism][uniprot_id]),
                        id=uniprot_id,
                        description=f"{protein_name} ({organism}) - {protein_mapping['protein_name']}"
                    )
                    
                    # Write to file
                    fasta_file = self.fasta_dir / f"{organism}_{uniprot_id}_{protein_name}.fasta"
                    with open(fasta_file, 'w') as f:
                        SeqIO.write(seq_record, f, 'fasta')
                    
                    organism_fasta_files.append(str(fasta_file))
                    logger.info(f"Created {organism} FASTA file: {fasta_file}")
            
            # Create combined FASTA file for this organism
            if organism_fasta_files:
                combined_fasta = self.output_dir / f"{organism}_proteins_combined.fasta"
                with open(combined_fasta, 'w') as f:
                    for protein_mapping in organism_mappings[organism]:
                        uniprot_id = protein_mapping['organism_uniprot_id']
                        protein_name = protein_mapping['avoidome_name']
                        
                        if organism in sequences and uniprot_id in sequences[organism]:
                            seq_record = SeqRecord(
                                Seq(sequences[organism][uniprot_id]),
                                id=uniprot_id,
                                description=f"{protein_name} ({organism}) - {protein_mapping['protein_name']}"
                            )
                            SeqIO.write(seq_record, f, 'fasta')
                
                organism_fasta_files.append(str(combined_fasta))
                logger.info(f"Created combined {organism} FASTA file: {combined_fasta}")
            
            all_fasta_files[organism] = organism_fasta_files
        
        return all_fasta_files
    
    def save_organism_mappings(self, organism_mappings: Dict[str, List[Dict]]) -> str:
        """
        Save organism mappings to CSV files
        
        Args:
            organism_mappings: Dictionary of organism mappings
            
        Returns:
            Path to summary file
        """
        logger.info("Saving organism mappings to CSV files")
        
        summary_data = []
        
        for organism in self.organisms:
            if organism not in organism_mappings or not organism_mappings[organism]:
                continue
                
            # Create DataFrame for this organism
            org_df = pd.DataFrame(organism_mappings[organism])
            org_csv = self.output_dir / f"{organism}_protein_mappings.csv"
            org_df.to_csv(org_csv, index=False)
            logger.info(f"Saved {organism} mappings to {org_csv}")
            
            # Add to summary
            for mapping in organism_mappings[organism]:
                summary_data.append({
                    'organism': organism,
                    'avoidome_name': mapping['avoidome_name'],
                    'avoidome_uniprot_id': mapping['avoidome_uniprot_id'],
                    'organism_uniprot_id': mapping['organism_uniprot_id'],
                    'protein_name': mapping['protein_name']
                })
        
        # Create combined summary
        summary_df = pd.DataFrame(summary_data)
        summary_csv = self.output_dir / "organism_mappings_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        
        logger.info(f"Saved organism mappings summary to {summary_csv}")
        return str(summary_csv)
    
    def save_sequences(self, sequences: Dict[str, Dict[str, str]]) -> Dict[str, str]:
        """
        Save sequences to CSV files for each organism
        
        Args:
            sequences: Dictionary of sequences by organism
            
        Returns:
            Dictionary of sequence CSV files by organism
        """
        logger.info("Saving sequences to CSV files")
        
        sequence_files = {}
        
        for organism in self.organisms:
            if organism not in sequences or not sequences[organism]:
                continue
                
            # Create DataFrame for this organism
            seq_data = []
            for uniprot_id, sequence in sequences[organism].items():
                seq_data.append({
                    'uniprot_id': uniprot_id,
                    'organism': organism,
                    'sequence': sequence,
                    'sequence_length': len(sequence)
                })
            
            seq_df = pd.DataFrame(seq_data)
            seq_csv = self.output_dir / f"{organism}_sequences.csv"
            seq_df.to_csv(seq_csv, index=False)
            sequence_files[organism] = str(seq_csv)
            logger.info(f"Saved {organism} sequences to {seq_csv}")
        
        return sequence_files
    
    def run_organism_mapping(self) -> Dict[str, any]:
        """
        Run the complete organism-specific protein mapping workflow
        
        Returns:
            Dictionary with results
        """
        logger.info("Starting AQSE-org organism-specific protein mapping")
        
        # Load avoidome proteins
        avoidome_df = self.load_avoidome_proteins()
        
        # Load organism mapping
        organism_df = self.load_organism_mapping()
        
        # Create organism-specific mappings
        organism_mappings = self.create_organism_protein_mapping(avoidome_df, organism_df)
        
        # Fetch sequences for each organism
        sequences = {}
        failed_sequences = {}
        
        for organism in self.organisms:
            if organism in organism_mappings and organism_mappings[organism]:
                uniprot_ids = [mapping['organism_uniprot_id'] for mapping in organism_mappings[organism]]
                org_sequences, org_failed = self.fetch_protein_sequences(uniprot_ids, organism)
                sequences[organism] = org_sequences
                failed_sequences[organism] = org_failed
        
        # Create FASTA files
        fasta_files = self.create_fasta_files(organism_mappings, sequences)
        
        # Save mappings and sequences
        mappings_summary = self.save_organism_mappings(organism_mappings)
        sequence_files = self.save_sequences(sequences)
        
        # Create overall summary
        summary_file = self.output_dir / "organism_mapping_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("AQSE-org Organism-Specific Protein Mapping Summary\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Total avoidome proteins: {len(avoidome_df)}\n")
            f.write(f"Organisms processed: {', '.join(self.organisms)}\n\n")
            
            for organism in self.organisms:
                if organism in organism_mappings:
                    f.write(f"{organism.capitalize()}:\n")
                    f.write(f"  Proteins mapped: {len(organism_mappings[organism])}\n")
                    f.write(f"  Sequences fetched: {len(sequences.get(organism, {}))}\n")
                    f.write(f"  Failed sequences: {len(failed_sequences.get(organism, []))}\n")
                    f.write(f"  FASTA files: {len(fasta_files.get(organism, []))}\n\n")
        
        logger.info("Organism-specific protein mapping completed successfully")
        
        return {
            'organism_mappings': organism_mappings,
            'sequences': sequences,
            'failed_sequences': failed_sequences,
            'fasta_files': fasta_files,
            'mappings_summary': mappings_summary,
            'sequence_files': sequence_files,
            'summary_file': str(summary_file)
        }

def main():
    """Main function to run organism-specific protein mapping"""
    
    # Set up paths
    avoidome_file = "/home/serramelendezcsm/RA/Avoidome/primary_data/avoidome_prot_list.csv"
    organism_mapping_file = "/home/serramelendezcsm/RA/Avoidome/primary_data/prot_orgs_extended.csv"
    output_dir = "/home/serramelendezcsm/RA/Avoidome/analyses/avoidome_qsar_similarity_expansion_organism/01_organism_mapping"
    
    # Initialize mapping class
    mapper = OrganismProteinMapping(avoidome_file, organism_mapping_file, output_dir)
    
    # Run mapping
    results = mapper.run_organism_mapping()
    
    print("\n" + "="*60)
    print("AQSE-ORG ORGANISM-SPECIFIC PROTEIN MAPPING COMPLETED")
    print("="*60)
    print(f"Organisms processed: {', '.join(['human', 'mouse', 'rat'])}")
    for organism in ['human', 'mouse', 'rat']:
        if organism in results['organism_mappings']:
            print(f"{organism.capitalize()}: {len(results['organism_mappings'][organism])} proteins mapped")
    print(f"Summary file: {results['summary_file']}")
    print("="*60)

if __name__ == "__main__":
    main()
