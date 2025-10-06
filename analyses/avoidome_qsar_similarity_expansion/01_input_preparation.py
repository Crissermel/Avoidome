#!/usr/bin/env python3
"""
AQSE Workflow - Step 1: Input Preparation

This script prepares the input data for the Avoidome QSAR Similarity Expansion (AQSE) workflow.
It loads avoidome protein sequences and prepares them for BLAST similarity search against Papyrus.

Author: AQSE Pipeline
Date: 2025
"""

import pandas as pd
import os
from pathlib import Path
import logging
from typing import List, Dict, Tuple
import requests
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AvoidomeInputPreparation:
    """Handles input preparation for AQSE workflow"""
    
    def __init__(self, avoidome_file: str, output_dir: str):
        """
        Initialize the input preparation class
        
        Args:
            avoidome_file: Path to avoidome_prot_list.csv
            output_dir: Output directory for prepared files
        """
        self.avoidome_file = avoidome_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.fasta_dir = self.output_dir / "fasta_files"
        self.fasta_dir.mkdir(exist_ok=True)
        
        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        
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
    
    def fetch_protein_sequences(self, uniprot_ids: List[str]) -> Dict[str, str]:
        """
        Fetch protein sequences from UniProt
        
        Args:
            uniprot_ids: List of UniProt IDs
            
        Returns:
            Dictionary mapping UniProt ID to sequence
        """
        logger.info(f"Fetching sequences for {len(uniprot_ids)} proteins")
        
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
                        logger.info(f"Fetched sequence for {uniprot_id} ({len(sequence)} aa)")
                    else:
                        logger.warning(f"No sequence found for {uniprot_id}")
                        failed_ids.append(uniprot_id)
                else:
                    logger.warning(f"Failed to fetch {uniprot_id}: HTTP {response.status_code}")
                    failed_ids.append(uniprot_id)
                    
            except Exception as e:
                logger.warning(f"Error fetching {uniprot_id}: {e}")
                failed_ids.append(uniprot_id)
            
            # Add delay to avoid overwhelming UniProt
            if i % 10 == 0 and i > 0:
                import time
                time.sleep(1)
        
        logger.info(f"Successfully fetched {len(sequences)} sequences")
        logger.info(f"Failed to fetch {len(failed_ids)} sequences: {failed_ids}")
        
        return sequences, failed_ids
    
    def create_fasta_files(self, df: pd.DataFrame, sequences: Dict[str, str]) -> List[str]:
        """
        Create FASTA files for BLAST search
        
        Args:
            df: Avoidome proteins DataFrame
            sequences: Dictionary of UniProt ID to sequence
            
        Returns:
            List of created FASTA file paths
        """
        logger.info("Creating FASTA files for BLAST search")
        
        fasta_files = []
        
        # Create individual FASTA files for each protein
        for _, row in df.iterrows():
            uniprot_id = row['UniProt ID']
            protein_name = row['Name']
            
            if uniprot_id in sequences:
                # Create FASTA record
                seq_record = SeqRecord(
                    Seq(sequences[uniprot_id]),
                    id=uniprot_id,
                    description=f"{protein_name} - {row.get('Alternative Names', '')}"
                )
                
                # Write to file
                fasta_file = self.fasta_dir / f"{uniprot_id}_{protein_name}.fasta"
                with open(fasta_file, 'w') as f:
                    SeqIO.write(seq_record, f, 'fasta')
                
                fasta_files.append(str(fasta_file))
                logger.info(f"Created FASTA file: {fasta_file}")
        
        # Create combined FASTA file for all proteins
        combined_fasta = self.output_dir / "avoidome_proteins_combined.fasta"
        with open(combined_fasta, 'w') as f:
            for _, row in df.iterrows():
                uniprot_id = row['UniProt ID']
                protein_name = row['Name']
                
                if uniprot_id in sequences:
                    seq_record = SeqRecord(
                        Seq(sequences[uniprot_id]),
                        id=uniprot_id,
                        description=f"{protein_name} - {row.get('Alternative Names', '')}"
                    )
                    SeqIO.write(seq_record, f, 'fasta')
        
        fasta_files.append(str(combined_fasta))
        logger.info(f"Created combined FASTA file: {combined_fasta}")
        
        return fasta_files
    
    def create_blast_config(self, fasta_files: List[str]) -> str:
        """
        Create BLAST configuration file
        
        Args:
            fasta_files: List of FASTA file paths
            
        Returns:
            Path to BLAST config file
        """
        config_file = self.output_dir / "blast_config.txt"
        
        config_content = f"""# AQSE BLAST Configuration
# Generated for Avoidome QSAR Similarity Expansion

# Input files
FASTA_FILES = {', '.join(fasta_files)}

# BLAST parameters
BLAST_TYPE = blastp
E_VALUE = 1e-5
MAX_TARGET_SEQS = 1000
OUTPUT_FORMAT = 6

# Similarity thresholds
HIGH_SIMILARITY = 70
MEDIUM_SIMILARITY = 50
LOW_SIMILARITY = 30

# Output columns
OUTPUT_COLUMNS = qseqid,sseqid,pident,length,mismatch,gapopen,qstart,qend,sstart,send,evalue,bitscore

# Papyrus database path (to be set)
PAPYRUS_DB_PATH = /path/to/papyrus_protein_db
"""
        
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        logger.info(f"Created BLAST config file: {config_file}")
        return str(config_file)
    
    def save_summary(self, df: pd.DataFrame, sequences: Dict[str, str], failed_ids: List[str]) -> str:
        """
        Save input preparation summary
        
        Args:
            df: Avoidome proteins DataFrame
            sequences: Dictionary of sequences
            failed_ids: List of failed UniProt IDs
            
        Returns:
            Path to summary file
        """
        summary_file = self.output_dir / "input_preparation_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("AQSE Input Preparation Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total avoidome proteins: {len(df)}\n")
            f.write(f"Proteins with sequences: {len(sequences)}\n")
            f.write(f"Failed to fetch: {len(failed_ids)}\n\n")
            
            f.write("Proteins with sequences:\n")
            f.write("-" * 30 + "\n")
            for uniprot_id, seq in sequences.items():
                protein_name = df[df['UniProt ID'] == uniprot_id]['Name'].iloc[0]
                f.write(f"{uniprot_id} ({protein_name}): {len(seq)} aa\n")
            
            if failed_ids:
                f.write(f"\nFailed to fetch sequences:\n")
                f.write("-" * 30 + "\n")
                for uniprot_id in failed_ids:
                    f.write(f"{uniprot_id}\n")
        
        logger.info(f"Saved summary to: {summary_file}")
        return str(summary_file)
    
    def run_preparation(self) -> Dict[str, str]:
        """
        Run the complete input preparation workflow
        
        Returns:
            Dictionary with output file paths
        """
        logger.info("Starting AQSE input preparation")
        
        # Load avoidome proteins
        df = self.load_avoidome_proteins()
        
        # Get UniProt IDs
        uniprot_ids = df['UniProt ID'].tolist()
        
        # Fetch sequences
        sequences, failed_ids = self.fetch_protein_sequences(uniprot_ids)
        
        # Create FASTA files
        fasta_files = self.create_fasta_files(df, sequences)
        
        # Create BLAST config
        blast_config = self.create_blast_config(fasta_files)
        
        # Save summary
        summary_file = self.save_summary(df, sequences, failed_ids)
        
        # Save sequences to CSV for easy access
        sequences_df = pd.DataFrame([
            {
                'uniprot_id': uniprot_id,
                'protein_name': df[df['UniProt ID'] == uniprot_id]['Name'].iloc[0],
                'sequence': seq,
                'sequence_length': len(seq)
            }
            for uniprot_id, seq in sequences.items()
        ])
        
        sequences_csv = self.output_dir / "avoidome_sequences.csv"
        sequences_df.to_csv(sequences_csv, index=False)
        
        logger.info("Input preparation completed successfully")
        
        return {
            'sequences_csv': str(sequences_csv),
            'blast_config': blast_config,
            'summary_file': summary_file,
            'fasta_files': fasta_files,
            'failed_ids': failed_ids
        }

def main():
    """Main function to run input preparation"""
    
    # Set up paths
    avoidome_file = "/home/serramelendezcsm/RA/Avoidome/primary_data/avoidome_prot_list.csv"
    output_dir = "/home/serramelendezcsm/RA/Avoidome/analyses/avoidome_qsar_similarity_expansion"
    
    # Initialize preparation class
    preparer = AvoidomeInputPreparation(avoidome_file, output_dir)
    
    # Run preparation
    results = preparer.run_preparation()
    
    print("\n" + "="*60)
    print("AQSE INPUT PREPARATION COMPLETED")
    print("="*60)
    print(f"Sequences CSV: {results['sequences_csv']}")
    print(f"BLAST Config: {results['blast_config']}")
    print(f"Summary: {results['summary_file']}")
    print(f"FASTA files: {len(results['fasta_files'])} created")
    print(f"Failed sequences: {len(results['failed_ids'])}")
    print("="*60)

if __name__ == "__main__":
    main()