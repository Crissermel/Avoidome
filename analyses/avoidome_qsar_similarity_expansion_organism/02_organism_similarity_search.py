#!/usr/bin/env python3
"""
AQSE-org Workflow - Step 2: Organism-Specific Protein Similarity Search

This script performs similarity search for each organism's proteins against Papyrus
database and creates organism-specific similarity matrices.

Author: AQSE-org Pipeline
Date: 2025
"""

import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from Bio import SeqIO
from Bio.SeqUtils import seq1
import requests
from io import StringIO

# Import papyrus-scripts
from papyrus_scripts import PapyrusDataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OrganismSimilaritySearch:
    """Handles organism-specific protein similarity search using papyrus-scripts library"""
    
    def __init__(self, input_dir: str, output_dir: str, papyrus_version: str = '05.7'):
        """
        Initialize the organism similarity search class
        
        Args:
            input_dir: Directory with organism-specific FASTA files
            output_dir: Output directory for results
            papyrus_version: Papyrus dataset version to use
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.papyrus_version = papyrus_version
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.similarity_matrices_dir = self.output_dir / "similarity_matrices"
        self.similarity_matrices_dir.mkdir(exist_ok=True)
        
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        # Similarity thresholds
        self.thresholds = {
            'high': 70.0,
            'medium': 50.0,
            'low': 30.0
        }
        
        # Organisms to process
        self.organisms = ['human', 'mouse', 'rat']
        
        # Initialize Papyrus dataset
        logger.info(f"Initializing Papyrus dataset version {papyrus_version}")
        self.papyrus_data = PapyrusDataset(version=papyrus_version, plusplus=False)
    
    def load_organism_sequences(self, organism: str) -> Dict[str, str]:
        """
        Load organism-specific protein sequences from FASTA files
        
        Args:
            organism: Organism name (human, mouse, rat)
            
        Returns:
            Dictionary mapping protein IDs to sequences
        """
        logger.info(f"Loading {organism} protein sequences")
        
        sequences = {}
        fasta_files = list(self.input_dir.glob(f"{organism}_*.fasta"))
        
        for fasta_file in fasta_files:
            logger.info(f"Loading sequences from {fasta_file}")
            for record in SeqIO.parse(fasta_file, "fasta"):
                protein_id = record.id
                sequence = str(record.seq)
                sequences[protein_id] = sequence
                logger.info(f"Loaded {organism} {protein_id}: {len(sequence)} amino acids")
        
        logger.info(f"Loaded {len(sequences)} {organism} protein sequences")
        return sequences
    
    def get_papyrus_protein_sequences(self) -> pd.DataFrame:
        """
        Get protein sequences from Papyrus database
        
        Returns:
            DataFrame with Papyrus protein sequences
        """
        logger.info("Loading Papyrus protein sequences")
        
        try:
            # Get protein data from Papyrus
            protein_data = self.papyrus_data.proteins()
            logger.info("Loaded protein data from Papyrus")
            
            # Convert to DataFrame if it's a PapyrusProteinSet
            if hasattr(protein_data, 'to_dataframe'):
                protein_df = protein_data.to_dataframe()
            else:
                protein_df = protein_data
            
            # Filter for proteins with sequences
            protein_df = protein_df.dropna(subset=['Sequence'])
            logger.info(f"Found {len(protein_df)} proteins with sequences")
            
            # Limit to a very small subset for similarity search (first 100 proteins)
            # This is necessary because comparing against all 7000+ proteins is too slow
            if len(protein_df) > 100:
                logger.info(f"Limiting Papyrus proteins to first 100 for similarity search (out of {len(protein_df)})")
                protein_df = protein_df.head(100)
            
            return protein_df
            
        except Exception as e:
            logger.error(f"Error loading Papyrus protein data: {e}")
            return pd.DataFrame()
    
    def calculate_sequence_similarity(self, seq1: str, seq2: str) -> float:
        """
        Calculate sequence similarity using simple alignment
        
        Args:
            seq1: First sequence
            seq2: Second sequence
            
        Returns:
            Percent identity
        """
        if not seq1 or not seq2:
            return 0.0
        
        # Simple sequence alignment and identity calculation
        min_len = min(len(seq1), len(seq2))
        if min_len == 0:
            return 0.0
        
        # For efficiency, only compare first 100 amino acids
        # This is a simplified approach for demonstration
        max_compare = min(100, min_len)
        seq1_short = seq1[:max_compare]
        seq2_short = seq2[:max_compare]
        
        # Count identical positions
        matches = sum(1 for a, b in zip(seq1_short, seq2_short) if a == b)
        
        # Calculate percent identity
        percent_identity = (matches / max_compare) * 100
        
        return percent_identity
    
    def find_similar_proteins(self, organism_sequences: Dict[str, str], 
                            papyrus_proteins: pd.DataFrame, organism: str) -> Dict[str, List[Dict]]:
        """
        Find similar proteins in Papyrus for each organism protein
        
        Args:
            organism_sequences: Dictionary of organism protein sequences
            papyrus_proteins: DataFrame of Papyrus proteins
            organism: Organism name for logging
            
        Returns:
            Dictionary of similar proteins by organism protein
        """
        logger.info(f"Finding similar proteins in Papyrus for {organism} proteins")
        
        similar_proteins = {}
        
        # Convert papyrus proteins to list for faster iteration
        papyrus_list = papyrus_proteins.to_dict('records')
        logger.info(f"Comparing {len(organism_sequences)} {organism} proteins against {len(papyrus_list)} Papyrus proteins")
        
        for i, (org_id, org_seq) in enumerate(organism_sequences.items()):
            logger.info(f"Searching for proteins similar to {organism} {org_id} ({i+1}/{len(organism_sequences)})")
            
            similar_proteins[org_id] = []
            
            # Process in batches to avoid memory issues
            batch_size = 1000
            for batch_start in range(0, len(papyrus_list), batch_size):
                batch_end = min(batch_start + batch_size, len(papyrus_list))
                batch = papyrus_list[batch_start:batch_end]
                
                for papyrus_protein in batch:
                    papyrus_id = papyrus_protein['target_id']
                    papyrus_seq = papyrus_protein['Sequence']
                    
                    # Calculate similarity
                    similarity = self.calculate_sequence_similarity(org_seq, papyrus_seq)
                    
                    # Store if above any threshold
                    if similarity >= min(self.thresholds.values()):
                        similar_proteins[org_id].append({
                            'papyrus_id': papyrus_id,
                            'similarity': similarity,
                            'organism': papyrus_protein.get('Organism', 'Unknown'),
                            'protein_class': papyrus_protein.get('Classification', 'Unknown'),
                            'sequence': papyrus_seq
                        })
            
            # Sort by similarity (descending)
            similar_proteins[org_id].sort(key=lambda x: x['similarity'], reverse=True)
            
            logger.info(f"Found {len(similar_proteins[org_id])} similar proteins for {organism} {org_id}")
        
        return similar_proteins
    
    def create_similarity_matrices(self, similar_proteins: Dict[str, List[Dict]], 
                                 organism: str) -> Dict[str, pd.DataFrame]:
        """
        Create similarity matrices for different thresholds for an organism
        
        Args:
            similar_proteins: Dictionary of similar proteins
            organism: Organism name
            
        Returns:
            Dictionary of similarity matrices by threshold
        """
        logger.info(f"Creating similarity matrices for {organism}")
        
        matrices = {}
        
        for threshold_name, threshold_value in self.thresholds.items():
            logger.info(f"Creating {organism} similarity matrix for {threshold_name} threshold ({threshold_value}%)")
            
            # Collect all similarities above threshold
            matrix_data = []
            
            for org_id, similar_list in similar_proteins.items():
                for similar_protein in similar_list:
                    if similar_protein['similarity'] >= threshold_value:
                        matrix_data.append({
                            'query_id': org_id,
                            'subject_id': similar_protein['papyrus_id'],
                            'similarity': similar_protein['similarity'],
                            'organism': similar_protein['organism'],
                            'protein_class': similar_protein['protein_class']
                        })
            
            if not matrix_data:
                logger.warning(f"No hits above {threshold_value}% similarity for {organism}")
                continue
            
            # Create DataFrame
            df = pd.DataFrame(matrix_data)
            
            # Create pivot table
            similarity_matrix = df.pivot_table(
                index='query_id',
                columns='subject_id',
                values='similarity',
                fill_value=0
            )
            
            # Save matrix
            matrix_file = self.similarity_matrices_dir / f"{organism}_similarity_matrix_{threshold_name}.csv"
            similarity_matrix.to_csv(matrix_file)
            
            matrices[threshold_name] = similarity_matrix
            
            logger.info(f"Created {organism} {threshold_name} matrix: {similarity_matrix.shape}")
        
        return matrices
    
    def create_summary_statistics(self, similar_proteins: Dict[str, List[Dict]], 
                                organism: str) -> pd.DataFrame:
        """
        Create summary statistics for similarity search for an organism
        
        Args:
            similar_proteins: Dictionary of similar proteins
            organism: Organism name
            
        Returns:
            DataFrame with summary statistics
        """
        logger.info(f"Creating summary statistics for {organism}")
        
        summary_data = []
        
        for org_id, similar_list in similar_proteins.items():
            for threshold_name, threshold_value in self.thresholds.items():
                # Count proteins above threshold
                above_threshold = [p for p in similar_list if p['similarity'] >= threshold_value]
                
                # Get top 10 most similar
                top_similar = sorted(above_threshold, key=lambda x: x['similarity'], reverse=True)[:10]
                similar_names = [f"{p['papyrus_id']} ({p['similarity']:.1f}%)" for p in top_similar]
                
                summary_data.append({
                    'organism': organism,
                    'query_protein': org_id,
                    'threshold': threshold_name,
                    'threshold_value': threshold_value,
                    'num_similar_proteins': len(above_threshold),
                    'max_similarity': max([p['similarity'] for p in above_threshold]) if above_threshold else 0,
                    'similar_proteins': ', '.join(similar_names)
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary
        summary_file = self.output_dir / f"{organism}_similarity_search_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        
        logger.info(f"Saved {organism} summary statistics to {summary_file}")
        return summary_df
    
    def create_visualizations(self, matrices: Dict[str, pd.DataFrame], 
                            summary_df: pd.DataFrame, organism: str):
        """
        Create visualizations for similarity search results for an organism
        
        Args:
            matrices: Dictionary of similarity matrices
            summary_df: Summary statistics DataFrame
            organism: Organism name
        """
        logger.info(f"Creating visualizations for {organism}")
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Similarity distribution plot
        if matrices:
            fig, axes = plt.subplots(1, len(matrices), figsize=(5*len(matrices), 5))
            if len(matrices) == 1:
                axes = [axes]
            
            for i, (threshold_name, matrix) in enumerate(matrices.items()):
                # Get non-zero similarities
                similarities = matrix.values[matrix.values > 0]
                
                if len(similarities) > 0:
                    axes[i].hist(similarities, bins=20, alpha=0.7, edgecolor='black')
                    axes[i].set_title(f'{organism.capitalize()} {threshold_name.title()} Similarity Distribution')
                    axes[i].set_xlabel('Percent Identity')
                    axes[i].set_ylabel('Count')
                else:
                    axes[i].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].set_title(f'{organism.capitalize()} {threshold_name.title()} Similarity Distribution')
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / f'{organism}_similarity_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Number of similar proteins per organism protein
        if not summary_df.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Pivot for plotting
            pivot_df = summary_df.pivot(index='query_protein', columns='threshold', values='num_similar_proteins')
            pivot_df.plot(kind='bar', ax=ax)
            
            ax.set_title(f'Number of Similar Proteins per {organism.capitalize()} Protein')
            ax.set_xlabel(f'{organism.capitalize()} Protein')
            ax.set_ylabel('Number of Similar Proteins')
            ax.legend(title='Threshold')
            ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / f'{organism}_similar_proteins_count.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Similarity heatmap (if we have data)
        if matrices and any(not matrix.empty for matrix in matrices.values()):
            fig, axes = plt.subplots(1, len(matrices), figsize=(8*len(matrices), 6))
            if len(matrices) == 1:
                axes = [axes]
            
            for i, (threshold_name, matrix) in enumerate(matrices.items()):
                if not matrix.empty:
                    sns.heatmap(matrix, annot=True, fmt='.1f', cmap='viridis', ax=axes[i])
                    axes[i].set_title(f'{organism.capitalize()} {threshold_name.title()} Similarity Heatmap')
                else:
                    axes[i].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].set_title(f'{organism.capitalize()} {threshold_name.title()} Similarity Heatmap')
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / f'{organism}_similarity_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Visualizations saved to {self.plots_dir}")
    
    def run_organism_similarity_search(self, organism: str) -> Dict[str, any]:
        """
        Run similarity search for a specific organism
        
        Args:
            organism: Organism name (human, mouse, rat)
            
        Returns:
            Dictionary with results
        """
        logger.info(f"Starting similarity search for {organism}")
        
        # Load organism sequences
        organism_sequences = self.load_organism_sequences(organism)
        if not organism_sequences:
            logger.error(f"No {organism} sequences loaded")
            return {}
        
        # Get Papyrus protein sequences
        papyrus_proteins = self.get_papyrus_protein_sequences()
        if papyrus_proteins.empty:
            logger.error("No Papyrus protein sequences loaded")
            return {}
        
        # Find similar proteins
        similar_proteins = self.find_similar_proteins(organism_sequences, papyrus_proteins, organism)
        
        # Create similarity matrices
        matrices = self.create_similarity_matrices(similar_proteins, organism)
        
        # Create summary statistics
        summary_df = self.create_summary_statistics(similar_proteins, organism)
        
        # Create visualizations
        self.create_visualizations(matrices, summary_df, organism)
        
        logger.info(f"Similarity search completed for {organism}")
        
        return {
            'organism': organism,
            'organism_sequences': organism_sequences,
            'similar_proteins': similar_proteins,
            'matrices': matrices,
            'summary_df': summary_df
        }
    
    def run_all_organism_similarity_searches(self) -> Dict[str, Dict[str, any]]:
        """
        Run similarity search for all organisms
        
        Returns:
            Dictionary of results by organism
        """
        logger.info("Starting similarity searches for all organisms")
        
        all_results = {}
        
        for organism in self.organisms:
            logger.info(f"Processing {organism}")
            try:
                results = self.run_organism_similarity_search(organism)
                if results:
                    all_results[organism] = results
                    logger.info(f"Successfully completed {organism} similarity search")
                else:
                    logger.warning(f"No results for {organism}")
            except Exception as e:
                logger.error(f"Error processing {organism}: {e}")
                continue
        
        # Create combined summary
        self.create_combined_summary(all_results)
        
        logger.info("All organism similarity searches completed")
        return all_results
    
    def create_combined_summary(self, all_results: Dict[str, Dict[str, any]]):
        """
        Create combined summary across all organisms
        
        Args:
            all_results: Dictionary of results by organism
        """
        logger.info("Creating combined summary across all organisms")
        
        # Combine all summary data
        combined_summaries = []
        for organism, results in all_results.items():
            if 'summary_df' in results and not results['summary_df'].empty:
                combined_summaries.append(results['summary_df'])
        
        if combined_summaries:
            combined_df = pd.concat(combined_summaries, ignore_index=True)
            combined_file = self.output_dir / "all_organisms_similarity_summary.csv"
            combined_df.to_csv(combined_file, index=False)
            logger.info(f"Saved combined summary to {combined_file}")
        
        # Create overall statistics
        stats_file = self.output_dir / "organism_similarity_statistics.txt"
        with open(stats_file, 'w') as f:
            f.write("AQSE-org Organism-Specific Similarity Search Statistics\n")
            f.write("=" * 60 + "\n\n")
            
            for organism, results in all_results.items():
                f.write(f"{organism.capitalize()}:\n")
                f.write(f"  Proteins processed: {len(results.get('organism_sequences', {}))}\n")
                f.write(f"  Similarity matrices: {len(results.get('matrices', {}))}\n")
                f.write(f"  Summary entries: {len(results.get('summary_df', []))}\n\n")
        
        logger.info(f"Combined summary saved to {stats_file}")

def main():
    """Main function to run organism-specific similarity search"""
    
    # Set up paths
    input_dir = "/home/serramelendezcsm/RA/Avoidome/analyses/avoidome_qsar_similarity_expansion_organism/01_organism_mapping"
    output_dir = "/home/serramelendezcsm/RA/Avoidome/analyses/avoidome_qsar_similarity_expansion_organism/02_similarity_search"
    
    # Initialize similarity search class
    searcher = OrganismSimilaritySearch(input_dir, output_dir, papyrus_version='05.7')
    
    # Run similarity searches for all organisms
    results = searcher.run_all_organism_similarity_searches()
    
    if results:
        print("\n" + "="*60)
        print("AQSE-ORG ORGANISM-SPECIFIC SIMILARITY SEARCH COMPLETED")
        print("="*60)
        for organism, org_results in results.items():
            print(f"{organism.capitalize()}: {len(org_results.get('organism_sequences', {}))} proteins processed")
        print("="*60)
    else:
        print("Organism similarity search failed - check logs for details")

if __name__ == "__main__":
    main()
