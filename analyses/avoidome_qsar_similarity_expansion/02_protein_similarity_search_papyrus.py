#!/usr/bin/env python3
"""
AQSE Workflow - Step 2: Protein Similarity Search using Papyrus-scripts

This script uses the papyrus-scripts library to find similar proteins in Papyrus
database based on sequence similarity and creates similarity matrices.

Author: AQSE Pipeline
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

class ProteinSimilaritySearchPapyrus:
    """Handles protein similarity search using papyrus-scripts library"""
    
    def __init__(self, input_dir: str, output_dir: str, papyrus_version: str = '05.7'):
        """
        Initialize the similarity search class
        
        Args:
            input_dir: Directory with prepared FASTA files
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
        
        # Initialize Papyrus dataset
        logger.info(f"Initializing Papyrus dataset version {papyrus_version}")
        self.papyrus_data = PapyrusDataset(version=papyrus_version, plusplus=False)
        
    def load_avoidome_sequences(self) -> Dict[str, str]:
        """
        Load avoidome protein sequences from FASTA files
        
        Returns:
            Dictionary mapping protein IDs to sequences
        """
        logger.info("Loading avoidome protein sequences")
        
        sequences = {}
        fasta_files = list(self.input_dir.glob("*.fasta"))
        
        for fasta_file in fasta_files:
            logger.info(f"Loading sequences from {fasta_file}")
            for record in SeqIO.parse(fasta_file, "fasta"):
                protein_id = record.id
                sequence = str(record.seq)
                sequences[protein_id] = sequence
                logger.info(f"Loaded {protein_id}: {len(sequence)} amino acids")
        
        logger.info(f"Loaded {len(sequences)} avoidome protein sequences")
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
        
        # Count identical positions
        matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
        
        # Calculate percent identity
        percent_identity = (matches / min_len) * 100
        
        return percent_identity
    
    def find_similar_proteins(self, avoidome_sequences: Dict[str, str], 
                            papyrus_proteins: pd.DataFrame) -> Dict[str, List[Dict]]:
        """
        Find similar proteins in Papyrus for each avoidome protein
        
        Args:
            avoidome_sequences: Dictionary of avoidome protein sequences
            papyrus_proteins: DataFrame of Papyrus proteins
            
        Returns:
            Dictionary of similar proteins by avoidome protein
        """
        logger.info("Finding similar proteins in Papyrus database")
        
        similar_proteins = {}
        
        for avoidome_id, avoidome_seq in avoidome_sequences.items():
            logger.info(f"Searching for proteins similar to {avoidome_id}")
            
            similar_proteins[avoidome_id] = []
            
            for _, papyrus_protein in papyrus_proteins.iterrows():
                papyrus_id = papyrus_protein['target_id']
                papyrus_seq = papyrus_protein['Sequence']
                
                # Calculate similarity
                similarity = self.calculate_sequence_similarity(avoidome_seq, papyrus_seq)
                
                # Store if above any threshold
                if similarity >= min(self.thresholds.values()):
                    similar_proteins[avoidome_id].append({
                        'papyrus_id': papyrus_id,
                        'similarity': similarity,
                        'organism': papyrus_protein.get('Organism', 'Unknown'),
                        'protein_class': papyrus_protein.get('Classification', 'Unknown'),
                        'sequence': papyrus_seq
                    })
            
            # Sort by similarity (descending)
            similar_proteins[avoidome_id].sort(key=lambda x: x['similarity'], reverse=True)
            
            logger.info(f"Found {len(similar_proteins[avoidome_id])} similar proteins for {avoidome_id}")
        
        return similar_proteins
    
    def create_similarity_matrices(self, similar_proteins: Dict[str, List[Dict]]) -> Dict[str, pd.DataFrame]:
        """
        Create similarity matrices for different thresholds
        
        Args:
            similar_proteins: Dictionary of similar proteins
            
        Returns:
            Dictionary of similarity matrices by threshold
        """
        logger.info("Creating similarity matrices")
        
        matrices = {}
        
        for threshold_name, threshold_value in self.thresholds.items():
            logger.info(f"Creating similarity matrix for {threshold_name} threshold ({threshold_value}%)")
            
            # Collect all similarities above threshold
            matrix_data = []
            
            for avoidome_id, similar_list in similar_proteins.items():
                for similar_protein in similar_list:
                    if similar_protein['similarity'] >= threshold_value:
                        matrix_data.append({
                            'query_id': avoidome_id,
                            'subject_id': similar_protein['papyrus_id'],
                            'similarity': similar_protein['similarity'],
                            'organism': similar_protein['organism'],
                            'protein_class': similar_protein['protein_class']
                        })
            
            if not matrix_data:
                logger.warning(f"No hits above {threshold_value}% similarity")
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
            matrix_file = self.similarity_matrices_dir / f"similarity_matrix_{threshold_name}.csv"
            similarity_matrix.to_csv(matrix_file)
            
            matrices[threshold_name] = similarity_matrix
            
            logger.info(f"Created {threshold_name} matrix: {similarity_matrix.shape}")
        
        return matrices
    
    def create_summary_statistics(self, similar_proteins: Dict[str, List[Dict]]) -> pd.DataFrame:
        """
        Create summary statistics for similarity search
        
        Args:
            similar_proteins: Dictionary of similar proteins
            
        Returns:
            DataFrame with summary statistics
        """
        logger.info("Creating summary statistics")
        
        summary_data = []
        
        for avoidome_id, similar_list in similar_proteins.items():
            for threshold_name, threshold_value in self.thresholds.items():
                # Count proteins above threshold
                above_threshold = [p for p in similar_list if p['similarity'] >= threshold_value]
                
                # Get top 10 most similar
                top_similar = sorted(above_threshold, key=lambda x: x['similarity'], reverse=True)[:10]
                similar_names = [f"{p['papyrus_id']} ({p['similarity']:.1f}%)" for p in top_similar]
                
                summary_data.append({
                    'query_protein': avoidome_id,
                    'threshold': threshold_name,
                    'threshold_value': threshold_value,
                    'num_similar_proteins': len(above_threshold),
                    'max_similarity': max([p['similarity'] for p in above_threshold]) if above_threshold else 0,
                    'similar_proteins': ', '.join(similar_names)
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary
        summary_file = self.output_dir / "similarity_search_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        
        logger.info(f"Saved summary statistics to {summary_file}")
        return summary_df
    
    def create_visualizations(self, matrices: Dict[str, pd.DataFrame], summary_df: pd.DataFrame):
        """
        Create visualizations for similarity search results
        
        Args:
            matrices: Dictionary of similarity matrices
            summary_df: Summary statistics DataFrame
        """
        logger.info("Creating visualizations")
        
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
                    axes[i].set_title(f'{threshold_name.title()} Similarity Distribution')
                    axes[i].set_xlabel('Percent Identity')
                    axes[i].set_ylabel('Count')
                else:
                    axes[i].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].set_title(f'{threshold_name.title()} Similarity Distribution')
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'similarity_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Number of similar proteins per avoidome protein
        if not summary_df.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Pivot for plotting
            pivot_df = summary_df.pivot(index='query_protein', columns='threshold', values='num_similar_proteins')
            pivot_df.plot(kind='bar', ax=ax)
            
            ax.set_title('Number of Similar Proteins per Avoidome Protein')
            ax.set_xlabel('Avoidome Protein')
            ax.set_ylabel('Number of Similar Proteins')
            ax.legend(title='Threshold')
            ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'similar_proteins_count.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Similarity heatmap (if we have data)
        if matrices and any(not matrix.empty for matrix in matrices.values()):
            fig, axes = plt.subplots(1, len(matrices), figsize=(8*len(matrices), 6))
            if len(matrices) == 1:
                axes = [axes]
            
            for i, (threshold_name, matrix) in enumerate(matrices.items()):
                if not matrix.empty:
                    sns.heatmap(matrix, annot=True, fmt='.1f', cmap='viridis', ax=axes[i])
                    axes[i].set_title(f'{threshold_name.title()} Similarity Heatmap')
                else:
                    axes[i].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].set_title(f'{threshold_name.title()} Similarity Heatmap')
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'similarity_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Visualizations saved to {self.plots_dir}")
    
    def run_similarity_search(self) -> Dict[str, any]:
        """
        Run the complete similarity search workflow using Papyrus
        
        Returns:
            Dictionary with results
        """
        logger.info("Starting AQSE protein similarity search with Papyrus")
        
        # Load avoidome sequences
        avoidome_sequences = self.load_avoidome_sequences()
        if not avoidome_sequences:
            logger.error("No avoidome sequences loaded")
            return {}
        
        # Get Papyrus protein sequences
        papyrus_proteins = self.get_papyrus_protein_sequences()
        if papyrus_proteins.empty:
            logger.error("No Papyrus protein sequences loaded")
            return {}
        
        # Find similar proteins
        similar_proteins = self.find_similar_proteins(avoidome_sequences, papyrus_proteins)
        
        # Create similarity matrices
        matrices = self.create_similarity_matrices(similar_proteins)
        
        # Create summary statistics
        summary_df = self.create_summary_statistics(similar_proteins)
        
        # Create visualizations
        self.create_visualizations(matrices, summary_df)
        
        logger.info("Protein similarity search completed successfully")
        
        return {
            'avoidome_sequences': avoidome_sequences,
            'papyrus_proteins': papyrus_proteins,
            'similar_proteins': similar_proteins,
            'matrices': matrices,
            'summary_df': summary_df
        }

def main():
    """Main function to run protein similarity search"""
    
    # Set up paths
    input_dir = "/home/serramelendezcsm/RA/Avoidome/analyses/avoidome_qsar_similarity_expansion/01_input_preparation"
    output_dir = "/home/serramelendezcsm/RA/Avoidome/analyses/avoidome_qsar_similarity_expansion/02_similarity_search"
    
    # Initialize similarity search class
    searcher = ProteinSimilaritySearchPapyrus(input_dir, output_dir, papyrus_version='05.7')
    
    # Run similarity search
    results = searcher.run_similarity_search()
    
    if results:
        print("\n" + "="*60)
        print("AQSE PROTEIN SIMILARITY SEARCH COMPLETED")
        print("="*60)
        print(f"Avoidome sequences: {len(results['avoidome_sequences'])}")
        print(f"Papyrus proteins: {len(results['papyrus_proteins'])}")
        print(f"Similarity matrices created: {len(results['matrices'])}")
        print(f"Summary statistics: {len(results['summary_df'])} rows")
        print("="*60)
    else:
        print("Similarity search failed - check logs for details")

if __name__ == "__main__":
    main()