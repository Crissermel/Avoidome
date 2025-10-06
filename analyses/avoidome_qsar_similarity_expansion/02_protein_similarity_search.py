#!/usr/bin/env python3
"""
AQSE Workflow - Step 2: Protein Similarity Search

This script performs BLAST similarity search between avoidome proteins and Papyrus database.
It creates similarity matrices and identifies proteins at different similarity thresholds.

Author: AQSE Pipeline
Date: 2025
"""

import pandas as pd
import numpy as np
import os
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProteinSimilaritySearch:
    """Handles BLAST similarity search for AQSE workflow"""
    
    def __init__(self, input_dir: str, papyrus_db_path: str, output_dir: str):
        """
        Initialize the similarity search class
        
        Args:
            input_dir: Directory with prepared FASTA files
            papyrus_db_path: Path to Papyrus BLAST database
            output_dir: Output directory for results
        """
        self.input_dir = Path(input_dir)
        self.papyrus_db_path = papyrus_db_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.blast_results_dir = self.output_dir / "blast_results"
        self.blast_results_dir.mkdir(exist_ok=True)
        
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
    
    def run_blast_search(self, query_file: str, output_file: str) -> bool:
        """
        Run BLAST search for a single query file
        
        Args:
            query_file: Path to query FASTA file
            output_file: Path to output file
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Running BLAST search for {query_file}")
        
        blast_cmd = [
            'blastp',
            '-query', query_file,
            '-db', self.papyrus_db_path,
            '-out', output_file,
            '-outfmt', '6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore',
            '-evalue', '1e-5',
            '-max_target_seqs', '1000'
        ]
        
        try:
            result = subprocess.run(blast_cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info(f"BLAST search completed for {query_file}")
                return True
            else:
                logger.error(f"BLAST search failed for {query_file}: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"BLAST search timed out for {query_file}")
            return False
        except Exception as e:
            logger.error(f"Error running BLAST for {query_file}: {e}")
            return False
    
    def run_all_blast_searches(self) -> List[str]:
        """
        Run BLAST searches for all avoidome proteins
        
        Returns:
            List of successful output files
        """
        logger.info("Starting BLAST searches for all avoidome proteins")
        
        # Get all FASTA files
        fasta_files = list(self.input_dir.glob("*.fasta"))
        logger.info(f"Found {len(fasta_files)} FASTA files to process")
        
        successful_outputs = []
        
        for fasta_file in fasta_files:
            # Create output filename
            output_file = self.blast_results_dir / f"{fasta_file.stem}_blast_results.txt"
            
            # Run BLAST
            if self.run_blast_search(str(fasta_file), str(output_file)):
                successful_outputs.append(str(output_file))
            else:
                logger.warning(f"Failed to process {fasta_file}")
        
        logger.info(f"Completed BLAST searches: {len(successful_outputs)} successful")
        return successful_outputs
    
    def parse_blast_results(self, blast_file: str) -> pd.DataFrame:
        """
        Parse BLAST results from output file
        
        Args:
            blast_file: Path to BLAST output file
            
        Returns:
            DataFrame with BLAST results
        """
        columns = [
            'query_id', 'subject_id', 'percent_identity', 'alignment_length',
            'mismatches', 'gap_opens', 'query_start', 'query_end',
            'subject_start', 'subject_end', 'evalue', 'bit_score'
        ]
        
        try:
            df = pd.read_csv(blast_file, sep='\t', names=columns)
            return df
        except Exception as e:
            logger.error(f"Error parsing BLAST results from {blast_file}: {e}")
            return pd.DataFrame()
    
    def create_similarity_matrices(self, blast_files: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Create similarity matrices for different thresholds
        
        Args:
            blast_files: List of BLAST result files
            
        Returns:
            Dictionary of similarity matrices by threshold
        """
        logger.info("Creating similarity matrices")
        
        # Load all BLAST results
        all_results = []
        for blast_file in blast_files:
            df = self.parse_blast_results(blast_file)
            if not df.empty:
                all_results.append(df)
        
        if not all_results:
            logger.error("No BLAST results found")
            return {}
        
        # Combine all results
        combined_results = pd.concat(all_results, ignore_index=True)
        logger.info(f"Combined {len(combined_results)} BLAST hits")
        
        # Create similarity matrices for each threshold
        matrices = {}
        
        for threshold_name, threshold_value in self.thresholds.items():
            logger.info(f"Creating similarity matrix for {threshold_name} threshold ({threshold_value}%)")
            
            # Filter by identity threshold
            filtered_results = combined_results[
                combined_results['percent_identity'] >= threshold_value
            ].copy()
            
            if filtered_results.empty:
                logger.warning(f"No hits above {threshold_value}% identity")
                continue
            
            # Create pivot table
            similarity_matrix = filtered_results.pivot_table(
                index='query_id',
                columns='subject_id',
                values='percent_identity',
                fill_value=0
            )
            
            # Make the matrix symmetric by adding reverse relationships
            # If A is similar to B, then B should also be similar to A
            symmetric_matrix = similarity_matrix.copy()
            
            # Add reverse relationships
            for query_id in similarity_matrix.index:
                for subject_id in similarity_matrix.columns:
                    if similarity_matrix.loc[query_id, subject_id] > 0:
                        # Add reverse relationship if subject_id is in the index
                        if subject_id in similarity_matrix.index:
                            symmetric_matrix.loc[subject_id, query_id] = similarity_matrix.loc[query_id, subject_id]
            
            # Ensure all proteins appear in both rows and columns
            all_proteins = set(similarity_matrix.index) | set(similarity_matrix.columns)
            all_proteins = sorted(list(all_proteins))
            
            # Reindex to include all proteins
            symmetric_matrix = symmetric_matrix.reindex(index=all_proteins, columns=all_proteins, fill_value=0)
            
            # Set diagonal to 100% (self-similarity)
            for protein in all_proteins:
                symmetric_matrix.loc[protein, protein] = 100.0
            
            # Add _WT suffix to column names for Step 4 compatibility
            symmetric_matrix.columns = [f"{col}_WT" for col in symmetric_matrix.columns]
            
            similarity_matrix = symmetric_matrix
            
            # Save matrix
            matrix_file = self.similarity_matrices_dir / f"similarity_matrix_{threshold_name}.csv"
            similarity_matrix.to_csv(matrix_file)
            
            matrices[threshold_name] = similarity_matrix
            
            logger.info(f"Created {threshold_name} matrix: {similarity_matrix.shape}")
        
        return matrices
    
    def identify_similar_proteins(self, matrices: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, List[str]]]:
        """
        Identify similar proteins for each avoidome protein at each threshold
        
        Args:
            matrices: Dictionary of similarity matrices
            
        Returns:
            Dictionary of similar proteins by threshold and query protein
        """
        logger.info("Identifying similar proteins")
        
        similar_proteins = {}
        
        for threshold_name, matrix in matrices.items():
            similar_proteins[threshold_name] = {}
            
            for query_protein in matrix.index:
                # Get proteins with similarity > 0 (excluding self)
                similar_prots = matrix.loc[query_protein][matrix.loc[query_protein] > 0].index.tolist()
                
                # Remove self if present
                if query_protein in similar_prots:
                    similar_prots.remove(query_protein)
                
                similar_proteins[threshold_name][query_protein] = similar_prots
                
                logger.info(f"{query_protein} ({threshold_name}): {len(similar_prots)} similar proteins")
        
        return similar_proteins
    
    def create_summary_statistics(self, similar_proteins: Dict[str, Dict[str, List[str]]]) -> pd.DataFrame:
        """
        Create summary statistics for similarity search
        
        Args:
            similar_proteins: Dictionary of similar proteins
            
        Returns:
            DataFrame with summary statistics
        """
        logger.info("Creating summary statistics")
        
        summary_data = []
        
        for threshold_name, proteins in similar_proteins.items():
            for query_protein, similar_list in proteins.items():
                summary_data.append({
                    'query_protein': query_protein,
                    'threshold': threshold_name,
                    'threshold_value': self.thresholds[threshold_name],
                    'num_similar_proteins': len(similar_list),
                    'similar_proteins': ', '.join(similar_list[:10]) + ('...' if len(similar_list) > 10 else '')
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
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, (threshold_name, matrix) in enumerate(matrices.items()):
            # Get non-zero similarities
            similarities = matrix.values[matrix.values > 0]
            
            if len(similarities) > 0:
                axes[i].hist(similarities, bins=20, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{threshold_name.title()} Similarity Distribution')
                axes[i].set_xlabel('Percent Identity')
                axes[i].set_ylabel('Count')
                axes[i].axvline(self.thresholds[threshold_name], color='red', linestyle='--', 
                              label=f'Threshold: {self.thresholds[threshold_name]}%')
                axes[i].legend()
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'similarity_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Number of similar proteins per query
        fig, ax = plt.subplots(figsize=(12, 6))
        
        threshold_counts = summary_df.groupby('threshold')['num_similar_proteins'].agg(['mean', 'std', 'count'])
        
        x_pos = np.arange(len(threshold_counts))
        ax.bar(x_pos, threshold_counts['mean'], yerr=threshold_counts['std'], 
               capsize=5, alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Similarity Threshold')
        ax.set_ylabel('Average Number of Similar Proteins')
        ax.set_title('Similar Proteins per Query Protein by Threshold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(threshold_counts.index)
        
        # Add count labels
        for i, (idx, row) in enumerate(threshold_counts.iterrows()):
            ax.text(i, row['mean'] + row['std'] + 0.5, f'n={int(row["count"])}', 
                   ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'similar_proteins_by_threshold.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Heatmap of similarity matrix (sample)
        if matrices:
            # Use the medium threshold matrix for heatmap
            matrix = list(matrices.values())[0]
            
            # Sample for visualization if too large
            if matrix.shape[0] > 20:
                matrix = matrix.iloc[:20, :20]
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(matrix, annot=True, fmt='.1f', cmap='viridis', 
                       cbar_kws={'label': 'Percent Identity'})
            plt.title('Protein Similarity Heatmap (Sample)')
            plt.xlabel('Subject Proteins')
            plt.ylabel('Query Proteins')
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'similarity_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Visualizations saved to {self.plots_dir}")
    
    def run_similarity_search(self) -> Dict[str, any]:
        """
        Run the complete similarity search workflow
        
        Returns:
            Dictionary with results
        """
        logger.info("Starting AQSE protein similarity search")
        
        # Run BLAST searches
        blast_files = self.run_all_blast_searches()
        
        if not blast_files:
            logger.error("No successful BLAST searches")
            return {}
        
        # Create similarity matrices
        matrices = self.create_similarity_matrices(blast_files)
        
        if not matrices:
            logger.error("No similarity matrices created")
            return {}
        
        # Identify similar proteins
        similar_proteins = self.identify_similar_proteins(matrices)
        
        # Create summary statistics
        summary_df = self.create_summary_statistics(similar_proteins)
        
        # Create visualizations
        self.create_visualizations(matrices, summary_df)
        
        logger.info("Protein similarity search completed successfully")
        
        return {
            'blast_files': blast_files,
            'matrices': matrices,
            'similar_proteins': similar_proteins,
            'summary_df': summary_df
        }

def main():
    """Main function to run protein similarity search"""
    
    # Set up paths
    input_dir = "/home/serramelendezcsm/RA/Avoidome/analyses/avoidome_qsar_similarity_expansion"
    papyrus_db_path = "/path/to/papyrus_protein_db"  # Update this path
    output_dir = "/home/serramelendezcsm/RA/Avoidome/analyses/avoidome_qsar_similarity_expansion/similarity_search"
    
    # Initialize similarity search class
    searcher = ProteinSimilaritySearch(input_dir, papyrus_db_path, output_dir)
    
    # Run similarity search
    results = searcher.run_similarity_search()
    
    if results:
        print("\n" + "="*60)
        print("AQSE PROTEIN SIMILARITY SEARCH COMPLETED")
        print("="*60)
        print(f"BLAST files processed: {len(results['blast_files'])}")
        print(f"Similarity matrices created: {len(results['matrices'])}")
        print(f"Summary statistics: {len(results['summary_df'])} rows")
        print("="*60)
    else:
        print("Similarity search failed - check logs for details")

if __name__ == "__main__":
    main()