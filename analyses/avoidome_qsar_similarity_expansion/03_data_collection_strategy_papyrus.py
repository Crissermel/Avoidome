#!/usr/bin/env python3
"""
AQSE Workflow - Step 3: Data Collection Strategy (Fixed for Papyrus API)

This script collects bioactivity data from Papyrus for proteins identified in the similarity search.
It creates expanded datasets at different similarity levels and prepares them for QSAR modeling.

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

# Papyrus imports
from papyrus_scripts import PapyrusDataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataCollectionStrategy:
    """Handles data collection from Papyrus for AQSE workflow"""
    
    def __init__(self, similarity_results_dir: str, output_dir: str):
        """
        Initialize the data collection class
        
        Args:
            similarity_results_dir: Directory with similarity search results
            output_dir: Output directory for collected data
        """
        self.similarity_results_dir = Path(similarity_results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.datasets_dir = self.output_dir / "expanded_datasets"
        self.datasets_dir.mkdir(exist_ok=True)
        
        self.statistics_dir = self.output_dir / "statistics"
        self.statistics_dir.mkdir(exist_ok=True)
        
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        # Initialize Papyrus dataset
        self.papyrus_data = None
        self.papyrus_df = None
        
        # Activity thresholds (pchembl_value based)
        self.activity_thresholds = {
            'high_activity': 5.0,  # pchembl_value >= 5.0 (high activity)
            'medium_activity': 4.0,  # pchembl_value >= 4.0 (medium activity)
            'low_activity': 3.0   # pchembl_value >= 3.0 (low activity)
        }
    
    def initialize_papyrus_dataset(self):
        """Initialize Papyrus dataset"""
        try:
            logger.info("Initializing Papyrus dataset...")
            self.papyrus_data = PapyrusDataset(version='latest', plusplus=True)
            logger.info("Loading full Papyrus dataset into DataFrame...")
            self.papyrus_df = self.papyrus_data.to_dataframe()
            logger.info(f"Loaded {len(self.papyrus_df)} total activities from Papyrus")
        except Exception as e:
            logger.error(f"Error initializing Papyrus dataset: {e}")
            raise
    
    def load_similarity_results(self) -> Tuple[Dict[str, Dict[str, List[str]]], pd.DataFrame]:
        """
        Load similarity search results from summary file
        
        Returns:
            Tuple of (similar_proteins_dict, summary_dataframe)
        """
        logger.info("Loading similarity search results")
        
        # Load summary file
        summary_file = self.similarity_results_dir / "similarity_search_summary.csv"
        if not summary_file.exists():
            logger.error(f"Summary file not found: {summary_file}")
            return {}, pd.DataFrame()
        
        summary_df = pd.read_csv(summary_file)
        
        # Convert to nested dictionary structure
        similar_proteins = {}
        for _, row in summary_df.iterrows():
            query_protein = row['query_protein']
            threshold = row['threshold']
            similar_proteins_str = row['similar_proteins']
            
            if query_protein not in similar_proteins:
                similar_proteins[query_protein] = {}
            
            # Parse similar proteins string
            if pd.notna(similar_proteins_str) and similar_proteins_str.strip():
                # Extract protein IDs from string like "P05177_WT (100.0%), P20813_WT (78.0%)"
                similar_list = []
                for protein_entry in similar_proteins_str.split(', '):
                    if '(' in protein_entry:
                        protein_id = protein_entry.split('_WT')[0]  # Remove _WT suffix
                        similar_list.append(protein_id)
                similar_proteins[query_protein][threshold] = similar_list
            else:
                similar_proteins[query_protein][threshold] = []
        
        logger.info(f"Loaded similarity results for {len(similar_proteins)} proteins")
        return similar_proteins, summary_df
    
    def collect_bioactivity_data(self, protein_ids: List[str]) -> pd.DataFrame:
        """
        Collect bioactivity data for given protein IDs from Papyrus
        
        Args:
            protein_ids: List of protein UniProt IDs
            
        Returns:
            DataFrame with bioactivity data
        """
        logger.info(f"Collecting bioactivity data for {len(protein_ids)} proteins")
        
        if self.papyrus_df is None:
            logger.error("Papyrus dataset not initialized")
            return pd.DataFrame()
        
        # Filter bioactivity data for the specified proteins
        bioactivity_data = self.papyrus_df[
            self.papyrus_df['accession'].isin(protein_ids)
        ].copy()
        
        if bioactivity_data.empty:
            logger.warning(f"No bioactivity data found for proteins: {protein_ids}")
            return pd.DataFrame()
        
        # Filter for valid data
        bioactivity_data = bioactivity_data.dropna(subset=['SMILES', 'pchembl_value_Mean'])
        bioactivity_data = bioactivity_data[bioactivity_data['SMILES'] != '']
        
        logger.info(f"Collected {len(bioactivity_data)} bioactivity data points")
        return bioactivity_data
    
    def prepare_activity_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare activity values using pchembl_value directly
        
        Args:
            df: DataFrame with bioactivity data
            
        Returns:
            DataFrame with prepared activity values
        """
        logger.info("Preparing activity values using pchembl_value")
        
        df = df.copy()
        
        # Use pchembl_value_Mean directly (already log-transformed)
        df['standardized_activity'] = df['pchembl_value_Mean']
        
        # Remove invalid values
        df = df.dropna(subset=['standardized_activity'])
        
        logger.info(f"Prepared {len(df)} activity values using pchembl_value")
        return df
    
    
    def create_expanded_datasets(self, similar_proteins: Dict[str, Dict[str, List[str]]]) -> Dict[str, pd.DataFrame]:
        """
        Create expanded datasets for each similarity level
        
        Args:
            similar_proteins: Dictionary of similar proteins by threshold
            
        Returns:
            Dictionary of expanded datasets by threshold
        """
        logger.info("Creating expanded datasets")
        
        # Initialize Papyrus dataset
        self.initialize_papyrus_dataset()
        
        expanded_datasets = {}
        
        for threshold_name in ['high', 'medium', 'low']:
            logger.info(f"Creating dataset for {threshold_name} threshold")
            
            # Collect all protein IDs for this threshold
            all_protein_ids = set()
            for query_protein, thresholds_dict in similar_proteins.items():
                if threshold_name in thresholds_dict:
                    all_protein_ids.add(query_protein)  # Add original avoidome protein
                    all_protein_ids.update(thresholds_dict[threshold_name])  # Add similar proteins
            
            # Convert to list
            protein_ids = list(all_protein_ids)
            
            if not protein_ids:
                logger.warning(f"No proteins found for {threshold_name} threshold")
                continue
            
            # Collect bioactivity data
            bioactivity_df = self.collect_bioactivity_data(protein_ids)
            
            if bioactivity_df.empty:
                logger.warning(f"No bioactivity data found for {threshold_name} threshold")
                continue
            
            # Prepare activity values using pchembl_value directly
            bioactivity_df = self.prepare_activity_values(bioactivity_df)
            
            # Add threshold information
            bioactivity_df['similarity_threshold'] = threshold_name
            bioactivity_df['threshold_value'] = self.get_threshold_value(threshold_name)
            
            # Rename columns to match expected format for Step 4
            bioactivity_df = bioactivity_df.rename(columns={
                'accession': 'accession',  # Keep as accession for Step 4
                'pchembl_value_Mean': 'activity_value',
                'SMILES': 'SMILES'  # Keep as SMILES for Step 4
            })
            
            
            # Save dataset
            dataset_file = self.datasets_dir / f"expanded_dataset_{threshold_name}.csv"
            bioactivity_df.to_csv(dataset_file, index=False)
            
            expanded_datasets[threshold_name] = bioactivity_df
            
            logger.info(f"Created {threshold_name} dataset: {len(bioactivity_df)} data points")
        
        return expanded_datasets
    
    def get_threshold_value(self, threshold_name: str) -> float:
        """Get numeric threshold value for threshold name"""
        threshold_values = {
            'high': 70.0,
            'medium': 50.0,
            'low': 30.0
        }
        return threshold_values.get(threshold_name, 0.0)
    
    def create_statistics(self, expanded_datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create statistics for expanded datasets
        
        Args:
            expanded_datasets: Dictionary of expanded datasets
            
        Returns:
            DataFrame with statistics
        """
        logger.info("Creating statistics")
        
        stats_data = []
        
        for threshold_name, df in expanded_datasets.items():
            if df.empty:
                continue
            
            stats = {
                'threshold': threshold_name,
                'num_records': len(df),
                'num_proteins': df['accession'].nunique(),
                'num_compounds': df['SMILES'].nunique(),
                'num_protein_combinations': len(df.groupby('accession')),
                'min_samples_per_protein': df['accession'].value_counts().min(),
                'max_samples_per_protein': df['accession'].value_counts().max(),
                'mean_samples_per_protein': df['accession'].value_counts().mean(),
                'min_activity': df['standardized_activity'].min(),
                'max_activity': df['standardized_activity'].max(),
                'mean_activity': df['standardized_activity'].mean(),
                'median_activity': df['standardized_activity'].median()
            }
            stats_data.append(stats)
        
        stats_df = pd.DataFrame(stats_data)
        
        # Save statistics
        stats_file = self.statistics_dir / "data_collection_summary.csv"
        stats_df.to_csv(stats_file, index=False)
        
        logger.info(f"Created statistics: {len(stats_df)} thresholds")
        return stats_df
    
    def create_visualizations(self, expanded_datasets: Dict[str, pd.DataFrame], stats_df: pd.DataFrame):
        """Create visualizations for the collected data"""
        logger.info("Creating visualizations")
        
        if stats_df.empty:
            logger.warning("No data available for visualization")
            return
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Dataset comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Number of records by threshold
        sns.barplot(data=stats_df, x='threshold', y='num_records', ax=axes[0, 0])
        axes[0, 0].set_title('Number of Records by Threshold')
        axes[0, 0].set_ylabel('Number of Records')
        
        # Number of proteins by threshold
        sns.barplot(data=stats_df, x='threshold', y='num_proteins', ax=axes[0, 1])
        axes[0, 1].set_title('Number of Proteins by Threshold')
        axes[0, 1].set_ylabel('Number of Proteins')
        
        # Number of compounds by threshold
        sns.barplot(data=stats_df, x='threshold', y='num_compounds', ax=axes[1, 0])
        axes[1, 0].set_title('Number of Compounds by Threshold')
        axes[1, 0].set_ylabel('Number of Compounds')
        
        # Mean activity by threshold
        sns.barplot(data=stats_df, x='threshold', y='mean_activity', ax=axes[1, 1])
        axes[1, 1].set_title('Mean Activity by Threshold')
        axes[1, 1].set_ylabel('Mean pchembl_value')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'dataset_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Activity distribution
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, (threshold_name, df) in enumerate(expanded_datasets.items()):
            if not df.empty:
                axes[i].hist(df['standardized_activity'], bins=50, alpha=0.7)
                axes[i].set_title(f'Activity Distribution - {threshold_name.title()}')
                axes[i].set_xlabel('pchembl_value')
                axes[i].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'activity_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {self.plots_dir}")
    
    def run_data_collection(self) -> Dict[str, pd.DataFrame]:
        """Run the complete data collection workflow"""
        logger.info("Starting data collection workflow")
        
        # Load similarity results
        similar_proteins, summary_df = self.load_similarity_results()
        
        if not similar_proteins:
            logger.error("No similarity results found")
            return {}
        
        # Create expanded datasets
        expanded_datasets = self.create_expanded_datasets(similar_proteins)
        
        if not expanded_datasets:
            logger.error("No expanded datasets created")
            return {}
        
        # Create statistics
        stats_df = self.create_statistics(expanded_datasets)
        
        # Create visualizations
        self.create_visualizations(expanded_datasets, stats_df)
        
        logger.info("Data collection workflow completed successfully")
        
        return expanded_datasets

def main():
    """Main function to run data collection"""
    
    # Set up paths
    similarity_results_dir = "/home/serramelendezcsm/RA/Avoidome/analyses/avoidome_qsar_similarity_expansion/02_similarity_search"
    output_dir = "/home/serramelendezcsm/RA/Avoidome/analyses/avoidome_qsar_similarity_expansion/03_data_collection"
    
    # Initialize data collection strategy
    data_collector = DataCollectionStrategy(similarity_results_dir, output_dir)
    
    # Run data collection
    results = data_collector.run_data_collection()
    
    if results:
        print("\n" + "="*60)
        print("AQSE DATA COLLECTION COMPLETED")
        print("="*60)
        for threshold, df in results.items():
            print(f"{threshold.title()} threshold: {len(df)} data points")
        print("="*60)
    else:
        print("Data collection failed - check logs for details")

if __name__ == "__main__":
    main()