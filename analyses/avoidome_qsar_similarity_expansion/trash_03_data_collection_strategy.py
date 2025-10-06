#!/usr/bin/env python3
"""
AQSE Workflow - Step 3: Data Collection Strategy

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
    
    def __init__(self, similarity_results_dir: str, papyrus_db_path: str, output_dir: str):
        """
        Initialize the data collection class
        
        Args:
            similarity_results_dir: Directory with similarity search results
            papyrus_db_path: Path to Papyrus database
            output_dir: Output directory for collected data
        """
        self.similarity_results_dir = Path(similarity_results_dir)
        self.papyrus_db_path = papyrus_db_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.datasets_dir = self.output_dir / "expanded_datasets"
        self.datasets_dir.mkdir(exist_ok=True)
        
        self.statistics_dir = self.output_dir / "statistics"
        self.statistics_dir.mkdir(exist_ok=True)
        
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        # Activity thresholds
        self.activity_thresholds = {
            'high_activity': 1.0,  # pIC50 > 1.0 (IC50 < 10 μM)
            'medium_activity': 0.0,  # pIC50 > 0.0 (IC50 < 1 μM)
            'low_activity': -1.0   # pIC50 > -1.0 (IC50 < 10 μM)
        }
    
    def load_similarity_results(self) -> Tuple[Dict[str, Dict[str, List[str]]], pd.DataFrame]:
        """
        Load similarity search results
        
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
        
        # Reconstruct similar proteins dictionary
        similar_proteins = {}
        for threshold in summary_df['threshold'].unique():
            similar_proteins[threshold] = {}
            
            threshold_data = summary_df[summary_df['threshold'] == threshold]
            for _, row in threshold_data.iterrows():
                query_protein = row['query_protein']
                similar_list = row['similar_proteins'].split(', ') if pd.notna(row['similar_proteins']) else []
                similar_proteins[threshold][query_protein] = similar_list
        
        logger.info(f"Loaded similarity results for {len(similar_proteins)} thresholds")
        return similar_proteins, summary_df
    
    def connect_to_papyrus_db(self) -> sqlite3.Connection:
        """
        Connect to Papyrus database
        
        Returns:
            SQLite connection object
        """
        try:
            conn = sqlite3.connect(self.papyrus_db_path)
            logger.info(f"Connected to Papyrus database: {self.papyrus_db_path}")
            return conn
        except Exception as e:
            logger.error(f"Error connecting to Papyrus database: {e}")
            raise
    
    def get_papyrus_schema(self, conn: sqlite3.Connection) -> Dict[str, List[str]]:
        """
        Get Papyrus database schema
        
        Args:
            conn: Database connection
            
        Returns:
            Dictionary mapping table names to column lists
        """
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        schema = {}
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table});")
            columns = [row[1] for row in cursor.fetchall()]
            schema[table] = columns
        
        logger.info(f"Found {len(tables)} tables in Papyrus database")
        return schema
    
    def collect_bioactivity_data(self, conn: sqlite3.Connection, protein_ids: List[str]) -> pd.DataFrame:
        """
        Collect bioactivity data for given protein IDs
        
        Args:
            conn: Database connection
            protein_ids: List of protein UniProt IDs
            
        Returns:
            DataFrame with bioactivity data
        """
        logger.info(f"Collecting bioactivity data for {len(protein_ids)} proteins")
        
        # Create placeholders for SQL IN clause
        placeholders = ','.join(['?' for _ in protein_ids])
        
        # Query bioactivity data
        query = f"""
        SELECT 
            p.uniprot_id,
            p.sequence,
            b.compound_id,
            b.activity_value,
            b.activity_type,
            b.activity_unit,
            b.organism,
            b.publication_year,
            c.smiles,
            c.molecular_weight,
            c.logp
        FROM proteins p
        JOIN bioactivity b ON p.uniprot_id = b.uniprot_id
        JOIN compounds c ON b.compound_id = c.compound_id
        WHERE p.uniprot_id IN ({placeholders})
        AND b.activity_value IS NOT NULL
        AND b.activity_type IN ('IC50', 'Ki', 'Kd', 'EC50')
        """
        
        try:
            df = pd.read_sql_query(query, conn, params=protein_ids)
            logger.info(f"Collected {len(df)} bioactivity data points")
            return df
        except Exception as e:
            logger.error(f"Error collecting bioactivity data: {e}")
            return pd.DataFrame()
    
    def standardize_activity_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize activity values to pIC50
        
        Args:
            df: DataFrame with bioactivity data
            
        Returns:
            DataFrame with standardized activity values
        """
        logger.info("Standardizing activity values to pIC50")
        
        df = df.copy()
        
        # Convert to pIC50
        def convert_to_pic50(row):
            activity_value = row['activity_value']
            activity_unit = row['activity_unit']
            
            if pd.isna(activity_value) or pd.isna(activity_unit):
                return np.nan
            
            # Convert to μM if needed
            if activity_unit == 'nM':
                activity_value = activity_value / 1000  # nM to μM
            elif activity_unit == 'pM':
                activity_value = activity_value / 1000000  # pM to μM
            elif activity_unit == 'mM':
                activity_value = activity_value * 1000  # mM to μM
            
            # Convert to pIC50 (negative log10 of μM)
            if activity_value > 0:
                return -np.log10(activity_value)
            else:
                return np.nan
        
        df['pIC50'] = df.apply(convert_to_pic50, axis=1)
        
        # Remove invalid values
        df = df.dropna(subset=['pIC50'])
        
        logger.info(f"Standardized {len(df)} activity values to pIC50")
        return df
    
    def filter_by_activity_threshold(self, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
        """
        Filter data by activity threshold
        
        Args:
            df: DataFrame with bioactivity data
            threshold: pIC50 threshold
            
        Returns:
            Filtered DataFrame
        """
        filtered_df = df[df['pIC50'] >= threshold].copy()
        logger.info(f"Filtered to {len(filtered_df)} data points with pIC50 >= {threshold}")
        return filtered_df
    
    def create_expanded_datasets(self, similar_proteins: Dict[str, Dict[str, List[str]]]) -> Dict[str, pd.DataFrame]:
        """
        Create expanded datasets for each similarity level
        
        Args:
            similar_proteins: Dictionary of similar proteins by threshold
            
        Returns:
            Dictionary of expanded datasets by threshold
        """
        logger.info("Creating expanded datasets")
        
        # Connect to Papyrus database
        conn = self.connect_to_papyrus_db()
        
        expanded_datasets = {}
        
        for threshold_name, proteins_dict in similar_proteins.items():
            logger.info(f"Creating dataset for {threshold_name} threshold")
            
            # Collect all protein IDs for this threshold
            all_protein_ids = set()
            for query_protein, similar_list in proteins_dict.items():
                all_protein_ids.add(query_protein)  # Add original avoidome protein
                all_protein_ids.update(similar_list)  # Add similar proteins
            
            # Convert to list
            protein_ids = list(all_protein_ids)
            
            # Collect bioactivity data
            bioactivity_df = self.collect_bioactivity_data(conn, protein_ids)
            
            if bioactivity_df.empty:
                logger.warning(f"No bioactivity data found for {threshold_name} threshold")
                continue
            
            # Standardize activity values
            bioactivity_df = self.standardize_activity_values(bioactivity_df)
            
            # Filter by activity threshold
            bioactivity_df = self.filter_by_activity_threshold(bioactivity_df, self.activity_thresholds['high_activity'])
            
            # Add threshold information
            bioactivity_df['similarity_threshold'] = threshold_name
            bioactivity_df['threshold_value'] = self.get_threshold_value(threshold_name)
            
            # Save dataset
            dataset_file = self.datasets_dir / f"expanded_dataset_{threshold_name}.csv"
            bioactivity_df.to_csv(dataset_file, index=False)
            
            expanded_datasets[threshold_name] = bioactivity_df
            
            logger.info(f"Created {threshold_name} dataset: {len(bioactivity_df)} data points")
        
        conn.close()
        return expanded_datasets
    
    def get_threshold_value(self, threshold_name: str) -> float:
        """Get numeric threshold value for threshold name"""
        threshold_values = {
            'high': 70.0,
            'medium': 50.0,
            'low': 30.0
        }
        return threshold_values.get(threshold_name, 0.0)
    
    def create_dataset_statistics(self, expanded_datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create statistics for expanded datasets
        
        Args:
            expanded_datasets: Dictionary of expanded datasets
            
        Returns:
            DataFrame with dataset statistics
        """
        logger.info("Creating dataset statistics")
        
        stats_data = []
        
        for threshold_name, df in expanded_datasets.items():
            if df.empty:
                continue
            
            # Basic statistics
            num_proteins = df['uniprot_id'].nunique()
            num_compounds = df['compound_id'].nunique()
            num_data_points = len(df)
            
            # Activity statistics
            activity_stats = df['pIC50'].describe()
            
            # Organism distribution
            organism_counts = df['organism'].value_counts()
            
            stats_data.append({
                'threshold': threshold_name,
                'threshold_value': self.get_threshold_value(threshold_name),
                'num_proteins': num_proteins,
                'num_compounds': num_compounds,
                'num_data_points': num_data_points,
                'mean_pIC50': activity_stats['mean'],
                'std_pIC50': activity_stats['std'],
                'min_pIC50': activity_stats['min'],
                'max_pIC50': activity_stats['max'],
                'human_data_points': organism_counts.get('Homo sapiens', 0),
                'mouse_data_points': organism_counts.get('Mus musculus', 0),
                'rat_data_points': organism_counts.get('Rattus norvegicus', 0)
            })
        
        stats_df = pd.DataFrame(stats_data)
        
        # Save statistics
        stats_file = self.statistics_dir / "dataset_statistics.csv"
        stats_df.to_csv(stats_file, index=False)
        
        logger.info(f"Saved dataset statistics to {stats_file}")
        return stats_df
    
    def create_visualizations(self, expanded_datasets: Dict[str, pd.DataFrame], stats_df: pd.DataFrame):
        """
        Create visualizations for expanded datasets
        
        Args:
            expanded_datasets: Dictionary of expanded datasets
            stats_df: Dataset statistics DataFrame
        """
        logger.info("Creating visualizations")
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Dataset size comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Number of data points
        axes[0, 0].bar(stats_df['threshold'], stats_df['num_data_points'])
        axes[0, 0].set_title('Number of Data Points by Threshold')
        axes[0, 0].set_ylabel('Data Points')
        
        # Number of proteins
        axes[0, 1].bar(stats_df['threshold'], stats_df['num_proteins'])
        axes[0, 1].set_title('Number of Proteins by Threshold')
        axes[0, 1].set_ylabel('Proteins')
        
        # Number of compounds
        axes[1, 0].bar(stats_df['threshold'], stats_df['num_compounds'])
        axes[1, 0].set_title('Number of Compounds by Threshold')
        axes[1, 0].set_ylabel('Compounds')
        
        # Activity distribution
        for threshold_name, df in expanded_datasets.items():
            if not df.empty:
                axes[1, 1].hist(df['pIC50'], alpha=0.7, label=f'{threshold_name} (n={len(df)})', bins=20)
        
        axes[1, 1].set_title('Activity Distribution by Threshold')
        axes[1, 1].set_xlabel('pIC50')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'dataset_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Activity distribution violin plot
        if expanded_datasets:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            plot_data = []
            for threshold_name, df in expanded_datasets.items():
                if not df.empty:
                    df_plot = df[['pIC50']].copy()
                    df_plot['threshold'] = threshold_name
                    plot_data.append(df_plot)
            
            if plot_data:
                combined_df = pd.concat(plot_data, ignore_index=True)
                sns.violinplot(data=combined_df, x='threshold', y='pIC50', ax=ax)
                ax.set_title('Activity Distribution by Similarity Threshold')
                ax.set_xlabel('Similarity Threshold')
                ax.set_ylabel('pIC50')
                
                plt.tight_layout()
                plt.savefig(self.plots_dir / 'activity_distribution.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # 3. Organism distribution
        if not stats_df.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            organism_cols = ['human_data_points', 'mouse_data_points', 'rat_data_points']
            organism_data = stats_df[['threshold'] + organism_cols].set_index('threshold')
            
            organism_data.plot(kind='bar', stacked=True, ax=ax)
            ax.set_title('Data Points by Organism and Threshold')
            ax.set_xlabel('Similarity Threshold')
            ax.set_ylabel('Data Points')
            ax.legend(['Human', 'Mouse', 'Rat'])
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'organism_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Visualizations saved to {self.plots_dir}")
    
    def run_data_collection(self) -> Dict[str, any]:
        """
        Run the complete data collection workflow
        
        Returns:
            Dictionary with results
        """
        logger.info("Starting AQSE data collection strategy")
        
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
        stats_df = self.create_dataset_statistics(expanded_datasets)
        
        # Create visualizations
        self.create_visualizations(expanded_datasets, stats_df)
        
        logger.info("Data collection strategy completed successfully")
        
        return {
            'expanded_datasets': expanded_datasets,
            'statistics': stats_df,
            'similar_proteins': similar_proteins
        }

def main():
    """Main function to run data collection strategy"""
    
    # Set up paths
    similarity_results_dir = "/home/serramelendezcsm/RA/Avoidome/analyses/avoidome_qsar_similarity_expansion/similarity_search"
    papyrus_db_path = "/path/to/papyrus.db"  # Update this path
    output_dir = "/home/serramelendezcsm/RA/Avoidome/analyses/avoidome_qsar_similarity_expansion/data_collection"
    
    # Initialize data collection class
    collector = DataCollectionStrategy(similarity_results_dir, papyrus_db_path, output_dir)
    
    # Run data collection
    results = collector.run_data_collection()
    
    if results:
        print("\n" + "="*60)
        print("AQSE DATA COLLECTION STRATEGY COMPLETED")
        print("="*60)
        print(f"Expanded datasets created: {len(results['expanded_datasets'])}")
        print(f"Total statistics: {len(results['statistics'])} thresholds")
        
        for threshold, df in results['expanded_datasets'].items():
            print(f"  {threshold}: {len(df)} data points")
        print("="*60)
    else:
        print("Data collection failed - check logs for details")

if __name__ == "__main__":
    main()