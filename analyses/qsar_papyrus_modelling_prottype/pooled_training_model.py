#!/usr/bin/env python3
"""
Pooled Training QSAR Model

This script implements pooled training for protein groups where:
1. All proteins in the same group are pooled together
2. Target protein data is split into train/test sets
3. Model is trained on pooled data (other proteins + target train set)
4. Model is tested on target protein test set

This approach combines the benefits of:
- More training data from related proteins
- Proper evaluation on target protein
- Same architecture as single protein models

Usage:
    python pooled_training_model.py

Date: 2025-01-30
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import warnings
import joblib
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold
import json

warnings.filterwarnings('ignore')

# Papyrus imports
from papyrus_scripts import PapyrusDataset

# Molecular fingerprinting
from rdkit import Chem
from rdkit.Chem import AllChem

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_modelling_prottype/pooled_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PooledTrainingQSARModel:
    """
    Pooled Training QSAR Model for protein groups
    """
    
    def __init__(self, 
                 original_proteins_path: str = "/home/serramelendezcsm/RA/Avoidome/primary_data/avoidome_prot_list.csv",
                 extended_proteins_path: str = "/home/serramelendezcsm/RA/Avoidome/primary_data/avoidome_prot_list_extended.csv"):
        """
        Initialize the pooled training model
        
        Args:
            original_proteins_path: Path to original protein list
            extended_proteins_path: Path to extended protein list with protein groups
        """
        self.original_proteins_path = original_proteins_path
        self.extended_proteins_path = extended_proteins_path
        self.original_proteins = None
        self.extended_proteins = None
        self.papyrus_data = None
        self.papyrus_df = None
        self.results = {}
        self.single_protein_results = {}
        
    def load_data(self):
        """Load protein lists and Papyrus data"""
        logger.info("Loading protein lists and Papyrus data...")
        
        # Load original protein list
        self.original_proteins = pd.read_csv(self.original_proteins_path)
        logger.info(f"Loaded {len(self.original_proteins)} original proteins")
        
        # Load extended protein list with protein groups
        self.extended_proteins = pd.read_csv(self.extended_proteins_path)
        logger.info(f"Loaded {len(self.extended_proteins)} extended proteins")
        
        # Initialize Papyrus dataset
        try:
            self.papyrus_data = PapyrusDataset(version='latest', plusplus=True)
            self.papyrus_df = self.papyrus_data.to_dataframe()
            logger.info(f"Loaded Papyrus dataset with {len(self.papyrus_df)} records")
        except ImportError:
            logger.error("Papyrus scripts not available. Please install papyrus-scripts package.")
            return False
        
        return True
    
    def get_protein_type(self, protein_name):
        """Get protein type for a given protein name"""
        if self.extended_proteins is None:
            return None
        
        protein_row = self.extended_proteins[self.extended_proteins['name2_entry'] == protein_name]
        if not protein_row.empty:
            return protein_row['prot_group'].iloc[0]
        return None
    
    def get_same_type_proteins(self, protein_type, exclude_protein=None):
        """Get all proteins of the same type, excluding the target protein"""
        if self.extended_proteins is None:
            return []
        
        same_type = self.extended_proteins[self.extended_proteins['prot_group'] == protein_type]
        if exclude_protein:
            same_type = same_type[same_type['name2_entry'] != exclude_protein]
        
        return same_type['name2_entry'].tolist()
    
    def get_protein_activities(self, uniprot_id):
        """Get bioactivity data for a UniProt ID"""
        if self.papyrus_df is None or not uniprot_id or pd.isna(uniprot_id):
            return pd.DataFrame()
        
        activities = self.papyrus_df[self.papyrus_df['accession'] == uniprot_id]
        return activities
    
    def create_morgan_fingerprints(self, smiles_list, radius=2, nBits=2048):
        """Create Morgan fingerprints from SMILES strings"""
        fingerprints = []
        valid_indices = []
        
        for i, smiles in enumerate(smiles_list):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
                    fingerprints.append(np.array(fp))
                    valid_indices.append(i)
            except Exception:
                continue
        
        if not fingerprints:
            return np.array([]), []
        
        return np.array(fingerprints), valid_indices
    
    def prepare_modeling_data(self, activities_df):
        """Prepare data for modeling"""
        if activities_df.empty:
            return None, None
        
        # Clean data
        clean_data = activities_df.dropna(subset=['SMILES', 'pchembl_value'])
        clean_data = clean_data.drop_duplicates(subset=['SMILES', 'pchembl_value'])
        
        if len(clean_data) < 5:
            return None, None
        
        # Process pchembl values
        def process_pchembl_value(value):
            if pd.isna(value):
                return None
            str_value = str(value).strip()
            if ';' in str_value:
                first_value = str_value.split(';')[0].strip()
                try:
                    return float(first_value)
                except ValueError:
                    return None
            else:
                try:
                    return float(str_value)
                except ValueError:
                    return None
        
        clean_data['pchembl_value_processed'] = clean_data['pchembl_value'].apply(process_pchembl_value)
        clean_data = clean_data.dropna(subset=['pchembl_value_processed'])
        
        if len(clean_data) < 5:
            return None, None
        
        # Create fingerprints
        X, valid_indices = self.create_morgan_fingerprints(clean_data['SMILES'].tolist())
        if len(X) == 0:
            return None, None
        
        y = clean_data['pchembl_value_processed'].iloc[valid_indices].values
        
        return X, y
    
    def train_single_protein_model(self, target_protein, test_size=0.2):
        """Train single protein model for comparison"""
        try:
            # Get target protein UniProt ID
            target_row = self.original_proteins[self.original_proteins['Name_2'] == target_protein]
            if target_row.empty:
                return None
            
            target_uniprot = target_row['UniProt ID'].iloc[0]
            
            # Get target protein activities
            target_activities = self.get_protein_activities(target_uniprot)
            if target_activities.empty:
                return None
            
            # Prepare data
            X, y = self.prepare_modeling_data(target_activities)
            if X is None or len(X) < 10:
                return None
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            
            # Predict and evaluate
            y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            return {
                'target_protein': target_protein,
                'status': 'success',
                'n_total_samples': len(X),
                'n_train_samples': len(X_train),
                'n_test_samples': len(X_test),
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'target_uniprot': target_uniprot
            }
            
        except Exception as e:
            logger.error(f"Error training single protein model for {target_protein}: {e}")
            return {
                'target_protein': target_protein,
                'status': 'error',
                'error_message': str(e)
            }
    
    def train_pooled_model(self, target_protein, protein_type, same_type_proteins, test_size=0.2):
        """Train pooled model for target protein"""
        try:
            # Get target protein UniProt ID
            target_row = self.original_proteins[self.original_proteins['Name_2'] == target_protein]
            if target_row.empty:
                return None
            
            target_uniprot = target_row['UniProt ID'].iloc[0]
            
            # Get target protein activities and split
            target_activities = self.get_protein_activities(target_uniprot)
            if target_activities.empty:
                return None
            
            target_X, target_y = self.prepare_modeling_data(target_activities)
            if target_X is None or len(target_X) < 10:
                return None
            
            # Split target protein data
            target_X_train, target_X_test, target_y_train, target_y_test = train_test_split(
                target_X, target_y, test_size=test_size, random_state=42
            )
            
            # Collect training data from same-type proteins
            training_activities = []
            for protein in same_type_proteins:
                protein_row = self.extended_proteins[self.extended_proteins['name2_entry'] == protein]
                if not protein_row.empty:
                    uniprot_id = protein_row['human_uniprot_id'].iloc[0]
                    if pd.isna(uniprot_id):
                        uniprot_id = protein_row['mouse_uniprot_id'].iloc[0] if not pd.isna(protein_row['mouse_uniprot_id'].iloc[0]) else protein_row['rat_uniprot_id'].iloc[0]
                    
                    if not pd.isna(uniprot_id):
                        activities = self.get_protein_activities(uniprot_id)
                        if not activities.empty:
                            training_activities.append(activities)
            
            if not training_activities:
                return {
                    'target_protein': target_protein,
                    'protein_type': protein_type,
                    'status': 'insufficient_training_data',
                    'n_target_samples': len(target_X),
                    'n_same_type_proteins': len(same_type_proteins)
                }
            
            # Combine training data from other proteins
            other_proteins_df = pd.concat(training_activities, ignore_index=True)
            other_X, other_y = self.prepare_modeling_data(other_proteins_df)
            
            if other_X is None or len(other_X) < 10:
                return {
                    'target_protein': target_protein,
                    'protein_type': protein_type,
                    'status': 'insufficient_training_data',
                    'n_target_samples': len(target_X),
                    'n_same_type_proteins': len(same_type_proteins)
                }
            
            # Pool training data: other proteins + target protein train set
            pooled_X_train = np.vstack([other_X, target_X_train])
            pooled_y_train = np.concatenate([other_y, target_y_train])
            
            # Train model on pooled data
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(pooled_X_train, pooled_y_train)
            
            # Test on target protein test set
            target_pred = model.predict(target_X_test)
            
            # Calculate metrics
            mse = mean_squared_error(target_y_test, target_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(target_y_test, target_pred)
            r2 = r2_score(target_y_test, target_pred)
            
            return {
                'target_protein': target_protein,
                'protein_type': protein_type,
                'status': 'success',
                'n_target_samples': len(target_X),
                'n_target_train_samples': len(target_X_train),
                'n_target_test_samples': len(target_X_test),
                'n_other_proteins_samples': len(other_X),
                'n_pooled_train_samples': len(pooled_X_train),
                'n_same_type_proteins': len(same_type_proteins),
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'target_uniprot': target_uniprot,
                'same_type_proteins': same_type_proteins
            }
            
        except Exception as e:
            logger.error(f"Error training pooled model for {target_protein}: {e}")
            return {
                'target_protein': target_protein,
                'protein_type': protein_type,
                'status': 'error',
                'error_message': str(e)
            }
    
    def run_comprehensive_training(self):
        """Run comprehensive training for all proteins"""
        logger.info("Starting comprehensive pooled training...")
        
        pooled_results = []
        single_results = []
        
        total_proteins = len(self.original_proteins)
        
        for i, (_, protein_row) in enumerate(self.original_proteins.iterrows()):
            protein_name = protein_row['Name_2']
            logger.info(f"Processing {protein_name} ({i+1}/{total_proteins})")
            
            # Get protein type
            protein_type = self.get_protein_type(protein_name)
            if not protein_type:
                logger.warning(f"No protein type found for {protein_name}")
                continue
            
            # Get same-type proteins
            same_type_proteins = self.get_same_type_proteins(protein_type, exclude_protein=protein_name)
            
            # Train single protein model
            logger.info(f"Training single protein model for {protein_name}")
            single_result = self.train_single_protein_model(protein_name)
            if single_result:
                single_results.append(single_result)
                if single_result['status'] == 'success':
                    logger.info(f"Single protein model for {protein_name}: R²={single_result['r2']:.3f}, RMSE={single_result['rmse']:.3f}")
            
            # Train pooled model
            logger.info(f"Training pooled model for {protein_name} (type: {protein_type})")
            pooled_result = self.train_pooled_model(protein_name, protein_type, same_type_proteins)
            if pooled_result:
                pooled_results.append(pooled_result)
                if pooled_result['status'] == 'success':
                    logger.info(f"Pooled model for {protein_name}: R²={pooled_result['r2']:.3f}, RMSE={pooled_result['rmse']:.3f}")
        
        # Store results
        self.results = pooled_results
        self.single_protein_results = single_results
        
        # Save results
        self.save_results()
        
        logger.info("Comprehensive training completed!")
        return pooled_results, single_results
    
    def save_results(self):
        """Save results to CSV files"""
        output_dir = Path("/home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_modelling_prottype/pooled_training_results")
        output_dir.mkdir(exist_ok=True)
        
        # Save pooled results
        if self.results:
            pooled_df = pd.DataFrame(self.results)
            pooled_df.to_csv(output_dir / "pooled_training_results.csv", index=False)
            logger.info(f"Saved pooled results to {output_dir / 'pooled_training_results.csv'}")
        
        # Save single protein results
        if self.single_protein_results:
            single_df = pd.DataFrame(self.single_protein_results)
            single_df.to_csv(output_dir / "single_protein_results.csv", index=False)
            logger.info(f"Saved single protein results to {output_dir / 'single_protein_results.csv'}")
        
        # Create comparison
        if self.results and self.single_protein_results:
            self.create_comparison()
    
    def create_comparison(self):
        """Create comparison between pooled and single protein models"""
        output_dir = Path("/home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_modelling_prottype/pooled_training_results")
        
        # Convert to DataFrames
        pooled_df = pd.DataFrame(self.results)
        single_df = pd.DataFrame(self.single_protein_results)
        
        # Filter successful models
        pooled_success = pooled_df[pooled_df['status'] == 'success'].copy()
        single_success = single_df[single_df['status'] == 'success'].copy()
        
        # Merge on target_protein
        comparison = pd.merge(
            pooled_success[['target_protein', 'r2', 'rmse', 'mae', 'protein_type', 'n_pooled_train_samples', 'n_target_test_samples']],
            single_success[['target_protein', 'r2', 'rmse', 'mae', 'n_total_samples', 'n_test_samples']],
            on='target_protein',
            suffixes=('_pooled', '_single')
        )
        
        # Calculate differences
        comparison['r2_difference'] = comparison['r2_pooled'] - comparison['r2_single']
        comparison['rmse_difference'] = comparison['rmse_pooled'] - comparison['rmse_single']
        comparison['mae_difference'] = comparison['mae_pooled'] - comparison['mae_single']
        
        # Save comparison
        comparison.to_csv(output_dir / "pooled_vs_single_comparison.csv", index=False)
        logger.info(f"Saved comparison to {output_dir / 'pooled_vs_single_comparison.csv'}")
        
        # Create summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_proteins': len(comparison),
            'pooled_avg_r2': comparison['r2_pooled'].mean(),
            'single_avg_r2': comparison['r2_single'].mean(),
            'pooled_avg_rmse': comparison['rmse_pooled'].mean(),
            'single_avg_rmse': comparison['rmse_single'].mean(),
            'improved_proteins': len(comparison[comparison['r2_difference'] > 0]),
            'worsened_proteins': len(comparison[comparison['r2_difference'] < 0])
        }
        
        with open(output_dir / "training_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Training summary: {summary}")

def main():
    """Main function to run pooled training"""
    model = PooledTrainingQSARModel()
    
    if not model.load_data():
        logger.error("Failed to load data")
        return
    
    # Run comprehensive training
    pooled_results, single_results = model.run_comprehensive_training()
    
    logger.info("Pooled training completed successfully!")

if __name__ == "__main__":
    main()