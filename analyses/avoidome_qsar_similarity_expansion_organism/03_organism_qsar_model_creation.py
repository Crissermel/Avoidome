#!/usr/bin/env python3
"""
AQSE-org Workflow - Step 3: Organism-Specific QSAR Model Creation

This script creates QSAR models for each organism using similar proteins
and organism-specific bioactivity data from Papyrus.

Author: AQSE-org Pipeline
Date: 2025
"""

import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pickle
import json
from datetime import datetime
import hashlib

# Machine Learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# Molecular descriptors
from rdkit import Chem
from rdkit.Chem import AllChem

# Import custom descriptor functions
import sys
sys.path.append('/home/serramelendezcsm/RA/Avoidome/analyses')
from physicochemical_descriptors import calculate_physicochemical_descriptors
try:
    from run_esm_embeddings.single_esmc_embeddings import get_single_esmc_embedding
    ESM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ESM not available: {e}")
    ESM_AVAILABLE = False
    def get_single_esmc_embedding(protein_sequence, return_per_residue=False):
        """Dummy function when ESM is not available"""
        return np.zeros(1280)  # Return dummy embedding

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OrganismQSARModelCreation:
    """Handles organism-specific QSAR model creation for AQSE-org workflow"""
    
    def __init__(self, output_dir: str, organism_mapping_file: str, similarity_file: str):
        """
        Initialize the organism QSAR model creation class
        
        Args:
            output_dir: Output directory for results and cached fingerprints
            organism_mapping_file: Path to organism mappings summary
            similarity_file: Path to similarity search summary
        """
        self.output_dir = Path(output_dir)
        self.organism_mapping_file = Path(organism_mapping_file)
        self.similarity_file = Path(similarity_file)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.fingerprints_dir = self.output_dir / "morgan_fingerprints"
        self.fingerprints_dir.mkdir(exist_ok=True)
        
        self.models_dir = self.output_dir / "qsar_models"
        self.models_dir.mkdir(exist_ok=True)
        
        self.results_dir = self.output_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        self.descriptors_dir = self.output_dir / "descriptors_cache"
        self.descriptors_dir.mkdir(exist_ok=True)
        
        # Organisms to process
        self.organisms = ['human', 'mouse', 'rat']
        
        # Load organism mappings
        self.organism_mappings = self.load_organism_mappings()
        
        # Load similarity search results
        self.similarity_results = self.load_similarity_results()
        
        # Initialize results storage
        self.results = {}
        
        logger.info(f"Initialized organism-specific QSAR model creation")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Loaded mappings for {len(self.organism_mappings)} organisms")
    
    def load_organism_mappings(self) -> Dict[str, List[Dict[str, str]]]:
        """Load organism-specific protein mappings"""
        logger.info("Loading organism-specific protein mappings")
        
        try:
            df = pd.read_csv(self.organism_mapping_file)
            
            # Group by organism
            organism_mappings = {}
            for organism in self.organisms:
                org_data = df[df['organism'] == organism]
                mappings = []
                for _, row in org_data.iterrows():
                    mappings.append({
                        'avoidome_name': row['avoidome_name'],
                        'avoidome_uniprot_id': row['avoidome_uniprot_id'],
                        'organism_uniprot_id': row['organism_uniprot_id'],
                        'protein_name': row['protein_name']
                    })
                organism_mappings[organism] = mappings
                logger.info(f"Loaded {len(mappings)} {organism} protein mappings")
            
            return organism_mappings
            
        except Exception as e:
            logger.error(f"Error loading organism mappings: {e}")
            return {}
    
    def load_similarity_results(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Load similarity search results for all organisms"""
        logger.info("Loading similarity search results")
        
        if not self.similarity_file.exists():
            logger.error(f"Similarity results not found: {self.similarity_file}")
            return {}
        
        try:
            df = pd.read_csv(self.similarity_file)
            
            # Convert to nested dictionary structure by organism
            similarity_results = {}
            for organism in self.organisms:
                similarity_results[organism] = {}
                org_data = df[df['organism'] == organism]
                
                for _, row in org_data.iterrows():
                    target = row['query_protein']
                    threshold = row['threshold']
                    num_similar = row['num_similar_proteins']
                    similar_proteins = row['similar_proteins']
                    
                    if target not in similarity_results[organism]:
                        similarity_results[organism][target] = {}
                    
                    similarity_results[organism][target][threshold] = {
                        'num_similar': num_similar,
                        'similar_proteins': similar_proteins
                    }
            
            logger.info(f"Loaded similarity results for {len(similarity_results)} organisms")
            return similarity_results
            
        except Exception as e:
            logger.error(f"Error loading similarity results: {e}")
            return {}
    
    def step_0_calculate_morgan_fingerprints(self) -> Dict[str, Any]:
        """
        Step 0: Calculate and save Morgan fingerprints for all compounds in Papyrus database
        """
        logger.info("Step 0: Calculating Morgan fingerprints for all Papyrus compounds")
        
        try:
            # Import Papyrus dataset
            from papyrus_scripts import PapyrusDataset
            
            # Load Papyrus dataset
            logger.info("Loading Papyrus dataset...")
            papyrus_data = PapyrusDataset(version='latest', plusplus=True)
            papyrus_df = papyrus_data.to_dataframe()
            
            logger.info(f"Loaded {len(papyrus_df)} total activities from Papyrus")
            
            # Filter for valid SMILES
            valid_data = papyrus_df.dropna(subset=['SMILES'])
            valid_data = valid_data[valid_data['SMILES'] != '']
            
            # Get unique SMILES
            unique_smiles = valid_data['SMILES'].unique()
            logger.info(f"Found {len(unique_smiles)} unique SMILES")
            
            # Calculate Morgan fingerprints
            fingerprints = {}
            valid_count = 0
            
            for i, smiles in enumerate(unique_smiles):
                if i % 10000 == 0:
                    logger.info(f"Processing SMILES {i+1}/{len(unique_smiles)}")
                
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        # Morgan fingerprints (ECFP4) - radius 2, bit length 2048
                        morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                        morgan_array = np.array(morgan_fp, dtype=np.float32)
                        fingerprints[smiles] = morgan_array
                        valid_count += 1
                except Exception as e:
                    logger.warning(f"Error processing SMILES {smiles}: {e}")
                    continue
            
            # Save fingerprints
            fingerprints_file = self.fingerprints_dir / "papyrus_morgan_fingerprints.pkl"
            with open(fingerprints_file, 'wb') as f:
                pickle.dump(fingerprints, f)
            
            logger.info(f"Step 0 completed: {valid_count} valid Morgan fingerprints saved to {fingerprints_file}")
            
            return {
                'total_smiles': len(unique_smiles),
                'valid_fingerprints': valid_count,
                'fingerprints_file': str(fingerprints_file)
            }
            
        except Exception as e:
            logger.error(f"Error in Step 0: {e}")
            return {}
    
    def load_morgan_fingerprints(self) -> Dict[str, np.ndarray]:
        """Load pre-computed Morgan fingerprints"""
        fingerprints_file = self.fingerprints_dir / "papyrus_morgan_fingerprints.pkl"
        
        if not fingerprints_file.exists():
            logger.error(f"Morgan fingerprints file not found: {fingerprints_file}")
            return {}
        
        try:
            with open(fingerprints_file, 'rb') as f:
                fingerprints = pickle.load(f)
            logger.info(f"Loaded {len(fingerprints)} Morgan fingerprints")
            return fingerprints
        except Exception as e:
            logger.error(f"Error loading Morgan fingerprints: {e}")
            return {}
    
    def has_similar_proteins(self, organism: str, uniprot_id: str) -> bool:
        """Check if protein has similar proteins in any threshold"""
        if organism not in self.similarity_results:
            return False
        
        if uniprot_id not in self.similarity_results[organism]:
            return False
        
        # Check if there are similar proteins in any threshold
        for threshold in ['high', 'medium', 'low']:
            if threshold in self.similarity_results[organism][uniprot_id]:
                num_similar = self.similarity_results[organism][uniprot_id][threshold]['num_similar']
                # If num_similar > 1, it means there are other proteins besides itself
                if num_similar > 1:
                    return True
        
        return False
    
    def get_similar_proteins(self, organism: str, uniprot_id: str) -> List[str]:
        """Get list of similar proteins for a target protein (union of all thresholds)"""
        if organism not in self.similarity_results:
            return []
        
        if uniprot_id not in self.similarity_results[organism]:
            return []
        
        similar_proteins = set()
        
        # Check all thresholds and collect similar proteins
        for threshold in ['high', 'medium', 'low']:
            if threshold in self.similarity_results[organism][uniprot_id]:
                threshold_data = self.similarity_results[organism][uniprot_id][threshold]
                if isinstance(threshold_data, dict) and 'similar_proteins' in threshold_data:
                    # Parse similar proteins from the string format
                    similar_str = threshold_data['similar_proteins']
                    if similar_str and similar_str.strip():
                        # Parse format like "P05177_WT (100.0%), P24460_WT (78.0%)"
                        for protein_info in similar_str.split(','):
                            protein_info = protein_info.strip()
                            if '_WT' in protein_info:
                                # Extract protein ID and remove _WT suffix
                                protein_id = protein_info.split('_WT')[0]
                                if protein_id != uniprot_id:  # Don't include the target itself
                                    similar_proteins.add(protein_id)
        
        return list(similar_proteins)
    
    def get_similar_proteins_for_threshold(self, organism: str, uniprot_id: str, threshold: str) -> List[str]:
        """Get list of similar proteins for a target protein at a specific threshold"""
        if organism not in self.similarity_results:
            return []
        
        if uniprot_id not in self.similarity_results[organism]:
            return []
        
        if threshold not in self.similarity_results[organism][uniprot_id]:
            return []
        
        similar_proteins = []
        threshold_data = self.similarity_results[organism][uniprot_id][threshold]
        
        if isinstance(threshold_data, dict) and 'similar_proteins' in threshold_data:
            similar_str = threshold_data['similar_proteins']
            if similar_str and similar_str.strip():
                # Parse format like "P05177_WT (100.0%), P24460_WT (78.0%)"
                for protein_info in similar_str.split(','):
                    protein_info = protein_info.strip()
                    if '_WT' in protein_info:
                        # Extract protein ID and remove _WT suffix
                        protein_id = protein_info.split('_WT')[0]
                        if protein_id != uniprot_id:  # Don't include the target itself
                            similar_proteins.append(protein_id)
        
        return similar_proteins
    
    def load_bioactivity_data_for_protein(self, uniprot_id: str) -> pd.DataFrame:
        """Load bioactivity data for a specific protein from Papyrus"""
        logger.info(f"Loading bioactivity data for protein {uniprot_id}")
        
        try:
            # Import Papyrus dataset
            from papyrus_scripts import PapyrusDataset
            
            # Load Papyrus dataset
            papyrus_data = PapyrusDataset(version='latest', plusplus=True)
            papyrus_df = papyrus_data.to_dataframe()
            
            # Filter for the specific protein
            protein_data = papyrus_df[papyrus_df['accession'] == uniprot_id]
            
            # Filter for valid data
            valid_data = protein_data.dropna(subset=['SMILES', 'pchembl_value_Mean'])
            valid_data = valid_data[valid_data['SMILES'] != '']
            
            logger.info(f"Found {len(valid_data)} bioactivity records for {uniprot_id}")
            return valid_data
            
        except Exception as e:
            logger.error(f"Error loading bioactivity data for {uniprot_id}: {e}")
            return pd.DataFrame()
    
    def load_bioactivity_data_for_proteins(self, uniprot_ids: List[str]) -> pd.DataFrame:
        """Load bioactivity data for multiple proteins from Papyrus"""
        logger.info(f"Loading bioactivity data for {len(uniprot_ids)} proteins")
        
        try:
            # Import Papyrus dataset
            from papyrus_scripts import PapyrusDataset
            
            # Load Papyrus dataset
            papyrus_data = PapyrusDataset(version='latest', plusplus=True)
            papyrus_df = papyrus_data.to_dataframe()
            
            # Filter for the specified proteins
            protein_data = papyrus_df[papyrus_df['accession'].isin(uniprot_ids)]
            
            # Filter for valid data
            valid_data = protein_data.dropna(subset=['SMILES', 'pchembl_value_Mean'])
            valid_data = valid_data[valid_data['SMILES'] != '']
            
            logger.info(f"Found {len(valid_data)} bioactivity records for {len(uniprot_ids)} proteins")
            return valid_data
            
        except Exception as e:
            logger.error(f"Error loading bioactivity data for proteins {uniprot_ids}: {e}")
            return pd.DataFrame()
    
    def load_protein_sequence(self, uniprot_id: str) -> str:
        """Load protein sequence from Papyrus database"""
        logger.info(f"Loading protein sequence for {uniprot_id}")
        
        try:
            # Import Papyrus dataset
            from papyrus_scripts import PapyrusDataset
            
            # Load Papyrus dataset
            papyrus_data = PapyrusDataset(version='latest', plusplus=True)
            
            # Get protein sequences from Papyrus
            proteins_df = papyrus_data.proteins().to_dataframe()
            
            # Look for the protein with _WT suffix
            papyrus_target_id = f"{uniprot_id}_WT"
            protein_row = proteins_df[proteins_df['target_id'] == papyrus_target_id]
            
            if not protein_row.empty:
                sequence = protein_row.iloc[0]['Sequence']
                logger.info(f"Found sequence for {uniprot_id}: {len(sequence)} amino acids")
                return sequence
            else:
                logger.warning(f"Sequence not found in Papyrus for {uniprot_id} (looked for {papyrus_target_id})")
                return ""
                
        except Exception as e:
            logger.error(f"Error loading sequence for {uniprot_id}: {e}")
            return ""
    
    def get_descriptors_cache_file(self, organism: str, uniprot_id: str) -> Path:
        """Get the cache file path for descriptors"""
        return self.descriptors_dir / f"{organism}_{uniprot_id}_descriptors.pkl"
    
    def are_descriptors_cached(self, organism: str, uniprot_id: str) -> bool:
        """Check if descriptors are already cached for a protein"""
        cache_file = self.get_descriptors_cache_file(organism, uniprot_id)
        return cache_file.exists()
    
    def load_cached_descriptors(self, organism: str, uniprot_id: str) -> Optional[pd.DataFrame]:
        """Load cached descriptors for a protein"""
        cache_file = self.get_descriptors_cache_file(organism, uniprot_id)
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                descriptors = pickle.load(f)
            logger.info(f"Loaded cached descriptors for {organism} {uniprot_id}: {len(descriptors)} samples")
            return descriptors
        except Exception as e:
            logger.warning(f"Error loading cached descriptors for {organism} {uniprot_id}: {e}")
            return None
    
    def save_descriptors_cache(self, organism: str, uniprot_id: str, descriptors: pd.DataFrame) -> None:
        """Save descriptors to cache"""
        cache_file = self.get_descriptors_cache_file(organism, uniprot_id)
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(descriptors, f)
            logger.info(f"Saved descriptors cache for {organism} {uniprot_id}: {len(descriptors)} samples")
        except Exception as e:
            logger.warning(f"Error saving descriptors cache for {organism} {uniprot_id}: {e}")
    
    def create_qsar_model_with_threshold(self, organism: str, target_name: str, uniprot_id: str, threshold: str) -> Dict[str, Any]:
        """Create QSAR model for a protein using similar proteins from a specific threshold"""
        logger.info(f"Creating QSAR model with {threshold} threshold for {organism} {target_name} ({uniprot_id})")
        
        try:
            # Get similar proteins for specific threshold
            similar_proteins = self.get_similar_proteins_for_threshold(organism, uniprot_id, threshold)
            if not similar_proteins:
                logger.warning(f"No similar proteins found for {organism} {uniprot_id} at {threshold} threshold")
                return {}
            
            logger.info(f"Found {len(similar_proteins)} similar proteins at {threshold} threshold: {similar_proteins}")
            
            # Load bioactivity data for similar proteins
            similar_proteins_data = self.load_bioactivity_data_for_proteins(similar_proteins)
            if len(similar_proteins_data) < 30:
                logger.warning(f"Insufficient training data from similar proteins at {threshold} threshold: {len(similar_proteins_data)} samples")
                return {}
            
            # Load bioactivity data for target protein
            target_data = self.load_bioactivity_data_for_protein(uniprot_id)
            if len(target_data) < 20:  # Need at least 20 samples for 80/20 split
                logger.warning(f"Insufficient target protein data: {len(target_data)} samples (need at least 20)")
                return {}
            
            # Split target protein data: 80% train, 20% test
            target_train, target_test = train_test_split(
                target_data, 
                test_size=0.2, 
                random_state=42,
                stratify=None  # No stratification needed for regression
            )
            
            logger.info(f"Target protein data split: {len(target_train)} train, {len(target_test)} test")
            
            # Check if we have enough test data
            if len(target_test) < 5:
                logger.warning(f"Insufficient test data after split: {len(target_test)} samples")
                return {}
            
            # Load Morgan fingerprints
            morgan_fingerprints = self.load_morgan_fingerprints()
            if not morgan_fingerprints:
                logger.error("Failed to load Morgan fingerprints")
                return {}
            
            # Load target protein sequence for ESM descriptors
            protein_sequence = self.load_protein_sequence(uniprot_id)
            if not protein_sequence:
                logger.error(f"Failed to load protein sequence for {uniprot_id}")
                return {}
            
            # Prepare training data (similar proteins + target protein train split)
            train_descriptors = []
            
            # Process similar proteins data
            for _, row in similar_proteins_data.iterrows():
                smiles = row['SMILES']
                if smiles in morgan_fingerprints:
                    try:
                        physico_desc = calculate_physicochemical_descriptors(smiles, include_sasa=False, verbose=False)
                        
                        descriptor_row = {
                            'SMILES': smiles,
                            'pchembl_value_Mean': row['pchembl_value_Mean'],
                            'accession': row['accession'],
                            'molecular_weight': physico_desc['Molecular_Weight'],
                            'logp': physico_desc['ALogP'],
                            'tpsa': physico_desc['Molecular_Polar_Surface_Area'],
                            'hbd': physico_desc['Num_H_Donors'],
                            'hba': physico_desc['Num_H_Acceptors'],
                            'rotatable_bonds': physico_desc['Num_Rotatable_Bonds'],
                            'aromatic_rings': physico_desc['Num_Aromatic_Rings'],
                            'heavy_atoms': physico_desc['Num_Heavy_Atoms'],
                            'num_rings': physico_desc['Num_Rings'],
                            'formal_charge': physico_desc['Formal_Charge'],
                            'log_solubility': physico_desc['LogS'],
                            'molecular_surface_area': physico_desc['Molecular_Surface_Area'],
                            'num_atoms': physico_desc['Num_Atoms'],
                            'num_saturated_rings': physico_desc['Num_Saturated_Rings']
                        }
                        
                        # Add Morgan fingerprint bits
                        morgan_fp = morgan_fingerprints[smiles]
                        for j, bit in enumerate(morgan_fp):
                            descriptor_row[f'morgan_bit_{j}'] = int(bit)
                        
                        train_descriptors.append(descriptor_row)
                        
                    except Exception as e:
                        logger.warning(f"Error calculating descriptors for {smiles}: {e}")
                        continue
            
            # Process target protein training data
            for _, row in target_train.iterrows():
                smiles = row['SMILES']
                if smiles in morgan_fingerprints:
                    try:
                        physico_desc = calculate_physicochemical_descriptors(smiles, include_sasa=False, verbose=False)
                        
                        descriptor_row = {
                            'SMILES': smiles,
                            'pchembl_value_Mean': row['pchembl_value_Mean'],
                            'accession': row['accession'],
                            'molecular_weight': physico_desc['Molecular_Weight'],
                            'logp': physico_desc['ALogP'],
                            'tpsa': physico_desc['Molecular_Polar_Surface_Area'],
                            'hbd': physico_desc['Num_H_Donors'],
                            'hba': physico_desc['Num_H_Acceptors'],
                            'rotatable_bonds': physico_desc['Num_Rotatable_Bonds'],
                            'aromatic_rings': physico_desc['Num_Aromatic_Rings'],
                            'heavy_atoms': physico_desc['Num_Heavy_Atoms'],
                            'num_rings': physico_desc['Num_Rings'],
                            'formal_charge': physico_desc['Formal_Charge'],
                            'log_solubility': physico_desc['LogS'],
                            'molecular_surface_area': physico_desc['Molecular_Surface_Area'],
                            'num_atoms': physico_desc['Num_Atoms'],
                            'num_saturated_rings': physico_desc['Num_Saturated_Rings']
                        }
                        
                        # Add Morgan fingerprint bits
                        morgan_fp = morgan_fingerprints[smiles]
                        for j, bit in enumerate(morgan_fp):
                            descriptor_row[f'morgan_bit_{j}'] = int(bit)
                        
                        train_descriptors.append(descriptor_row)
                        
                    except Exception as e:
                        logger.warning(f"Error calculating descriptors for {smiles}: {e}")
                        continue
            
            # Prepare test data (target protein holdout)
            test_descriptors = []
            for _, row in target_test.iterrows():
                smiles = row['SMILES']
                if smiles in morgan_fingerprints:
                    try:
                        physico_desc = calculate_physicochemical_descriptors(smiles, include_sasa=False, verbose=False)
                        
                        descriptor_row = {
                            'SMILES': smiles,
                            'pchembl_value_Mean': row['pchembl_value_Mean'],
                            'accession': row['accession'],
                            'molecular_weight': physico_desc['Molecular_Weight'],
                            'logp': physico_desc['ALogP'],
                            'tpsa': physico_desc['Molecular_Polar_Surface_Area'],
                            'hbd': physico_desc['Num_H_Donors'],
                            'hba': physico_desc['Num_H_Acceptors'],
                            'rotatable_bonds': physico_desc['Num_Rotatable_Bonds'],
                            'aromatic_rings': physico_desc['Num_Aromatic_Rings'],
                            'heavy_atoms': physico_desc['Num_Heavy_Atoms'],
                            'num_rings': physico_desc['Num_Rings'],
                            'formal_charge': physico_desc['Formal_Charge'],
                            'log_solubility': physico_desc['LogS'],
                            'molecular_surface_area': physico_desc['Molecular_Surface_Area'],
                            'num_atoms': physico_desc['Num_Atoms'],
                            'num_saturated_rings': physico_desc['Num_Saturated_Rings']
                        }
                        
                        # Add Morgan fingerprint bits
                        morgan_fp = morgan_fingerprints[smiles]
                        for j, bit in enumerate(morgan_fp):
                            descriptor_row[f'morgan_bit_{j}'] = int(bit)
                        
                        test_descriptors.append(descriptor_row)
                        
                    except Exception as e:
                        logger.warning(f"Error calculating descriptors for {smiles}: {e}")
                        continue
            
            if len(train_descriptors) < 30 or len(test_descriptors) < 5:
                logger.warning(f"Insufficient data: train={len(train_descriptors)}, test={len(test_descriptors)}")
                return {}
            
            # Convert to DataFrames
            train_df = pd.DataFrame(train_descriptors)
            test_df = pd.DataFrame(test_descriptors)
            
            # Calculate ESM C descriptors for target protein
            try:
                esm_result = get_single_esmc_embedding(
                    protein_sequence=protein_sequence,
                    model_name="esmc_300m",
                    return_per_residue=False,
                    verbose=False
                )
                
                # Extract embedding vector
                if isinstance(esm_result, dict):
                    esm_embedding = esm_result['embedding']
                else:
                    esm_embedding = esm_result
                
                # Ensure embedding is a 1D numpy array
                if hasattr(esm_embedding, 'shape') and len(esm_embedding.shape) > 1:
                    esm_embedding = np.mean(esm_embedding, axis=0)
                elif hasattr(esm_embedding, 'flatten'):
                    esm_embedding = esm_embedding.flatten()
                
                esm_embedding = np.array(esm_embedding)
                
                # Add ESM descriptors to both train and test sets
                for i, val in enumerate(esm_embedding):
                    train_df[f'esm_dim_{i}'] = float(val)
                    test_df[f'esm_dim_{i}'] = float(val)
                
            except Exception as e:
                logger.warning(f"Error calculating ESM descriptors for {uniprot_id}: {e}")
                # Use dummy ESM descriptors
                for i in range(1280):
                    train_df[f'esm_dim_{i}'] = 0.0
                    test_df[f'esm_dim_{i}'] = 0.0
            
            # Prepare features and targets
            feature_cols = [col for col in train_df.columns if col.startswith((
                'morgan_bit_', 'esm_dim_', 'molecular_', 'logp', 'tpsa', 'hbd', 'hba', 
                'rotatable_', 'aromatic_', 'heavy_', 'num_', 'formal_', 'log_'
            ))]
            
            X_train = train_df[feature_cols]
            y_train = train_df['pchembl_value_Mean'].values
            X_test = test_df[feature_cols]
            y_test = test_df['pchembl_value_Mean'].values
            
            # Train Random Forest model on combined data (similar proteins + target train)
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            
            # Make predictions on target protein holdout
            y_test_pred = model.predict(X_test)
            
            # Calculate metrics
            test_r2 = r2_score(y_test, y_test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            test_mae = mean_absolute_error(y_test, y_test_pred)
            
            # Cross-validation on combined training set (similar proteins + target train)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            q2 = cv_scores.mean()
            
            # Print results
            print(f"\n{'='*80}")
            print(f"QSAR MODEL WITH {threshold.upper()} THRESHOLD: {organism.upper()} {target_name} ({uniprot_id})")
            print(f"{'='*80}")
            print(f"Training samples (similar proteins + target train): {len(X_train)}")
            print(f"  - Similar proteins ({threshold}): {len(similar_proteins_data)}")
            print(f"  - Target protein train: {len(target_train)}")
            print(f"Test samples (target protein holdout): {len(X_test)}")
            print(f"Total features: {X_train.shape[1]}")
            print(f"  - Morgan fingerprint bits: {len([col for col in feature_cols if col.startswith('morgan_bit_')])}")
            print(f"  - Physicochemical descriptors: {len([col for col in feature_cols if col.startswith(('molecular_', 'logp', 'tpsa', 'hbd', 'hba', 'rotatable_', 'aromatic_', 'heavy_', 'num_', 'formal_', 'log_'))])}")
            print(f"  - ESM C descriptors: {len([col for col in feature_cols if col.startswith('esm_dim_')])}")
            print(f"Similar proteins used ({threshold}): {similar_proteins}")
            print(f"Test R² (holdout): {test_r2:.4f}")
            print(f"Test RMSE (holdout): {test_rmse:.4f}")
            print(f"Test MAE (holdout): {test_mae:.4f}")
            print(f"Training Q² (CV on combined data): {q2:.4f} ± {cv_scores.std():.4f}")
            print(f"{'='*80}")
            
            # Store results
            results = {
                'organism': organism,
                'target_name': target_name,
                'uniprot_id': uniprot_id,
                'threshold': threshold,
                'model': model,
                'test_r2': test_r2,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'q2': q2,
                'n_train': len(X_train),
                'n_test': len(X_test),
                'n_similar_proteins': len(similar_proteins_data),
                'n_target_train': len(target_train),
                'n_target_test': len(target_test),
                'n_features': X_train.shape[1],
                'similar_proteins': similar_proteins,
                'test_predictions': y_test_pred,
                'test_actual': y_test
            }
            
            # Save model
            model_file = self.models_dir / f"{organism}_{target_name}_{uniprot_id}_{threshold}_model.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            
            # Save predictions
            pred_data = pd.DataFrame({
                'actual': y_test,
                'predicted': y_test_pred
            })
            pred_file = self.results_dir / f"{organism}_{target_name}_{uniprot_id}_{threshold}_predictions.csv"
            pred_data.to_csv(pred_file, index=False)
            
            # Save metrics
            metrics = {
                'organism': organism,
                'target_name': target_name,
                'uniprot_id': uniprot_id,
                'threshold': threshold,
                'test_r2': test_r2,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'q2': q2,
                'n_train': len(X_train),
                'n_test': len(X_test),
                'n_similar_proteins': len(similar_proteins_data),
                'n_target_train': len(target_train),
                'n_target_test': len(target_test),
                'n_features': X_train.shape[1],
                'similar_proteins': similar_proteins
            }
            
            metrics_file = self.results_dir / f"{organism}_{target_name}_{uniprot_id}_{threshold}_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"Successfully created QSAR model with {threshold} threshold for {organism} {target_name}")
            return results
            
        except Exception as e:
            logger.error(f"Error creating QSAR model with {threshold} threshold for {organism} {target_name}: {e}")
            return {}
    
    def export_results_to_csv(self) -> str:
        """Export all model results to CSV format"""
        logger.info("Exporting results to CSV")
        
        csv_data = []
        
        for model_key, model_results in self.results.items():
            # Determine model type and extract information
            if '_high' in model_key or '_medium' in model_key or '_low' in model_key:
                # Threshold model
                parts = model_key.split('_')
                organism = parts[0]
                target_name = '_'.join(parts[1:-2])
                uniprot_id = parts[-2]
                threshold = parts[-1]
                model_type = f"Threshold ({threshold})"
                
                row = {
                    'model_key': model_key,
                    'organism': organism,
                    'target_name': target_name,
                    'uniprot_id': uniprot_id,
                    'model_type': model_type,
                    'threshold': threshold,
                    'r2': model_results.get('test_r2', np.nan),
                    'q2': model_results.get('q2', np.nan),
                    'rmse': model_results.get('test_rmse', np.nan),
                    'mae': model_results.get('test_mae', np.nan),
                    'n_train': model_results.get('n_train', np.nan),
                    'n_test': model_results.get('n_test', np.nan),
                    'n_similar_proteins': model_results.get('n_similar_proteins', np.nan),
                    'n_target_train': model_results.get('n_target_train', np.nan),
                    'n_target_test': model_results.get('n_target_test', np.nan),
                    'n_features': model_results.get('n_features', np.nan),
                    'similar_proteins': str(model_results.get('similar_proteins', [])),
                    'r2_std': np.nan,  # Not available for threshold models
                    'q2_std': np.nan,
                    'rmse_std': np.nan,
                    'mae_std': np.nan
                }
            
            csv_data.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(csv_data)
        csv_file = self.results_dir / "organism_qsar_model_results.csv"
        df.to_csv(csv_file, index=False)
        
        logger.info(f"Results exported to CSV: {csv_file}")
        return str(csv_file)
    
    def run_organism_qsar_workflow(self) -> Dict[str, Any]:
        """Run the complete organism-specific QSAR workflow"""
        logger.info("Starting organism-specific QSAR model creation workflow")
        
        # Step 0: Calculate Morgan fingerprints
        step_0_results = self.step_0_calculate_morgan_fingerprints()
        if not step_0_results:
            logger.error("Step 0 failed - cannot proceed")
            return {}
        
        # Process each organism
        models_created = 0
        proteins_skipped = 0
        
        for organism in self.organisms:
            if organism not in self.organism_mappings or not self.organism_mappings[organism]:
                logger.warning(f"No mappings for {organism}, skipping")
                continue
            
            logger.info(f"Processing {organism} proteins")
            
            for protein_mapping in self.organism_mappings[organism]:
                target_name = protein_mapping['avoidome_name']
                uniprot_id = protein_mapping['organism_uniprot_id']
                
                logger.info(f"Processing {organism} target: {target_name} ({uniprot_id})")
                
                # Check if protein has similar proteins
                if self.has_similar_proteins(organism, uniprot_id):
                    # Create QSAR models for each threshold
                    threshold_models_created = 0
                    for threshold in ['high', 'medium', 'low']:
                        # Check if this threshold has similar proteins
                        threshold_similar_proteins = self.get_similar_proteins_for_threshold(organism, uniprot_id, threshold)
                        if len(threshold_similar_proteins) > 0:
                            # Create model for this threshold
                            results = self.create_qsar_model_with_threshold(organism, target_name, uniprot_id, threshold)
                            if results:
                                self.results[f"{organism}_{target_name}_{uniprot_id}_{threshold}"] = results
                                threshold_models_created += 1
                    
                    if threshold_models_created > 0:
                        models_created += threshold_models_created
                    else:
                        print(f"\n{'='*60}")
                        print(f"SKIPPING: {organism} {target_name} ({uniprot_id})")
                        print(f"REASON: Insufficient data for threshold-specific models")
                        print(f"{'='*60}")
                        proteins_skipped += 1
                else:
                    print(f"\n{'='*60}")
                    print(f"SKIPPING: {organism} {target_name} ({uniprot_id})")
                    print(f"REASON: No similar proteins found")
                    print(f"{'='*60}")
                    proteins_skipped += 1
        
        # Export results to CSV
        if self.results:
            csv_file = self.export_results_to_csv()
            logger.info(f"Results exported to: {csv_file}")
        
        # Create summary
        summary = {
            'step_0_results': step_0_results,
            'models_created': models_created,
            'proteins_skipped': proteins_skipped,
            'total_organisms': len(self.organisms),
            'results': self.results
        }
        
        # Save summary
        summary_file = self.results_dir / "organism_workflow_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'models_created': models_created,
                'proteins_skipped': proteins_skipped,
                'total_organisms': len(self.organisms),
                'step_0_results': step_0_results
            }, f, indent=2)
        
        # Print final summary
        print(f"\n{'='*80}")
        print("AQSE-ORG ORGANISM-SPECIFIC QSAR MODEL CREATION WORKFLOW COMPLETED")
        print(f"{'='*80}")
        print(f"Step 0 - Morgan fingerprints: {step_0_results['valid_fingerprints']} compounds processed")
        print(f"Total organisms: {len(self.organisms)}")
        print(f"Proteins skipped (no similar proteins): {proteins_skipped}")
        print(f"QSAR models created (with similar proteins): {models_created}")
        
        if models_created > 0:
            print(f"\nSUCCESSFUL MODELS:")
            for model_key, model_results in self.results.items():
                if '_high' in model_key or '_medium' in model_key or '_low' in model_key:
                    parts = model_key.split('_')
                    organism = parts[0]
                    threshold = parts[-1]
                    print(f"  {model_key}: Test R²={model_results['test_r2']:.3f}, Q²={model_results['q2']:.3f} ({organism.upper()} {threshold.upper()} threshold model)")
        
        print(f"{'='*80}")
        
        logger.info("Organism-specific QSAR workflow completed successfully")
        return summary

def main():
    """Main function to run organism-specific QSAR model creation"""
    
    # Set up paths
    output_dir = "/home/serramelendezcsm/RA/Avoidome/analyses/avoidome_qsar_similarity_expansion_organism/03_qsar_models"
    organism_mapping_file = "/home/serramelendezcsm/RA/Avoidome/analyses/avoidome_qsar_similarity_expansion_organism/01_organism_mapping/organism_mappings_summary.csv"
    similarity_file = "/home/serramelendezcsm/RA/Avoidome/analyses/avoidome_qsar_similarity_expansion_organism/02_similarity_search/all_organisms_similarity_summary.csv"
    
    # Initialize organism QSAR model creation class
    qsar_creator = OrganismQSARModelCreation(output_dir, organism_mapping_file, similarity_file)
    
    # Run organism QSAR workflow
    results = qsar_creator.run_organism_qsar_workflow()
    
    if results:
        print("\n" + "="*60)
        print("AQSE-ORG ORGANISM-SPECIFIC QSAR MODEL CREATION COMPLETED")
        print("="*60)
        print(f"Models created: {results['models_created']}")
        print(f"Proteins skipped: {results['proteins_skipped']}")
        print(f"Total organisms: {results['total_organisms']}")
        print("="*60)
    else:
        print("Organism QSAR workflow failed - check logs for details")

if __name__ == "__main__":
    main()
