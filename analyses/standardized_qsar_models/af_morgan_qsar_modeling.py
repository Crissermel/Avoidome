#!/usr/bin/env python3
"""
AlphaFold + Morgan QSAR Modeling with Physicochemical Descriptors

This script creates QSAR models for each protein in the Avoidome dataset using:
- Morgan fingerprints (2048 bits, radius=2)
- Physicochemical descriptors (calculated using RDKit)
- AlphaFold embeddings (structure-level)
- Papyrus bioactivity data

Models:
- Regression: Predicting pchembl_value (continuous)
- Classification: Binary classification (active if pchembl_value > 7)

Features:
- 5-fold cross-validation
- Minimum 50 samples per protein
- Standardized data preprocessing
- Comprehensive performance metrics
- Model persistence

Author: Generated for Avoidome QSAR modeling
Date: 2025-01-30
"""

import pandas as pd
import numpy as np
import pickle
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Papyrus imports
from papyrus_scripts import PapyrusDataset

# ML imports
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Molecular fingerprinting and descriptors
from rdkit import Chem
from rdkit.Chem import AllChem
import sys
sys.path.append('/home/serramelendezcsm/RA/Avoidome/analyses')
from physicochemical_descriptors import calculate_physicochemical_descriptors

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/serramelendezcsm/RA/Avoidome/analyses/standardized_qsar_models/af_morgan_qsar.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AFMorganQSARModel:
    """
    QSAR model using AlphaFold embeddings, Morgan fingerprints, and physicochemical descriptors
    """
    
    def __init__(self, 
                 data_path: str = "/home/serramelendezcsm/RA/Avoidome/processed_data/papyrus_protein_check_results.csv",
                 sequences_path: str = "/home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_esm_emb/targets_w_sequences.csv",
                 af_embeddings_dir: str = "/home/serramelendezcsm/RA/Avoidome/analyses/af_embeddings/alphafold_embeddings",
                 output_dir: str = "/home/serramelendezcsm/RA/Avoidome/analyses/standardized_qsar_models"):
        """
        Initialize the QSAR model with AlphaFold embeddings
        
        Args:
            data_path: Path to the protein check results CSV file
            sequences_path: Path to the protein sequences CSV file
            af_embeddings_dir: Directory containing AlphaFold embeddings
            output_dir: Directory to save models and results
        """
        self.data_path = data_path
        self.sequences_path = sequences_path
        self.af_embeddings_dir = Path(af_embeddings_dir)
        self.output_dir = Path(output_dir)
        self.papyrus_data = None
        self.proteins_df = None
        self.sequences_df = None
        self.papyrus_df = None
        self.results = []
        
        # Create output directories
        self.regression_dir = self.output_dir / "af_morgan_regression"
        self.classification_dir = self.output_dir / "af_morgan_classification"
        self.regression_dir.mkdir(exist_ok=True)
        self.classification_dir.mkdir(exist_ok=True)
        
    def load_data(self):
        """Load protein data, sequences, and initialize Papyrus dataset"""
        logger.info("Loading protein data, sequences, and Papyrus dataset...")
        
        # Load protein check results
        self.proteins_df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(self.proteins_df)} proteins from {self.data_path}")
        
        # Load protein sequences
        self.sequences_df = pd.read_csv(self.sequences_path)
        logger.info(f"Loaded {len(self.sequences_df)} protein sequences from {self.sequences_path}")
        
        # Initialize Papyrus dataset
        logger.info("Initializing Papyrus dataset...")
        self.papyrus_data = PapyrusDataset(version='latest', plusplus=True)
        logger.info("Papyrus dataset initialized successfully")
        
        # Load full dataset into DataFrame for efficient filtering
        logger.info("Loading full Papyrus dataset into DataFrame...")
        self.papyrus_df = self.papyrus_data.to_dataframe()
        logger.info(f"Loaded {len(self.papyrus_df)} total activities from Papyrus")
        
    def get_protein_activities(self, uniprot_id: str) -> Optional[pd.DataFrame]:
        """
        Retrieve bioactivity data for a given UniProt ID
        
        Args:
            uniprot_id: UniProt ID to retrieve data for
            
        Returns:
            DataFrame with bioactivity data or None if not found
        """
        try:
            if not uniprot_id or pd.isna(uniprot_id):
                return None
                
            # Filter bioactivity data for the specific protein
            activities_df = self.papyrus_df[self.papyrus_df['accession'] == uniprot_id].copy()
            
            if len(activities_df) > 0:
                logger.info(f"Retrieved {len(activities_df)} activities for {uniprot_id}")
                return activities_df
            else:
                logger.warning(f"No activities found for {uniprot_id}")
                return None
            
        except Exception as e:
            logger.error(f"Error retrieving data for {uniprot_id}: {e}")
            return None
    
    def get_protein_sequence(self, uniprot_id: str) -> Optional[str]:
        """
        Get protein sequence for a given UniProt ID
        
        Args:
            uniprot_id: UniProt ID to get sequence for
            
        Returns:
            Protein sequence string or None if not found
        """
        try:
            if not uniprot_id or pd.isna(uniprot_id):
                return None
                
            # Find the protein in sequences_df
            protein_row = self.sequences_df[
                (self.sequences_df['human_uniprot_id'] == uniprot_id) |
                (self.sequences_df['mouse_uniprot_id'] == uniprot_id) |
                (self.sequences_df['rat_uniprot_id'] == uniprot_id)
            ]
            
            if len(protein_row) > 0:
                sequence = protein_row.iloc[0]['sequence']
                logger.info(f"Retrieved sequence for {uniprot_id} (length: {len(sequence)})")
                return sequence
            else:
                logger.warning(f"No sequence found for {uniprot_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving sequence for {uniprot_id}: {e}")
            return None
    
    def load_alphafold_embedding(self, uniprot_id: str) -> Optional[np.ndarray]:
        """
        Load AlphaFold embedding for a protein
        
        Args:
            uniprot_id: UniProt ID to load embedding for
            
        Returns:
            AlphaFold embedding array or None if not found
        """
        try:
            if not uniprot_id or pd.isna(uniprot_id):
                return None
                
            # Path to the single embedding file
            embedding_path = self.af_embeddings_dir / uniprot_id / "embeddings" / "single_embedding.npy"
            
            if not embedding_path.exists():
                logger.warning(f"AlphaFold embedding not found for {uniprot_id} at {embedding_path}")
                return None
            
            # Load the embedding
            embedding = np.load(embedding_path)
            
            # Ensure we get a 1D array
            if embedding.ndim > 1:
                embedding = embedding.flatten()
            
            logger.info(f"Loaded AlphaFold embedding for {uniprot_id} (dimension: {len(embedding)})")
            return embedding
            
        except Exception as e:
            logger.error(f"Error loading AlphaFold embedding for {uniprot_id}: {e}")
            return None
    
    def calculate_morgan_fingerprints(self, smiles_list: List[str]) -> np.ndarray:
        """
        Calculate Morgan fingerprints for a list of SMILES strings
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Array of Morgan fingerprints (n_samples, 2048)
        """
        fingerprints = []
        
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                    fingerprints.append(np.array(fp))
                else:
                    # Add zero vector for invalid SMILES
                    fingerprints.append(np.zeros(2048))
            except Exception as e:
                logger.warning(f"Error calculating fingerprint for {smiles}: {e}")
                fingerprints.append(np.zeros(2048))
        
        return np.array(fingerprints)
    
    def calculate_physicochemical_features(self, smiles_list: List[str]) -> np.ndarray:
        """
        Calculate physicochemical descriptors for a list of SMILES strings
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Array of physicochemical descriptors (n_samples, n_descriptors)
        """
        descriptors_list = []
        
        for smiles in smiles_list:
            try:
                descriptors = calculate_physicochemical_descriptors(smiles, include_sasa=False, verbose=False)
                # Extract only the numerical descriptors (exclude SASA if NaN)
                descriptor_values = [
                    descriptors['ALogP'],
                    descriptors['Molecular_Weight'],
                    descriptors['Num_H_Donors'],
                    descriptors['Num_H_Acceptors'],
                    descriptors['Num_Rotatable_Bonds'],
                    descriptors['Num_Atoms'],
                    descriptors['Num_Rings'],
                    descriptors['Num_Aromatic_Rings'],
                    descriptors['LogS'],
                    descriptors['Molecular_Surface_Area'],
                    descriptors['Molecular_Polar_Surface_Area'],
                    descriptors['Num_Heavy_Atoms'],
                    descriptors['Formal_Charge'],
                    descriptors['Num_Saturated_Rings']
                ]
                descriptors_list.append(descriptor_values)
            except Exception as e:
                logger.warning(f"Error calculating descriptors for {smiles}: {e}")
                # Add NaN values for failed calculations
                descriptors_list.append([np.nan] * 14)
        
        return np.array(descriptors_list)
    
    def prepare_features(self, activities_df: pd.DataFrame, uniprot_id: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features (AlphaFold + Morgan + physicochemical) and targets for modeling
        
        Args:
            activities_df: DataFrame with bioactivity data
            uniprot_id: UniProt ID for the protein
            
        Returns:
            Tuple of (features, targets, smiles_list)
        """
        try:
            # Get SMILES and targets
            smiles_list = activities_df['SMILES'].tolist()
            # Convert pchembl_value to numeric, handling any non-numeric values
            targets = pd.to_numeric(activities_df['pchembl_value'], errors='coerce').values
            # Remove any NaN values
            valid_mask = ~np.isnan(targets)
            targets = targets[valid_mask]
            smiles_list = [smiles_list[i] for i in range(len(smiles_list)) if valid_mask[i]]
            
            # Calculate Morgan fingerprints
            logger.info(f"Calculating Morgan fingerprints for {len(smiles_list)} compounds...")
            morgan_fps = self.calculate_morgan_fingerprints(smiles_list)
            
            # Calculate physicochemical descriptors
            logger.info(f"Calculating physicochemical descriptors for {len(smiles_list)} compounds...")
            physchem_descriptors = self.calculate_physicochemical_features(smiles_list)
            
            # Load AlphaFold embedding
            logger.info(f"Loading AlphaFold embedding for {uniprot_id}...")
            af_embedding = self.load_alphafold_embedding(uniprot_id)
            
            if af_embedding is None:
                logger.error(f"Could not load AlphaFold embedding for {uniprot_id}")
                return None, None, None
            
            # Repeat AlphaFold embedding for all samples
            af_embeddings = np.tile(af_embedding, (len(smiles_list), 1))
            
            # Combine all features
            logger.info("Combining features...")
            features = np.hstack([
                morgan_fps,           # 2048 dimensions
                physchem_descriptors, # 14 dimensions
                af_embeddings        # 384 dimensions
            ])
            
            logger.info(f"Combined features shape: {features.shape}")
            logger.info(f"Feature breakdown: Morgan={morgan_fps.shape[1]}, PhysChem={physchem_descriptors.shape[1]}, AlphaFold={af_embeddings.shape[1]}")
            
            return features, targets, smiles_list
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None, None, None
    
    def train_regression_model(self, X: np.ndarray, y: np.ndarray, protein_name: str) -> Dict:
        """
        Train regression model for continuous target prediction
        
        Args:
            X: Feature matrix
            y: Target values
            protein_name: Name of the protein
            
        Returns:
            Dictionary with model performance metrics
        """
        try:
            logger.info(f"Training regression model for {protein_name}...")
            
            # Handle missing values
            imputer = SimpleImputer(strategy='median')
            X_imputed = imputer.fit_transform(X)
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_imputed)
            
            # Initialize model
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            # 5-fold cross-validation
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = []
            
            for train_idx, val_idx in kf.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Train model
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_val)
                
                # Calculate metrics
                mse = mean_squared_error(y_val, y_pred)
                r2 = r2_score(y_val, y_pred)
                mae = mean_absolute_error(y_val, y_pred)
                
                cv_scores.append({'mse': mse, 'r2': r2, 'mae': mae})
            
            # Train final model on all data
            model.fit(X_scaled, y)
            
            # Calculate average CV scores
            avg_scores = {
                'mse': np.mean([s['mse'] for s in cv_scores]),
                'r2': np.mean([s['r2'] for s in cv_scores]),
                'mae': np.mean([s['mae'] for s in cv_scores])
            }
            
            # Save model and scaler
            model_path = self.regression_dir / f"{protein_name}_model.pkl"
            scaler_path = self.regression_dir / f"{protein_name}_scaler.pkl"
            imputer_path = self.regression_dir / f"{protein_name}_imputer.pkl"
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            with open(imputer_path, 'wb') as f:
                pickle.dump(imputer, f)
            
            logger.info(f"Regression model saved for {protein_name}")
            logger.info(f"CV Performance - MSE: {avg_scores['mse']:.4f}, R²: {avg_scores['r2']:.4f}, MAE: {avg_scores['mae']:.4f}")
            
            return {
                'model_type': 'regression',
                'protein_name': protein_name,
                'n_samples': len(y),
                'cv_scores': avg_scores,
                'model_path': str(model_path),
                'scaler_path': str(scaler_path),
                'imputer_path': str(imputer_path)
            }
            
        except Exception as e:
            logger.error(f"Error training regression model for {protein_name}: {e}")
            return None
    
    def train_classification_model(self, X: np.ndarray, y: np.ndarray, protein_name: str) -> Dict:
        """
        Train classification model for binary target prediction
        
        Args:
            X: Feature matrix
            y: Target values (binary)
            protein_name: Name of the protein
            
        Returns:
            Dictionary with model performance metrics
        """
        try:
            logger.info(f"Training classification model for {protein_name}...")
            
            # Handle missing values
            imputer = SimpleImputer(strategy='median')
            X_imputed = imputer.fit_transform(X)
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_imputed)
            
            # Initialize model
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            # 5-fold cross-validation
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = []
            
            for train_idx, val_idx in kf.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Train model
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_val)
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                
                # Calculate metrics
                accuracy = accuracy_score(y_val, y_pred)
                precision = precision_score(y_val, y_pred, average='weighted')
                recall = recall_score(y_val, y_pred, average='weighted')
                f1 = f1_score(y_val, y_pred, average='weighted')
                auc = roc_auc_score(y_val, y_pred_proba)
                
                cv_scores.append({
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auc': auc
                })
            
            # Train final model on all data
            model.fit(X_scaled, y)
            
            # Calculate average CV scores
            avg_scores = {
                'accuracy': np.mean([s['accuracy'] for s in cv_scores]),
                'precision': np.mean([s['precision'] for s in cv_scores]),
                'recall': np.mean([s['recall'] for s in cv_scores]),
                'f1': np.mean([s['f1'] for s in cv_scores]),
                'auc': np.mean([s['auc'] for s in cv_scores])
            }
            
            # Save model and scaler
            model_path = self.classification_dir / f"{protein_name}_model.pkl"
            scaler_path = self.classification_dir / f"{protein_name}_scaler.pkl"
            imputer_path = self.classification_dir / f"{protein_name}_imputer.pkl"
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            with open(imputer_path, 'wb') as f:
                pickle.dump(imputer, f)
            
            logger.info(f"Classification model saved for {protein_name}")
            logger.info(f"CV Performance - Accuracy: {avg_scores['accuracy']:.4f}, F1: {avg_scores['f1']:.4f}, AUC: {avg_scores['auc']:.4f}")
            
            return {
                'model_type': 'classification',
                'protein_name': protein_name,
                'n_samples': len(y),
                'cv_scores': avg_scores,
                'model_path': str(model_path),
                'scaler_path': str(scaler_path),
                'imputer_path': str(imputer_path)
            }
            
        except Exception as e:
            logger.error(f"Error training classification model for {protein_name}: {e}")
            return None
    
    def process_protein(self, uniprot_id: str, protein_name: str) -> Dict:
        """
        Process a single protein and train both regression and classification models
        
        Args:
            uniprot_id: UniProt ID of the protein
            protein_name: Name of the protein
            
        Returns:
            Dictionary with processing results
        """
        try:
            logger.info(f"Processing protein: {protein_name} ({uniprot_id})")
            
            # Get bioactivity data
            activities_df = self.get_protein_activities(uniprot_id)
            if activities_df is None or len(activities_df) < 50:
                logger.warning(f"Insufficient data for {protein_name} ({len(activities_df) if activities_df is not None else 0} samples)")
                return None
            
            # Prepare features
            X, y, smiles_list = self.prepare_features(activities_df, uniprot_id)
            if X is None:
                logger.error(f"Could not prepare features for {protein_name}")
                return None
            
            # Create binary targets for classification (active if pchembl_value > 7)
            y_binary = (y > 7).astype(int)
            
            # Check if we have both classes for classification
            if len(np.unique(y_binary)) < 2:
                logger.warning(f"Only one class present for {protein_name}, skipping classification")
                classification_result = None
            else:
                # Train classification model
                classification_result = self.train_classification_model(X, y_binary, protein_name)
            
            # Train regression model
            regression_result = self.train_regression_model(X, y, protein_name)
            
            return {
                'uniprot_id': uniprot_id,
                'protein_name': protein_name,
                'n_samples': len(y),
                'regression': regression_result,
                'classification': classification_result
            }
            
        except Exception as e:
            logger.error(f"Error processing protein {protein_name}: {e}")
            return None
    
    def run_modeling(self):
        """Run QSAR modeling for all proteins"""
        try:
            logger.info("Starting QSAR modeling with AlphaFold embeddings...")
            
            # Load data
            self.load_data()
            
            # Get list of proteins to process
            proteins_to_process = []
            for _, row in self.proteins_df.iterrows():
                # Get all possible UniProt IDs for this protein
                uniprot_ids = []
                if pd.notna(row['human_uniprot_id']):
                    uniprot_ids.append(row['human_uniprot_id'])
                if pd.notna(row['mouse_uniprot_id']):
                    uniprot_ids.append(row['mouse_uniprot_id'])
                if pd.notna(row['rat_uniprot_id']):
                    uniprot_ids.append(row['rat_uniprot_id'])
                
                protein_name = row['name2_entry']
                
                # Check if any of the UniProt IDs have AlphaFold embeddings
                found_embedding = False
                for uniprot_id in uniprot_ids:
                    af_embedding_path = self.af_embeddings_dir / uniprot_id / "embeddings" / "single_embedding.npy"
                    if af_embedding_path.exists():
                        proteins_to_process.append((uniprot_id, protein_name))
                        found_embedding = True
                        break
                
                if not found_embedding:
                    logger.warning(f"AlphaFold embedding not found for {protein_name} (tried: {uniprot_ids})")
            
            logger.info(f"Found {len(proteins_to_process)} proteins with AlphaFold embeddings")
            
            # Process each protein
            for uniprot_id, protein_name in proteins_to_process:
                result = self.process_protein(uniprot_id, protein_name)
                if result:
                    self.results.append(result)
                    logger.info(f"Successfully processed {protein_name}")
                else:
                    logger.warning(f"Failed to process {protein_name}")
            
            # Save results
            self.save_results()
            
            logger.info(f"QSAR modeling completed. Processed {len(self.results)} proteins successfully.")
            
        except Exception as e:
            logger.error(f"Error in QSAR modeling: {e}")
    
    def save_results(self):
        """Save modeling results to JSON files"""
        try:
            # Save regression results
            regression_results = []
            for result in self.results:
                if result['regression']:
                    regression_results.append({
                        'uniprot_id': result['uniprot_id'],
                        'protein_name': result['protein_name'],
                        'n_samples': result['n_samples'],
                        'cv_scores': result['regression']['cv_scores']
                    })
            
            with open(self.regression_dir / "regression_results.json", 'w') as f:
                json.dump(regression_results, f, indent=2)
            
            # Save classification results
            classification_results = []
            for result in self.results:
                if result['classification']:
                    classification_results.append({
                        'uniprot_id': result['uniprot_id'],
                        'protein_name': result['protein_name'],
                        'n_samples': result['n_samples'],
                        'cv_scores': result['classification']['cv_scores']
                    })
            
            with open(self.classification_dir / "classification_results.json", 'w') as f:
                json.dump(classification_results, f, indent=2)
            
            # Save summary
            summary = {
                'total_proteins': len(self.results),
                'regression_models': len(regression_results),
                'classification_models': len(classification_results),
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            with open(self.output_dir / "af_morgan_modeling_summary.json", 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info("Results saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")


def main():
    """Main function to run QSAR modeling"""
    try:
        # Initialize model
        qsar_model = AFMorganQSARModel()
        
        # Run modeling
        qsar_model.run_modeling()
        
    except Exception as e:
        logger.error(f"Error in main: {e}")


if __name__ == "__main__":
    main()
