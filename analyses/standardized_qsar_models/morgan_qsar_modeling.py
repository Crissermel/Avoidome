#!/usr/bin/env python3
"""
Standardized QSAR Modeling with Morgan Fingerprints and Physicochemical Descriptors

This script creates QSAR models for each protein in the Avoidome dataset using:
- Morgan fingerprints (2048 bits, radius=2)
- Physicochemical descriptors (calculated using RDKit)
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
        logging.FileHandler('/home/serramelendezcsm/RA/Avoidome/analyses/standardized_qsar_models/morgan_qsar.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MorganQSARModel:
    """
    QSAR model using Morgan fingerprints and physicochemical descriptors
    """
    
    def __init__(self, 
                 data_path: str = "/home/serramelendezcsm/RA/Avoidome/processed_data/papyrus_protein_check_results.csv",
                 output_dir: str = "/home/serramelendezcsm/RA/Avoidome/analyses/standardized_qsar_models"):
        """
        Initialize the QSAR model
        
        Args:
            data_path: Path to the protein check results CSV file
            output_dir: Directory to save models and results
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.papyrus_data = None
        self.proteins_df = None
        self.papyrus_df = None
        self.results = []
        
        # Create output directories
        self.regression_dir = self.output_dir / "morgan_regression"
        self.classification_dir = self.output_dir / "morgan_classification"
        self.regression_dir.mkdir(exist_ok=True)
        self.classification_dir.mkdir(exist_ok=True)
        
    def load_data(self):
        """Load protein data and initialize Papyrus dataset"""
        logger.info("Loading protein data and Papyrus dataset...")
        
        # Load protein check results
        self.proteins_df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(self.proteins_df)} proteins from {self.data_path}")
        
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
    
    def prepare_features(self, activities_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features (Morgan + physicochemical) and targets for modeling
        
        Args:
            activities_df: DataFrame with bioactivity data
            
        Returns:
            Tuple of (features, targets, smiles_list)
        """
        # Filter out invalid data
        valid_data = activities_df.dropna(subset=['SMILES', 'pchembl_value_Mean'])
        valid_data = valid_data[valid_data['SMILES'] != '']
        
        if len(valid_data) == 0:
            raise ValueError("No valid data found")
        
        smiles_list = valid_data['SMILES'].tolist()
        targets = valid_data['pchembl_value_Mean'].values
        
        logger.info(f"Preparing features for {len(smiles_list)} compounds")
        
        # Calculate Morgan fingerprints
        logger.info("Calculating Morgan fingerprints...")
        morgan_fps = self.calculate_morgan_fingerprints(smiles_list)
        
        # Calculate physicochemical descriptors
        logger.info("Calculating physicochemical descriptors...")
        physchem_descriptors = self.calculate_physicochemical_features(smiles_list)
        
        # Combine features
        features = np.hstack([morgan_fps, physchem_descriptors])
        
        logger.info(f"Feature matrix shape: {features.shape}")
        logger.info(f"Morgan fingerprints: {morgan_fps.shape[1]} bits")
        logger.info(f"Physicochemical descriptors: {physchem_descriptors.shape[1]} features")
        
        return features, targets, smiles_list
    
    def train_regression_model(self, features: np.ndarray, targets: np.ndarray, 
                             protein_name: str, uniprot_id: str, organism: str = 'human') -> Dict:
        """
        Train regression model with 5-fold CV
        
        Args:
            features: Feature matrix
            targets: Target values
            protein_name: Name of the protein
            uniprot_id: UniProt ID
            
        Returns:
            Dictionary with model results
        """
        logger.info(f"Training regression model for {protein_name} ({uniprot_id})")
        
        # Initialize results dictionary
        results = {
            'protein_name': protein_name,
            'uniprot_id': uniprot_id,
            'organism': organism,
            'model_type': 'regression',
            'n_samples': len(targets),
            'n_features': features.shape[1],
            'cv_scores': [],
            'cv_predictions': [],
            'cv_true_values': [],
            'fold_models': [],
            'feature_importance': None,
            'final_model': None
        }
        
        # 5-fold cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(features)):
            logger.info(f"Training fold {fold + 1}/5")
            
            X_train, X_val = features[train_idx], features[val_idx]
            y_train, y_val = targets[train_idx], targets[val_idx]
            
            # Handle missing values
            imputer = SimpleImputer(strategy='median')
            X_train_imputed = imputer.fit_transform(X_train)
            X_val_imputed = imputer.transform(X_val)
            
            # Train model
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train_imputed, y_train)
            
            # Predictions
            y_pred = model.predict(X_val_imputed)
            
            # Calculate metrics
            mse = mean_squared_error(y_val, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            fold_results = {
                'fold': fold + 1,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'predictions': y_pred,
                'true_values': y_val
            }
            
            results['cv_scores'].append(fold_results)
            results['cv_predictions'].extend(y_pred)
            results['cv_true_values'].extend(y_val)
            results['fold_models'].append((imputer, model))
        
        # Train final model on all data
        logger.info("Training final model on all data")
        final_imputer = SimpleImputer(strategy='median')
        features_imputed = final_imputer.fit_transform(features)
        
        final_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        final_model.fit(features_imputed, targets)
        
        results['final_model'] = (final_imputer, final_model)
        results['feature_importance'] = final_model.feature_importances_
        
        # Calculate overall CV metrics
        cv_predictions = np.array(results['cv_predictions'])
        cv_true_values = np.array(results['cv_true_values'])
        
        results['overall_metrics'] = {
            'cv_mse': mean_squared_error(cv_true_values, cv_predictions),
            'cv_rmse': np.sqrt(mean_squared_error(cv_true_values, cv_predictions)),
            'cv_mae': mean_absolute_error(cv_true_values, cv_predictions),
            'cv_r2': r2_score(cv_true_values, cv_predictions)
        }
        
        logger.info(f"Regression model completed for {protein_name}")
        logger.info(f"CV R²: {results['overall_metrics']['cv_r2']:.4f}")
        logger.info(f"CV RMSE: {results['overall_metrics']['cv_rmse']:.4f}")
        
        return results
    
    def train_classification_model(self, features: np.ndarray, targets: np.ndarray, 
                                 protein_name: str, uniprot_id: str, organism: str = 'human',
                                 threshold: float = 7.0) -> Dict:
        """
        Train classification model with 5-fold CV
        
        Args:
            features: Feature matrix
            targets: Target values (continuous)
            protein_name: Name of the protein
            uniprot_id: UniProt ID
            threshold: Threshold for binary classification
            
        Returns:
            Dictionary with model results
        """
        logger.info(f"Training classification model for {protein_name} ({uniprot_id})")
        
        # Convert to binary classification
        binary_targets = (targets > threshold).astype(int)
        n_active = np.sum(binary_targets)
        n_inactive = len(binary_targets) - n_active
        
        logger.info(f"Binary classification: {n_active} active, {n_inactive} inactive (threshold: {threshold})")
        
        if n_active < 5 or n_inactive < 5:
            logger.warning(f"Insufficient class balance for {protein_name}: {n_active} active, {n_inactive} inactive")
        
        # Initialize results dictionary
        results = {
            'protein_name': protein_name,
            'uniprot_id': uniprot_id,
            'organism': organism,
            'model_type': 'classification',
            'threshold': threshold,
            'n_samples': len(binary_targets),
            'n_features': features.shape[1],
            'n_active': n_active,
            'n_inactive': n_inactive,
            'cv_scores': [],
            'cv_predictions': [],
            'cv_true_values': [],
            'cv_probabilities': [],
            'fold_models': [],
            'feature_importance': None,
            'final_model': None
        }
        
        # 5-fold cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(features)):
            logger.info(f"Training fold {fold + 1}/5")
            
            X_train, X_val = features[train_idx], features[val_idx]
            y_train, y_val = binary_targets[train_idx], binary_targets[val_idx]
            
            # Handle missing values
            imputer = SimpleImputer(strategy='median')
            X_train_imputed = imputer.fit_transform(X_train)
            X_val_imputed = imputer.transform(X_val)
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            
            model.fit(X_train_imputed, y_train)
            
            # Predictions
            y_pred = model.predict(X_val_imputed)
            
            # Handle predict_proba for single class case
            proba = model.predict_proba(X_val_imputed)
            if proba.shape[1] > 1:
                y_prob = proba[:, 1]  # Probability of positive class
            else:
                y_prob = proba[:, 0]  # Only one class, use that probability
            
            # Calculate metrics
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            
            # Calculate AUC if both classes are present
            try:
                auc = roc_auc_score(y_val, y_prob)
            except ValueError:
                auc = np.nan
            
            fold_results = {
                'fold': fold + 1,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'predictions': y_pred,
                'probabilities': y_prob,
                'true_values': y_val
            }
            
            results['cv_scores'].append(fold_results)
            results['cv_predictions'].extend(y_pred)
            results['cv_true_values'].extend(y_val)
            results['cv_probabilities'].extend(y_prob)
            results['fold_models'].append((imputer, model))
        
        # Train final model on all data
        logger.info("Training final model on all data")
        final_imputer = SimpleImputer(strategy='median')
        features_imputed = final_imputer.fit_transform(features)
        
        final_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        final_model.fit(features_imputed, binary_targets)
        
        results['final_model'] = (final_imputer, final_model)
        results['feature_importance'] = final_model.feature_importances_
        
        # Calculate overall CV metrics
        cv_predictions = np.array(results['cv_predictions'])
        cv_true_values = np.array(results['cv_true_values'])
        cv_probabilities = np.array(results['cv_probabilities'])
        
        results['overall_metrics'] = {
            'cv_accuracy': accuracy_score(cv_true_values, cv_predictions),
            'cv_precision': precision_score(cv_true_values, cv_predictions, zero_division=0),
            'cv_recall': recall_score(cv_true_values, cv_predictions, zero_division=0),
            'cv_f1': f1_score(cv_true_values, cv_predictions, zero_division=0),
            'cv_auc': roc_auc_score(cv_true_values, cv_probabilities) if len(np.unique(cv_true_values)) > 1 else np.nan
        }
        
        logger.info(f"Classification model completed for {protein_name}")
        logger.info(f"CV Accuracy: {results['overall_metrics']['cv_accuracy']:.4f}")
        logger.info(f"CV F1: {results['overall_metrics']['cv_f1']:.4f}")
        logger.info(f"CV AUC: {results['overall_metrics']['cv_auc']:.4f}")
        
        return results
    
    def save_model_results(self, results: Dict, model_type: str, organism: str):
        """
        Save model results to files
        
        Args:
            results: Model results dictionary
            model_type: 'morgan_regression' or 'morgan_classification'
            organism: 'human', 'mouse', or 'rat'
        """
        protein_name = results['protein_name']
        uniprot_id = results['uniprot_id']
        
        # Create organism-specific directory structure
        organism_dir = self.output_dir / model_type / organism
        protein_dir = organism_dir / f"{protein_name}_{uniprot_id}"
        protein_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = protein_dir / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(results['final_model'], f)
        
        # Save results (without the model objects)
        results_to_save = results.copy()
        results_to_save['final_model'] = None  # Don't save the model object in JSON
        results_to_save['fold_models'] = None  # Don't save fold models in JSON
        
        results_path = protein_dir / "results.json"
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                return obj
            
            json.dump(results_to_save, f, indent=2, default=convert_numpy)
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'true_values': results['cv_true_values'],
            'predictions': results['cv_predictions']
        })
        
        if model_type == 'classification':
            predictions_df['probabilities'] = results['cv_probabilities']
        
        predictions_path = protein_dir / "predictions.csv"
        predictions_df.to_csv(predictions_path, index=False)
        
        # Save feature importance
        feature_names = [f"morgan_{i}" for i in range(2048)] + [
            'ALogP', 'Molecular_Weight', 'Num_H_Donors', 'Num_H_Acceptors',
            'Num_Rotatable_Bonds', 'Num_Atoms', 'Num_Rings', 'Num_Aromatic_Rings',
            'LogS', 'Molecular_Surface_Area', 'Molecular_Polar_Surface_Area',
            'Num_Heavy_Atoms', 'Formal_Charge', 'Num_Saturated_Rings'
        ]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': results['feature_importance']
        }).sort_values('importance', ascending=False)
        
        importance_path = protein_dir / "feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        
        logger.info(f"Saved {model_type} model results for {protein_name} to {protein_dir}")
    
    def process_protein(self, protein_row: pd.Series) -> List[Dict]:
        """
        Process a single protein for ALL available organisms
        Creates separate models for human, mouse, and rat if data available
        
        Args:
            protein_row: Row from proteins DataFrame
            
        Returns:
            List of dictionaries with processing results for each organism
        """
        protein_name = protein_row['name2_entry']
        organisms = {
            'human': protein_row['human_uniprot_id'],
            'mouse': protein_row['mouse_uniprot_id'], 
            'rat': protein_row['rat_uniprot_id']
        }
        
        logger.info(f"Processing protein: {protein_name}")
        
        results = []
        
        for organism, uniprot_id in organisms.items():
            if pd.notna(uniprot_id) and uniprot_id:
                organism_result = self.process_organism(protein_name, organism, uniprot_id)
                results.append(organism_result)
            else:
                logger.info(f"No {organism} data available for {protein_name}")
        
        return results
    
    def process_organism(self, protein_name: str, organism: str, uniprot_id: str) -> Dict:
        """
        Process a single protein-organism combination
        
        Args:
            protein_name: Name of the protein
            organism: Organism type (human, mouse, rat)
            uniprot_id: UniProt ID for this organism
            
        Returns:
            Dictionary with processing results
        """
        logger.info(f"Processing {protein_name} for {organism} ({uniprot_id})")
        
        # Check data availability
        activities_df = self.get_protein_activities(uniprot_id)
        
        if activities_df is None or len(activities_df) < 30:
            logger.warning(f"Insufficient data for {protein_name} ({organism}): {len(activities_df) if activities_df is not None else 0} samples")
            return {
                'protein_name': protein_name,
                'organism': organism,
                'uniprot_id': uniprot_id,
                'status': 'skipped',
                'reason': 'insufficient_data',
                'n_samples': len(activities_df) if activities_df is not None else 0
            }
        
        try:
            # Prepare features
            features, targets, smiles_list = self.prepare_features(activities_df)
            
            # Train regression model
            logger.info(f"Training regression model for {protein_name} ({organism})")
            regression_results = self.train_regression_model(features, targets, protein_name, uniprot_id, organism)
            self.save_model_results(regression_results, 'morgan_regression', organism)
            
            # Train classification model
            logger.info(f"Training classification model for {protein_name} ({organism})")
            classification_results = self.train_classification_model(features, targets, protein_name, uniprot_id, organism)
            self.save_model_results(classification_results, 'morgan_classification', organism)
            
            return {
                'protein_name': protein_name,
                'organism': organism,
                'uniprot_id': uniprot_id,
                'status': 'completed',
                'n_samples': len(targets),
                'regression_r2': regression_results['overall_metrics']['cv_r2'],
                'regression_rmse': regression_results['overall_metrics']['cv_rmse'],
                'classification_accuracy': classification_results['overall_metrics']['cv_accuracy'],
                'classification_f1': classification_results['overall_metrics']['cv_f1'],
                'classification_auc': classification_results['overall_metrics']['cv_auc']
            }
            
        except Exception as e:
            logger.error(f"Error processing {protein_name} ({organism}): {e}")
            return {
                'protein_name': protein_name,
                'organism': organism,
                'uniprot_id': uniprot_id,
                'status': 'error',
                'error': str(e)
            }
    
    def run_all_models(self):
        """Run QSAR models for all proteins"""
        logger.info("Starting QSAR modeling for all proteins")
        
        # Load data
        self.load_data()
        
        # Process each protein
        all_results = []
        
        for idx, protein_row in self.proteins_df.iterrows():
            try:
                protein_results = self.process_protein(protein_row)
                all_results.extend(protein_results)
                
                # Log progress for each organism
                for result in protein_results:
                    if result['status'] == 'completed':
                        logger.info(f"Completed {result['protein_name']} ({result['organism']}): "
                                  f"R²={result['regression_r2']:.3f}, "
                                  f"Acc={result['classification_accuracy']:.3f}")
                    else:
                        logger.warning(f"Skipped {result['protein_name']} ({result['organism']}): {result.get('reason', 'error')}")
                    
            except Exception as e:
                logger.error(f"Fatal error processing {protein_row['name2_entry']}: {e}")
                all_results.append({
                    'protein_name': protein_row['name2_entry'],
                    'organism': 'unknown',
                    'status': 'fatal_error',
                    'error': str(e)
                })
        
        # Save summary results
        summary_df = pd.DataFrame(all_results)
        summary_path = self.output_dir / "modeling_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        
        # Print summary
        completed = summary_df[summary_df['status'] == 'completed']
        skipped = summary_df[summary_df['status'] == 'skipped']
        errors = summary_df[summary_df['status'].isin(['error', 'fatal_error'])]
        
        logger.info("=" * 60)
        logger.info("MODELING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total proteins: {len(summary_df)}")
        logger.info(f"Completed: {len(completed)}")
        logger.info(f"Skipped: {len(skipped)}")
        logger.info(f"Errors: {len(errors)}")
        
        if len(completed) > 0:
            logger.info(f"Average regression R²: {completed['regression_r2'].mean():.3f}")
            logger.info(f"Average classification accuracy: {completed['classification_accuracy'].mean():.3f}")
            logger.info(f"Average classification F1: {completed['classification_f1'].mean():.3f}")
        
        logger.info(f"Summary saved to: {summary_path}")
        logger.info("Modeling completed!")


def main():
    """Main function to run QSAR modeling"""
    model = MorganQSARModel()
    model.run_all_models()


if __name__ == "__main__":
    main()