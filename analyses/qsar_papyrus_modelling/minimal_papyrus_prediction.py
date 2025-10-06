#!/usr/bin/env python3
"""
Minimal Papyrus QSAR Prediction Script

This script implements a basic Random Forest model for protein bioactivity prediction
using the papyrus Python package. For each protein in the avoidome dataset, it:
1. Retrieves bioactivity data for human, mouse, and rat UniProt IDs
2. Pools and shuffles the data
3. Creates Morgan fingerprints
4. Trains a 5-fold CV Random Forest model
5. Reports results for all folds


Date: 2025-01-30
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Papyrus imports
from papyrus_scripts import PapyrusDataset

# ML imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Molecular fingerprinting
from rdkit import Chem
from rdkit.Chem import AllChem

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_modelling/papyrus_prediction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PapyrusQSARModel:
    """
    Minimal QSAR model using Papyrus data and Random Forest
    """
    
    def __init__(self, data_path: str = "/home/serramelendezcsm/RA/Avoidome/processed_data/papyrus_protein_check_results.csv"):
        """
        Initialize the QSAR model
        
        Args:
            data_path: Path to the protein check results CSV file
        """
        self.data_path = data_path
        self.papyrus_data = None
        self.proteins_df = None
        self.results = []
        # Store per-sample predictions across all proteins/folds
        self.fold_predictions = []
        
    def load_data(self):
        """Load protein data and initialize Papyrus dataset"""
        logger.info("Loading protein data...")
        
        # Load protein check results
        self.proteins_df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(self.proteins_df)} proteins from {self.data_path}")
        
        # Initialize Papyrus dataset
        logger.info("Initializing Papyrus dataset...")
        papyrus_data = PapyrusDataset(version='latest', plusplus=True)
        logger.info("Papyrus dataset initialized successfully")
        
        # Load full dataset into DataFrame for efficient filtering
        logger.info("Loading full Papyrus dataset into DataFrame...")
        self.papyrus_df = papyrus_data.to_dataframe()
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
                
            # Filter bioactivity data for the specific protein using pandas
            activities_df = self.papyrus_df[self.papyrus_df['accession'] == uniprot_id]
            
            if len(activities_df) > 0:
                logger.info(f"Retrieved {len(activities_df)} activities for {uniprot_id}")
                return activities_df
            else:
                logger.warning(f"No activities found for {uniprot_id}")
                return None
            
        except Exception as e:
            logger.error(f"Error retrieving data for {uniprot_id}: {e}")
            return None
    
    def create_morgan_fingerprints(self, smiles_list: List[str], radius: int = 2, nBits: int = 2048) -> np.ndarray:
        """
        Create Morgan fingerprints from SMILES strings
        
        Args:
            smiles_list: List of SMILES strings
            radius: Morgan fingerprint radius
            nBits: Number of bits in fingerprint
            
        Returns:
            Array of Morgan fingerprints
        """
        fingerprints = []
        valid_indices = []
        
        for i, smiles in enumerate(smiles_list):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
                    fingerprints.append(np.array(fp))
                    valid_indices.append(i)
                else:
                    logger.warning(f"Invalid SMILES: {smiles}")
            except Exception as e:
                logger.warning(f"Error creating fingerprint for {smiles}: {e}")
        
        if not fingerprints:
            return np.array([])
            
        return np.array(fingerprints), valid_indices
    
    def train_rf_model(self, X: np.ndarray, y: np.ndarray, protein_name: str) -> Dict:
        """
        Train a Random Forest model with 5-fold CV
        
        Args:
            X: Feature matrix (Morgan fingerprints)
            y: Target values (bioactivity)
            protein_name: Name of the protein
            
        Returns:
            Dictionary with CV results
        """
        if len(X) < 10:
            logger.warning(f"Insufficient data for {protein_name}: {len(X)} samples")
            return {
                'protein': protein_name,
                'n_samples': len(X),
                'status': 'insufficient_data',
                'cv_results': []
            }
        
        # Initialize CV
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        cv_results = []
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train model
            rf.fit(X_train, y_train)
            
            # Predict
            y_pred = rf.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            cv_results.append({
                'fold': fold + 1,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'n_train': len(X_train),
                'n_test': len(X_test)
            })
            
            logger.info(f"Fold {fold + 1} - RMSE: {rmse:.3f}, R²: {r2:.3f}")

            # Save per-sample predictions for MCC computation later
            for yt, yp in zip(y_test.tolist(), y_pred.tolist()):
                self.fold_predictions.append({
                    'protein': protein_name,
                    'fold': fold + 1,
                    'y_true': yt,
                    'y_pred': yp
                })
        
        return {
            'protein': protein_name,
            'n_samples': len(X),
            'status': 'success',
            'cv_results': cv_results
        }
    
    def process_protein(self, row: pd.Series) -> Dict:
        """
        Process a single protein row
        
        Args:
            row: DataFrame row with protein information
            
        Returns:
            Dictionary with modeling results
        """
        protein_name = row['name2_entry']
        human_id = row['human_uniprot_id']
        mouse_id = row['mouse_uniprot_id']
        rat_id = row['rat_uniprot_id']
        
        logger.info(f"Processing protein: {protein_name}")
        logger.info(f"UniProt IDs - Human: {human_id}, Mouse: {mouse_id}, Rat: {rat_id}")
        
        # Collect all activities
        all_activities = []
        
        for uniprot_id in [human_id, mouse_id, rat_id]:
            if pd.notna(uniprot_id) and uniprot_id:
                activities = self.get_protein_activities(uniprot_id)
                if activities is not None and len(activities) > 0:
                    all_activities.append(activities)
        
        if not all_activities:
            logger.warning(f"No activities found for {protein_name}")
            return {
                'protein': protein_name,
                'n_samples': 0,
                'status': 'no_data',
                'cv_results': []
            }
        
        # Combine all activities
        combined_activities = pd.concat(all_activities, ignore_index=True)
        logger.info(f"Combined {len(combined_activities)} activities for {protein_name}")
        
        # Remove duplicates and invalid data
        combined_activities = combined_activities.dropna(subset=['SMILES', 'pchembl_value'])
        combined_activities = combined_activities.drop_duplicates(subset=['SMILES', 'pchembl_value'])
        
        # Clean pchembl_value column - take the first value if there are multiple separated by semicolons
        combined_activities['pchembl_value_clean'] = combined_activities['pchembl_value'].astype(str).str.split(';').str[0]
        
        # Convert to numeric, dropping non-numeric values
        combined_activities['pchembl_value_numeric'] = pd.to_numeric(combined_activities['pchembl_value_clean'], errors='coerce')
        combined_activities = combined_activities.dropna(subset=['pchembl_value_numeric'])
        
        # Shuffle the data
        combined_activities = combined_activities.sample(frac=1, random_state=42).reset_index(drop=True)
        
        if len(combined_activities) < 10:
            logger.warning(f"Insufficient data after filtering for {protein_name}: {len(combined_activities)} samples")
            return {
                'protein': protein_name,
                'n_samples': len(combined_activities),
                'status': 'insufficient_data',
                'cv_results': []
            }
        
        # Create Morgan fingerprints
        smiles_list = combined_activities['SMILES'].tolist()
        fingerprints, valid_indices = self.create_morgan_fingerprints(smiles_list)
        
        if len(fingerprints) == 0:
            logger.warning(f"No valid fingerprints created for {protein_name}")
            return {
                'protein': protein_name,
                'n_samples': 0,
                'status': 'no_valid_fingerprints',
                'cv_results': []
            }
        
        # Get corresponding bioactivity values
        y = combined_activities.iloc[valid_indices]['pchembl_value_numeric'].values
        
        logger.info(f"Created {len(fingerprints)} fingerprints for {protein_name}")
        
        # Train RF model
        result = self.train_rf_model(fingerprints, y, protein_name)
        
        return result
    
    def run_prediction_pipeline(self):
        """Run the complete prediction pipeline"""
        logger.info("Starting Papyrus prediction pipeline...")
        
        # Load data
        self.load_data()
        
        # Process each protein
        for idx, row in self.proteins_df.iterrows():
            try:
                result = self.process_protein(row)
                self.results.append(result)
                logger.info(f"Completed processing {row['name2_entry']}")
            except Exception as e:
                logger.error(f"Error processing {row['name2_entry']}: {e}")
                self.results.append({
                    'protein': row['name2_entry'],
                    'n_samples': 0,
                    'status': 'error',
                    'cv_results': []
                })
        
        # Save results
        self.save_results()
        
        logger.info("Prediction pipeline completed!")
    
    def save_results(self):
        """Save results to CSV file"""
        output_path = "/home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_modelling/prediction_results.csv"
        
        # Create results DataFrame
        results_data = []
        for result in self.results:
            if result['status'] == 'success' and result['cv_results']:
                for cv_result in result['cv_results']:
                    results_data.append({
                        'protein': result['protein'],
                        'n_samples': result['n_samples'],
                        'fold': cv_result['fold'],
                        'mse': cv_result['mse'],
                        'rmse': cv_result['rmse'],
                        'mae': cv_result['mae'],
                        'r2': cv_result['r2'],
                        'n_train': cv_result['n_train'],
                        'n_test': cv_result['n_test']
                    })
            else:
                results_data.append({
                    'protein': result['protein'],
                    'n_samples': result['n_samples'],
                    'fold': None,
                    'mse': None,
                    'rmse': None,
                    'mae': None,
                    'r2': None,
                    'n_train': None,
                    'n_test': None
                })
        
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")

        # Also save per-sample fold predictions for downstream MCC computation
        try:
            if len(self.fold_predictions) > 0:
                preds_df = pd.DataFrame(self.fold_predictions)
                preds_out = "/home/serramelendezcsm/RA/Avoidome/analyses/MCC_comparison/morgan_regression_fold_predictions.csv"
                Path(preds_out).parent.mkdir(parents=True, exist_ok=True)
                preds_df.to_csv(preds_out, index=False)
                logger.info(f"Per-sample fold predictions saved to {preds_out}")
            else:
                logger.warning("No per-sample predictions collected; skipping MCC export file for Morgan regression")
        except Exception as e:
            logger.error(f"Failed to save per-sample predictions: {e}")
        
        # Print summary
        successful_models = results_df[results_df['r2'].notna()]
        logger.info(f"Successfully trained models: {len(successful_models['protein'].unique())}")
        if len(successful_models) > 0:
            logger.info(f"Average R²: {successful_models['r2'].mean():.3f}")
            logger.info(f"Average RMSE: {successful_models['rmse'].mean():.3f}")

def main():
    """Main function to run the prediction pipeline"""
    model = PapyrusQSARModel()
    model.run_prediction_pipeline()

if __name__ == "__main__":
    main() 