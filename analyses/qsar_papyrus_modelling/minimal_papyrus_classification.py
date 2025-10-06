#!/usr/bin/env python3
"""
Minimal Papyrus QSAR Classification Script

This script implements a basic Random Forest classifier for protein bioactivity prediction
using the papyrus Python package. For each protein in the avoidome dataset, it:
1. Retrieves bioactivity data for human, mouse, and rat UniProt IDs
2. Converts pchembl values to binary classes (active ≥ 6.5, inactive < 6.5)
3. Pools and shuffles the data
4. Creates Morgan fingerprints
5. Trains a 5-fold CV Random Forest classifier
6. Reports classification results for all folds
7. Saves trained models to models_classification subdirectory

Date: 2025-01-30
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import warnings
import pickle
import os
warnings.filterwarnings('ignore')

# Papyrus imports
from papyrus_scripts import PapyrusDataset

# ML imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler

# Molecular fingerprinting
from rdkit import Chem
from rdkit.Chem import AllChem

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_modelling/papyrus_classification.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PapyrusQSARClassifier:
    """
    Minimal QSAR classifier using Papyrus data and Random Forest
    """
    
    def __init__(self, data_path: str = "/home/serramelendezcsm/RA/Avoidome/processed_data/papyrus_protein_check_results.csv", 
                 activity_threshold: float = 6.5):
        """
        Initialize the QSAR classifier
        
        Args:
            data_path: Path to the protein check results CSV file
            activity_threshold: Threshold for converting pchembl values to binary classes
                              (≥ threshold = active, < threshold = inactive)
        """
        self.data_path = data_path
        self.activity_threshold = activity_threshold
        self.papyrus_data = None
        self.proteins_df = None
        self.results = []
        
        # Create models directory
        self.models_dir = "/home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_modelling/models_classification"
        os.makedirs(self.models_dir, exist_ok=True)
        logger.info(f"Models will be saved to: {self.models_dir}")
        
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
                # Debug: Print sample of pchembl values
                if 'pchembl_value' in activities_df.columns:
                    sample_values = activities_df['pchembl_value'].head(5).tolist()
                    logger.info(f"Sample pchembl values for {uniprot_id}: {sample_values}")
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
        
        logger.info(f"Creating fingerprints for {len(smiles_list)} SMILES strings")
        
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
        
        logger.info(f"Successfully created {len(fingerprints)} valid fingerprints out of {len(smiles_list)} SMILES")
        
        if not fingerprints:
            return np.array([])
            
        return np.array(fingerprints), valid_indices
    
    def convert_to_binary_classes(self, pchembl_values: np.ndarray) -> np.ndarray:
        """
        Convert pchembl values to binary classes
        
        Args:
            pchembl_values: Array of pchembl values
            
        Returns:
            Array of binary classes (1 for active, 0 for inactive)
        """
        # Debug: Print statistics about pchembl values
        logger.info(f"Converting {len(pchembl_values)} pchembl values to binary classes")
        logger.info(f"pchembl value range: {pchembl_values.min():.3f} to {pchembl_values.max():.3f}")
        logger.info(f"pchembl value mean: {pchembl_values.mean():.3f}")
        logger.info(f"pchembl value median: {np.median(pchembl_values):.3f}")
        logger.info(f"Activity threshold: {self.activity_threshold}")
        
        # Count values above and below threshold
        above_threshold = np.sum(pchembl_values >= self.activity_threshold)
        below_threshold = np.sum(pchembl_values < self.activity_threshold)
        logger.info(f"Values >= {self.activity_threshold}: {above_threshold}")
        logger.info(f"Values < {self.activity_threshold}: {below_threshold}")
        
        return (pchembl_values >= self.activity_threshold).astype(int)
    
    def save_model(self, model, protein_name: str, fold: int):
        """
        Save a trained model to the models directory
        
        Args:
            model: Trained RandomForestClassifier
            protein_name: Name of the protein
            fold: Fold number
        """
        model_filename = f"{protein_name}_fold_{fold}.pkl"
        model_path = os.path.join(self.models_dir, model_filename)
        
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Saved model: {model_path}")
        except Exception as e:
            logger.error(f"Error saving model {model_path}: {e}")
    
    def train_rf_classifier(self, X: np.ndarray, y: np.ndarray, protein_name: str) -> Dict:
        """
        Train a Random Forest classifier with 5-fold CV
        
        Args:
            X: Feature matrix (Morgan fingerprints)
            y: Target values (binary classes)
            protein_name: Name of the protein
            
        Returns:
            Dictionary with CV results
        """
        if len(X) < 10:
            logger.warning(f"Insufficient data for {protein_name}: {len(X)} samples")
            return {
                'protein': protein_name,
                'n_samples': len(X),
                'n_active': np.sum(y == 1) if len(y) > 0 else 0,
                'n_inactive': np.sum(y == 0) if len(y) > 0 else 0,
                'status': 'insufficient_data',
                'cv_results': []
            }
        
        # Check class balance
        unique_classes, class_counts = np.unique(y, return_counts=True)
        logger.info(f"Class distribution for {protein_name}: {dict(zip(unique_classes, class_counts))}")
        
        if len(unique_classes) < 2:
            logger.warning(f"Only one class present for {protein_name}: {unique_classes}")
            return {
                'protein': protein_name,
                'n_samples': len(X),
                'n_active': np.sum(y == 1),
                'n_inactive': np.sum(y == 0),
                'status': 'single_class',
                'cv_results': []
            }
        
        # Initialize CV
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = []
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train model
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
            rf.fit(X_train, y_train)
            
            # Save model
            self.save_model(rf, protein_name, fold + 1)
            
            # Predict
            y_pred = rf.predict(X_test)
            y_pred_proba = rf.predict_proba(X_test)[:, 1]  # Probability of positive class
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            cv_results.append({
                'fold': fold + 1,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'n_train': len(X_train),
                'n_test': len(X_test),
                'n_train_active': np.sum(y_train == 1),
                'n_train_inactive': np.sum(y_train == 0),
                'n_test_active': np.sum(y_test == 1),
                'n_test_inactive': np.sum(y_test == 0)
            })
            
            logger.info(f"Fold {fold + 1} - Accuracy: {accuracy:.3f}, F1: {f1:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")
        
        return {
            'protein': protein_name,
            'n_samples': len(X),
            'status': 'success',
            'cv_results': cv_results,
            'n_active': np.sum(y == 1),
            'n_inactive': np.sum(y == 0)
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
                'n_active': 0,
                'n_inactive': 0,
                'status': 'no_data',
                'cv_results': []
            }
        
        # Combine all activities
        combined_activities = pd.concat(all_activities, ignore_index=True)
        logger.info(f"Combined {len(combined_activities)} activities for {protein_name}")
        
        # Debug: Print initial data info
        logger.info(f"Initial data shape: {combined_activities.shape}")
        if 'pchembl_value' in combined_activities.columns:
            logger.info(f"Initial pchembl_value column sample: {combined_activities['pchembl_value'].head(3).tolist()}")
        
        # Remove duplicates and invalid data
        combined_activities = combined_activities.dropna(subset=['SMILES', 'pchembl_value'])
        logger.info(f"After dropping NaN: {len(combined_activities)} activities")
        
        combined_activities = combined_activities.drop_duplicates(subset=['SMILES', 'pchembl_value'])
        logger.info(f"After dropping duplicates: {len(combined_activities)} activities")
        
        # Clean pchembl_value column - take the first value if there are multiple separated by semicolons
        combined_activities['pchembl_value_clean'] = combined_activities['pchembl_value'].astype(str).str.split(';').str[0]
        logger.info(f"Sample cleaned pchembl values: {combined_activities['pchembl_value_clean'].head(3).tolist()}")
        
        # Convert to numeric, dropping non-numeric values
        combined_activities['pchembl_value_numeric'] = pd.to_numeric(combined_activities['pchembl_value_clean'], errors='coerce')
        combined_activities = combined_activities.dropna(subset=['pchembl_value_numeric'])
        logger.info(f"After numeric conversion: {len(combined_activities)} activities")
        
        # Debug: Print numeric pchembl statistics
        if len(combined_activities) > 0:
            numeric_values = combined_activities['pchembl_value_numeric'].values
            logger.info(f"Numeric pchembl range: {numeric_values.min():.3f} to {numeric_values.max():.3f}")
            logger.info(f"Numeric pchembl mean: {numeric_values.mean():.3f}")
            logger.info(f"Numeric pchembl median: {np.median(numeric_values):.3f}")
        
        # Convert to binary classes
        combined_activities['activity_class'] = self.convert_to_binary_classes(
            combined_activities['pchembl_value_numeric'].values
        )
        
        # Log class distribution
        class_counts = combined_activities['activity_class'].value_counts()
        logger.info(f"Final class distribution for {protein_name}: {dict(class_counts)}")
        
        # Debug: Print some examples of the conversion
        if len(combined_activities) > 0:
            sample_data = combined_activities[['pchembl_value_numeric', 'activity_class']].head(5)
            logger.info(f"Sample pchembl -> class conversion:")
            for _, row in sample_data.iterrows():
                logger.info(f"  {row['pchembl_value_numeric']:.3f} -> {row['activity_class']}")
        
        # Shuffle the data
        combined_activities = combined_activities.sample(frac=1, random_state=42).reset_index(drop=True)
        
        if len(combined_activities) < 10:
            logger.warning(f"Insufficient data after filtering for {protein_name}: {len(combined_activities)} samples")
            return {
                'protein': protein_name,
                'n_samples': len(combined_activities),
                'n_active': combined_activities['activity_class'].sum(),
                'n_inactive': (combined_activities['activity_class'] == 0).sum(),
                'status': 'insufficient_data',
                'cv_results': []
            }
        
        # Create Morgan fingerprints
        smiles_list = combined_activities['SMILES'].tolist()
        logger.info(f"Creating fingerprints for {len(smiles_list)} SMILES strings")
        fingerprints, valid_indices = self.create_morgan_fingerprints(smiles_list)
        
        if len(fingerprints) == 0:
            logger.warning(f"No valid fingerprints created for {protein_name}")
            return {
                'protein': protein_name,
                'n_samples': 0,
                'n_active': 0,
                'n_inactive': 0,
                'status': 'no_valid_fingerprints',
                'cv_results': []
            }
        
        # Get corresponding binary class values
        y = combined_activities.iloc[valid_indices]['activity_class'].values
        
        logger.info(f"Created {len(fingerprints)} fingerprints for {protein_name}")
        logger.info(f"Corresponding class distribution: {np.sum(y == 1)} active, {np.sum(y == 0)} inactive")
        
        # Debug: Print final data summary before training
        logger.info(f"Final training data summary for {protein_name}:")
        logger.info(f"  - X shape: {fingerprints.shape}")
        logger.info(f"  - y shape: {y.shape}")
        logger.info(f"  - Active samples: {np.sum(y == 1)}")
        logger.info(f"  - Inactive samples: {np.sum(y == 0)}")
        
        # Train RF classifier
        result = self.train_rf_classifier(fingerprints, y, protein_name)
        
        return result
    
    def run_classification_pipeline(self):
        """Run the complete classification pipeline"""
        logger.info("Starting Papyrus classification pipeline...")
        logger.info(f"Activity threshold: {self.activity_threshold} (≥ threshold = active, < threshold = inactive)")
        
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
                    'n_active': 0,
                    'n_inactive': 0,
                    'status': 'error',
                    'cv_results': []
                })
        
        # Save results
        self.save_results()
        
        logger.info("Classification pipeline completed!")
    
    def save_results(self):
        """Save results to CSV file"""
        output_path = "/home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_modelling/classification_results.csv"
        
        # Create results DataFrame
        results_data = []
        for result in self.results:
            if result['status'] == 'success' and result['cv_results']:
                for cv_result in result['cv_results']:
                    results_data.append({
                        'protein': result['protein'],
                        'n_samples': result['n_samples'],
                        'n_active': result.get('n_active', 0),
                        'n_inactive': result.get('n_inactive', 0),
                        'fold': cv_result['fold'],
                        'accuracy': cv_result['accuracy'],
                        'precision': cv_result['precision'],
                        'recall': cv_result['recall'],
                        'f1_score': cv_result['f1_score'],
                        'n_train': cv_result['n_train'],
                        'n_test': cv_result['n_test'],
                        'n_train_active': cv_result['n_train_active'],
                        'n_train_inactive': cv_result['n_train_inactive'],
                        'n_test_active': cv_result['n_test_active'],
                        'n_test_inactive': cv_result['n_test_inactive']
                    })
            else:
                results_data.append({
                    'protein': result['protein'],
                    'n_samples': result['n_samples'],
                    'n_active': result.get('n_active', 0),
                    'n_inactive': result.get('n_inactive', 0),
                    'fold': None,
                    'accuracy': None,
                    'precision': None,
                    'recall': None,
                    'f1_score': None,
                    'n_train': None,
                    'n_test': None,
                    'n_train_active': None,
                    'n_train_inactive': None,
                    'n_test_active': None,
                    'n_test_inactive': None
                })
        
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
        
        # Print summary
        successful_models = results_df[results_df['accuracy'].notna()]
        logger.info(f"Successfully trained models: {len(successful_models['protein'].unique())}")
        if len(successful_models) > 0:
            logger.info(f"Average Accuracy: {successful_models['accuracy'].mean():.3f}")
            logger.info(f"Average F1-Score: {successful_models['f1_score'].mean():.3f}")
            logger.info(f"Average Precision: {successful_models['precision'].mean():.3f}")
            logger.info(f"Average Recall: {successful_models['recall'].mean():.3f}")

def main():
    """Main function to run the classification pipeline"""
    model = PapyrusQSARClassifier(activity_threshold=6.5)
    model.run_classification_pipeline()

if __name__ == "__main__":
    main() 