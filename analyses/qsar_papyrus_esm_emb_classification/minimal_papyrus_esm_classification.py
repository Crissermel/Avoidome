#!/usr/bin/env python3
"""
Minimal Papyrus QSAR Classification Script with ESM Embeddings

This script implements a basic Random Forest classifier for protein bioactivity classification
using the papyrus Python package combined with ESM protein embeddings. For each protein 
in the avoidome dataset, it:
1. Retrieves bioactivity data for human, mouse, and rat UniProt IDs
2. Pools and shuffles the data
3. Creates Morgan fingerprints for compounds
4. Loads ESM embeddings for proteins
5. Concatenates Morgan fingerprints with ESM embeddings
6. Converts continuous bioactivity values to binary classes using a threshold
7. Trains a 5-fold CV Random Forest classifier
8. Reports classification results for all folds

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Molecular fingerprinting
from rdkit import Chem
from rdkit.Chem import AllChem

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_esm_emb_classification/papyrus_esm_classification.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PapyrusESMQSARClassifier:
    """
    QSAR classifier using Papyrus data, Morgan fingerprints, and ESM embeddings
    """
    
    def __init__(self, 
                 data_path: str = "/home/serramelendezcsm/RA/Avoidome/processed_data/papyrus_protein_check_results.csv",
                 embeddings_path: str = "/home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_esm_emb/embeddings.npy",
                 targets_path: str = "/home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_esm_emb/targets_w_sequences.csv",
                 activity_threshold: float = 6.0):
        """
        Initialize the QSAR classifier with ESM embeddings
        
        Args:
            data_path: Path to the protein check results CSV file
            embeddings_path: Path to the ESM embeddings numpy file
            targets_path: Path to the targets with sequences CSV file
            activity_threshold: Threshold for converting continuous values to binary classes
        """
        self.data_path = data_path
        self.embeddings_path = embeddings_path
        self.targets_path = targets_path
        self.activity_threshold = activity_threshold
        self.papyrus_data = None
        self.proteins_df = None
        self.embeddings = None
        self.targets_df = None
        self.results = []
        
    def load_data(self):
        """Load protein data, ESM embeddings, and initialize Papyrus dataset"""
        logger.info("Loading protein data and ESM embeddings...")
        
        # Load protein check results
        self.proteins_df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(self.proteins_df)} proteins from {self.data_path}")
        
        # Load ESM embeddings
        self.embeddings = np.load(self.embeddings_path)
        logger.info(f"Loaded ESM embeddings with shape: {self.embeddings.shape}")
        
        # Load targets with sequences
        self.targets_df = pd.read_csv(self.targets_path)
        logger.info(f"Loaded {len(self.targets_df)} targets with sequences")
        
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
    
    def get_esm_embedding(self, uniprot_id: str) -> Optional[np.ndarray]:
        """
        Get ESM embedding for a given UniProt ID
        
        Args:
            uniprot_id: UniProt ID to get embedding for
            
        Returns:
            ESM embedding array or None if not found
        """
        try:
            # Find the protein in targets_df
            protein_row = self.targets_df[
                (self.targets_df['human_uniprot_id'] == uniprot_id) |
                (self.targets_df['mouse_uniprot_id'] == uniprot_id) |
                (self.targets_df['rat_uniprot_id'] == uniprot_id)
            ]
            
            if len(protein_row) > 0:
                # Get the index of the protein in the embeddings array
                protein_idx = protein_row.index[0]
                embedding = self.embeddings[protein_idx]
                logger.info(f"Retrieved ESM embedding for {uniprot_id} at index {protein_idx}")
                return embedding
            else:
                logger.warning(f"No ESM embedding found for {uniprot_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving ESM embedding for {uniprot_id}: {e}")
            return None
    
    def create_morgan_fingerprints(self, smiles_list: List[str], radius: int = 2, nBits: int = 2048) -> Tuple[np.ndarray, List[int]]:
        """
        Create Morgan fingerprints from SMILES strings
        
        Args:
            smiles_list: List of SMILES strings
            radius: Morgan fingerprint radius
            nBits: Number of bits in fingerprint
            
        Returns:
            Tuple of (fingerprint array, valid indices)
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
            return np.array([]), []
            
        return np.array(fingerprints), valid_indices
    
    def concatenate_features(self, morgan_fps: np.ndarray, esm_embedding: np.ndarray, n_samples: int) -> np.ndarray:
        """
        Concatenate Morgan fingerprints with ESM embeddings
        
        Args:
            morgan_fps: Morgan fingerprints array
            esm_embedding: ESM embedding array
            n_samples: Number of samples to create
            
        Returns:
            Concatenated feature array
        """
        # Repeat ESM embedding for each sample
        esm_features = np.tile(esm_embedding, (n_samples, 1))
        
        # Concatenate Morgan fingerprints with ESM features
        combined_features = np.concatenate([morgan_fps, esm_features], axis=1)
        
        logger.info(f"Concatenated features shape: {combined_features.shape}")
        logger.info(f"  - Morgan fingerprints: {morgan_fps.shape[1]} features")
        logger.info(f"  - ESM embeddings: {esm_embedding.shape[0]} features")
        
        return combined_features
    
    def convert_to_binary_classes(self, continuous_values: np.ndarray) -> np.ndarray:
        """
        Convert continuous bioactivity values to binary classes using threshold
        
        Args:
            continuous_values: Array of continuous bioactivity values
            
        Returns:
            Binary class array (1 for active, 0 for inactive)
        """
        binary_classes = (continuous_values >= self.activity_threshold).astype(int)
        
        # Log class distribution
        n_active = np.sum(binary_classes == 1)
        n_inactive = np.sum(binary_classes == 0)
        total = len(binary_classes)
        
        logger.info(f"Class distribution: {n_active} active ({n_active/total*100:.1f}%), {n_inactive} inactive ({n_inactive/total*100:.1f}%)")
        
        return binary_classes
    
    def train_rf_classifier(self, X: np.ndarray, y: np.ndarray, protein_name: str) -> Dict:
        """
        Train a Random Forest classifier with 5-fold CV
        
        Args:
            X: Feature matrix (Morgan fingerprints + ESM embeddings)
            y: Binary target values (0 or 1)
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
        
        # Check class balance
        unique_classes, class_counts = np.unique(y, return_counts=True)
        if len(unique_classes) < 2:
            logger.warning(f"Only one class present for {protein_name}: {unique_classes}")
            return {
                'protein': protein_name,
                'n_samples': len(X),
                'status': 'single_class',
                'cv_results': []
            }
        
        # Initialize CV
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
        
        cv_results = []
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Check if both classes are present in training set
            if len(np.unique(y_train)) < 2:
                logger.warning(f"Fold {fold + 1}: Only one class in training set for {protein_name}")
                continue
            
            # Train model
            rf.fit(X_train, y_train)
            
            # Predict
            y_pred = rf.predict(X_test)
            y_pred_proba = rf.predict_proba(X_test)[:, 1]  # Probability of positive class
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # Calculate AUC if both classes are present in test set
            auc = None
            if len(np.unique(y_test)) >= 2:
                try:
                    auc = roc_auc_score(y_test, y_pred_proba)
                except:
                    auc = None
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            cv_results.append({
                'fold': fold + 1,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'n_train': len(X_train),
                'n_test': len(X_test),
                'n_train_active': np.sum(y_train == 1),
                'n_train_inactive': np.sum(y_train == 0),
                'n_test_active': np.sum(y_test == 1),
                'n_test_inactive': np.sum(y_test == 0),
                'confusion_matrix': cm.tolist()
            })
            
            auc_str = f"{auc:.3f}" if auc is not None else "N/A"
            logger.info(f"Fold {fold + 1} - Accuracy: {accuracy:.3f}, F1: {f1:.3f}, AUC: {auc_str}")
        
        if not cv_results:
            return {
                'protein': protein_name,
                'n_samples': len(X),
                'status': 'no_valid_folds',
                'cv_results': []
            }
        
        # Calculate average AUC only if we have valid AUC values
        auc_values = [r['auc'] for r in cv_results if r['auc'] is not None]
        avg_auc = np.mean(auc_values) if auc_values else None
        
        return {
            'protein': protein_name,
            'n_samples': len(X),
            'status': 'success',
            'cv_results': cv_results,
            'avg_accuracy': np.mean([r['accuracy'] for r in cv_results]),
            'avg_precision': np.mean([r['precision'] for r in cv_results]),
            'avg_recall': np.mean([r['recall'] for r in cv_results]),
            'avg_f1': np.mean([r['f1'] for r in cv_results]),
            'avg_auc': avg_auc
        }
    
    def process_protein(self, row: pd.Series) -> Dict:
        """
        Process a single protein and train QSAR classifier
        
        Args:
            row: DataFrame row with protein information
            
        Returns:
            Dictionary with results
        """
        protein_name = row['name2_entry']
        human_id = row['human_uniprot_id']
        mouse_id = row['mouse_uniprot_id']
        rat_id = row['rat_uniprot_id']
        
        logger.info(f"Processing protein: {protein_name}")
        
        # Try to get activities for each organism
        all_activities = []
        
        for uniprot_id in [human_id, mouse_id, rat_id]:
            if pd.notna(uniprot_id):
                activities = self.get_protein_activities(uniprot_id)
                if activities is not None:
                    all_activities.append(activities)
        
        if not all_activities:
            logger.warning(f"No activities found for {protein_name}")
            return {
                'protein': protein_name,
                'status': 'no_activities',
                'n_samples': 0
            }
        
        # Combine all activities
        combined_activities = pd.concat(all_activities, ignore_index=True)
        logger.info(f"Combined {len(combined_activities)} activities for {protein_name}")
        
        # Get ESM embedding for the protein
        esm_embedding = None
        for uniprot_id in [human_id, mouse_id, rat_id]:
            if pd.notna(uniprot_id):
                esm_embedding = self.get_esm_embedding(uniprot_id)
                if esm_embedding is not None:
                    break
        
        if esm_embedding is None:
            logger.warning(f"No ESM embedding found for {protein_name}")
            return {
                'protein': protein_name,
                'status': 'no_esm_embedding',
                'n_samples': len(combined_activities)
            }
        
        # Shuffle the data
        combined_activities = combined_activities.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Create Morgan fingerprints
        smiles_list = combined_activities['SMILES'].tolist()
        morgan_fps, valid_indices = self.create_morgan_fingerprints(smiles_list)
        
        if len(morgan_fps) == 0:
            logger.warning(f"No valid fingerprints created for {protein_name}")
            return {
                'protein': protein_name,
                'status': 'no_valid_fingerprints',
                'n_samples': len(combined_activities)
            }
        
        # Filter activities to only include valid fingerprints
        filtered_activities = combined_activities.iloc[valid_indices]
        continuous_values = filtered_activities['pchembl_value_Mean'].values
        
        # Convert to binary classes
        y = self.convert_to_binary_classes(continuous_values)
        
        # Concatenate Morgan fingerprints with ESM embeddings
        X = self.concatenate_features(morgan_fps, esm_embedding, len(morgan_fps))
        
        # Train classifier
        result = self.train_rf_classifier(X, y, protein_name)
        
        logger.info(f"Completed processing {protein_name}: {result['status']}")
        return result
    
    def run_classification_pipeline(self):
        """Run the complete classification pipeline"""
        logger.info("Starting ESM QSAR classification pipeline...")
        logger.info(f"Using activity threshold: {self.activity_threshold}")
        
        # Load data
        self.load_data()
        
        # Process each protein
        for idx, row in self.proteins_df.iterrows():
            try:
                result = self.process_protein(row)
                self.results.append(result)
                
                if result['status'] == 'success':
                    logger.info(f"Protein {result['protein']}: Accuracy={result['avg_accuracy']:.3f}, F1={result['avg_f1']:.3f}")
                
            except Exception as e:
                logger.error(f"Error processing protein {row['name2_entry']}: {e}")
                self.results.append({
                    'protein': row['name2_entry'],
                    'status': 'error',
                    'error': str(e)
                })
        
        logger.info(f"Pipeline completed. Processed {len(self.results)} proteins.")
    
    def save_results(self):
        """Save results to CSV file"""
        results_df = pd.DataFrame(self.results)
        
        # Save detailed results
        output_path = '/home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_esm_emb_classification/esm_classification_results.csv'
        results_df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
        
        # Print summary
        successful_results = [r for r in self.results if r['status'] == 'success']
        if successful_results:
            avg_accuracy = np.mean([r['avg_accuracy'] for r in successful_results])
            avg_f1 = np.mean([r['avg_f1'] for r in successful_results])
            avg_auc_values = [r['avg_auc'] for r in successful_results if r['avg_auc'] is not None]
            avg_auc = np.mean(avg_auc_values) if avg_auc_values else None
            
            logger.info(f"Summary: {len(successful_results)} successful models")
            logger.info(f"Average Accuracy: {avg_accuracy:.3f}")
            logger.info(f"Average F1: {avg_f1:.3f}")
            if avg_auc is not None:
                logger.info(f"Average AUC: {avg_auc:.3f}")
            else:
                logger.info("Average AUC: N/A")

def main():
    """Main function to run the ESM QSAR classification pipeline"""
    model = PapyrusESMQSARClassifier(activity_threshold=6.0)
    model.run_classification_pipeline()
    model.save_results()

if __name__ == "__main__":
    main() 