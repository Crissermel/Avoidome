#!/usr/bin/env python3
"""
ESM-Only QSAR Prediction Pipeline
QSAR modeling using only ESM protein embeddings (no Morgan fingerprints).
"""

import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from papyrus_scripts import PapyrusDataset
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ESMOnlyQSARModel:
    """
    QSAR model using only ESM protein embeddings for bioactivity prediction.
    """
    
    def __init__(self, embeddings_path, targets_path, output_dir="analyses/qsar_papyrus_esm_only"):
        """
        Initialize the ESM-only QSAR model.
        
        Args:
            embeddings_path (str): Path to ESM embeddings .npy file
            targets_path (str): Path to targets CSV file
            output_dir (str): Output directory for results
        """
        self.embeddings_path = embeddings_path
        self.targets_path = targets_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.load_data()
        
    def load_data(self):
        """Load ESM embeddings and targets data."""
        logger.info("Loading ESM embeddings and targets data...")
        
        # Load ESM embeddings
        if os.path.exists(self.embeddings_path):
            self.embeddings = np.load(self.embeddings_path)
            logger.info(f"Loaded ESM embeddings: {self.embeddings.shape}")
        else:
            raise FileNotFoundError(f"ESM embeddings file not found: {self.embeddings_path}")
        
        # Load targets data
        if os.path.exists(self.targets_path):
            self.targets_df = pd.read_csv(self.targets_path)
            logger.info(f"Loaded targets data: {self.targets_df.shape}")
        else:
            raise FileNotFoundError(f"Targets file not found: {self.targets_path}")
        
        # Load Papyrus dataset for bioactivity data
        logger.info("Loading Papyrus dataset...")
        try:
            self.papyrus_dataset = PapyrusDataset()
            self.papyrus_df = self.papyrus_dataset.to_dataframe()
            logger.info(f"Loaded Papyrus dataset: {self.papyrus_df.shape}")
        except Exception as e:
            logger.error(f"Error loading Papyrus dataset: {e}")
            raise
    
    def get_esm_embedding(self, uniprot_id):
        """
        Get ESM embedding for a specific UniProt ID.
        
        Args:
            uniprot_id (str): UniProt ID
            
        Returns:
            np.array: ESM embedding vector (1280 dimensions)
        """
        # Find the index of the protein in targets_df
        protein_mask = self.targets_df['name2_entry'] == uniprot_id
        if protein_mask.any():
            protein_idx = self.targets_df[protein_mask].index[0]
            return self.embeddings[protein_idx]
        else:
            logger.warning(f"UniProt ID {uniprot_id} not found in ESM embeddings")
            return None
    
    def get_protein_activities(self, protein_name):
        """
        Get bioactivity data for a specific protein.
        
        Args:
            protein_name (str): Protein name
            
        Returns:
            pd.DataFrame: Bioactivity data for the protein
        """
        # Filter Papyrus data for the protein
        protein_activities = self.papyrus_df[
            (self.papyrus_df['protein_name'] == protein_name) |
            (self.papyrus_df['protein_name'].str.contains(protein_name, case=False, na=False))
        ]
        
        if len(protein_activities) == 0:
            logger.warning(f"No activities found for protein: {protein_name}")
            return pd.DataFrame()
        
        # Clean the data
        protein_activities = protein_activities.dropna(subset=['pchembl_value_Mean'])
        protein_activities = protein_activities[protein_activities['pchembl_value_Mean'].notna()]
        
        return protein_activities
    
    def train_esm_only_model(self, X, y, protein_name):
        """
        Train ESM-only QSAR model using Random Forest.
        
        Args:
            X (np.array): ESM embeddings (n_samples, 1280)
            y (np.array): Target values (pchembl_value_Mean)
            protein_name (str): Protein name for logging
            
        Returns:
            dict: Model performance metrics
        """
        if len(X) < 10:
            logger.warning(f"Insufficient data for {protein_name}: {len(X)} samples")
            return None
        
        # Initialize Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        
        # 5-fold cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        r2_scores = []
        rmse_scores = []
        mae_scores = []
        
        logger.info(f"Training ESM-only model for {protein_name} with {len(X)} samples")
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train model
            rf_model.fit(X_train, y_train)
            
            # Predict
            y_pred = rf_model.predict(X_test)
            
            # Calculate metrics
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            r2_scores.append(r2)
            rmse_scores.append(rmse)
            mae_scores.append(mae)
            
            logger.info(f"Fold {fold}: R²={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}")
        
        # Calculate average metrics
        avg_r2 = np.mean(r2_scores)
        avg_rmse = np.mean(rmse_scores)
        avg_mae = np.mean(mae_scores)
        
        std_r2 = np.std(r2_scores)
        std_rmse = np.std(rmse_scores)
        std_mae = np.std(mae_scores)
        
        results = {
            'protein': protein_name,
            'n_samples': len(X),
            'avg_r2': avg_r2,
            'std_r2': std_r2,
            'avg_rmse': avg_rmse,
            'std_rmse': std_rmse,
            'avg_mae': avg_mae,
            'std_mae': std_mae,
            'r2_scores': r2_scores,
            'rmse_scores': rmse_scores,
            'mae_scores': mae_scores,
            'status': 'success'
        }
        
        logger.info(f"ESM-only model for {protein_name}: Avg R²={avg_r2:.3f}±{std_r2:.3f}")
        
        return results
    
    def process_protein(self, protein_name, uniprot_id):
        """
        Process a single protein for ESM-only QSAR modeling.
        
        Args:
            protein_name (str): Protein name
            uniprot_id (str): UniProt ID
            
        Returns:
            dict: Model results or None if failed
        """
        try:
            # Get ESM embedding
            esm_embedding = self.get_esm_embedding(uniprot_id)
            if esm_embedding is None:
                logger.error(f"Could not get ESM embedding for {protein_name} ({uniprot_id})")
                return None
            
            # Get protein activities
            protein_activities = self.get_protein_activities(protein_name)
            if len(protein_activities) == 0:
                logger.error(f"No activities found for {protein_name}")
                return None
            
            # Prepare features and targets
            # For ESM-only, we repeat the same ESM embedding for all compounds
            X = np.tile(esm_embedding, (len(protein_activities), 1))  # Shape: (n_compounds, 1280)
            y = protein_activities['pchembl_value_Mean'].values
            
            # Remove any invalid values
            valid_mask = ~(np.isnan(y) | np.isinf(y))
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) < 10:
                logger.warning(f"Insufficient valid data for {protein_name}: {len(X)} samples")
                return {
                    'protein': protein_name,
                    'n_samples': len(X),
                    'status': 'insufficient_data'
                }
            
            # Train model
            results = self.train_esm_only_model(X, y, protein_name)
            
            if results:
                results['uniprot_id'] = uniprot_id
                results['feature_dimensions'] = X.shape[1]
                results['feature_type'] = 'ESM_only'
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing protein {protein_name}: {e}")
            return {
                'protein': protein_name,
                'uniprot_id': uniprot_id,
                'status': 'error',
                'error_message': str(e)
            }
    
    def run_all_proteins(self, protein_list):
        """
        Run ESM-only QSAR modeling for all proteins in the list.
        
        Args:
            protein_list (list): List of (protein_name, uniprot_id) tuples
            
        Returns:
            pd.DataFrame: Results for all proteins
        """
        logger.info(f"Starting ESM-only QSAR modeling for {len(protein_list)} proteins")
        
        results = []
        
        for i, (protein_name, uniprot_id) in enumerate(protein_list, 1):
            logger.info(f"Processing protein {i}/{len(protein_list)}: {protein_name}")
            
            result = self.process_protein(protein_name, uniprot_id)
            if result:
                results.append(result)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        results_path = self.output_dir / "esm_only_prediction_results.csv"
        results_df.to_csv(results_path, index=False)
        logger.info(f"Saved results to {results_path}")
        
        # Print summary
        self.print_summary(results_df)
        
        return results_df
    
    def print_summary(self, results_df):
        """Print a summary of the ESM-only modeling results."""
        logger.info("\n" + "="*50)
        logger.info("ESM-ONLY QSAR MODELING SUMMARY")
        logger.info("="*50)
        
        total_proteins = len(results_df)
        successful_models = results_df[results_df['status'] == 'success']
        
        logger.info(f"Total proteins processed: {total_proteins}")
        logger.info(f"Successful models: {len(successful_models)}")
        logger.info(f"Success rate: {len(successful_models)/total_proteins*100:.1f}%")
        
        if len(successful_models) > 0:
            logger.info(f"\nPerformance Summary (Successful Models):")
            logger.info(f"Average R²: {successful_models['avg_r2'].mean():.3f}")
            logger.info(f"Average RMSE: {successful_models['avg_rmse'].mean():.3f}")
            logger.info(f"Average MAE: {successful_models['avg_mae'].mean():.3f}")
            
            # Top performing models
            top_models = successful_models.nlargest(5, 'avg_r2')
            logger.info(f"\nTop 5 Performing Models:")
            for _, model in top_models.iterrows():
                logger.info(f"- {model['protein']}: R²={model['avg_r2']:.3f}, RMSE={model['avg_rmse']:.3f}")
        
        # Status breakdown
        status_counts = results_df['status'].value_counts()
        logger.info(f"\nStatus Breakdown:")
        for status, count in status_counts.items():
            logger.info(f"- {status}: {count} proteins")


def main():
    """Main function to run ESM-only QSAR modeling."""
    # File paths
    embeddings_path = "../../embeddings.npy"
    targets_path = "../qsar_papyrus_esm_emb/targets_w_sequences.csv"
    
    # Initialize model
    model = ESMOnlyQSARModel(embeddings_path, targets_path)
    
    # Load protein check results to get protein list
    protein_check_path = "../qsar_papyrus_esm_emb/papyrus_protein_check_results.csv"
    if os.path.exists(protein_check_path):
        protein_check_df = pd.read_csv(protein_check_path)
        
        # Create protein list with UniProt IDs
        protein_list = []
        for _, row in protein_check_df.iterrows():
            protein_name = row['protein_name']
            # Try to get UniProt ID from the targets data
            protein_mask = model.targets_df['name2_entry'] == protein_name
            if protein_mask.any():
                uniprot_id = protein_name  # The name2_entry is the UniProt ID
                protein_list.append((protein_name, uniprot_id))
        
        logger.info(f"Found {len(protein_list)} proteins with ESM embeddings")
        
        # Run modeling
        results = model.run_all_proteins(protein_list)
        
    else:
        logger.error(f"Protein check results file not found: {protein_check_path}")


if __name__ == "__main__":
    main() 