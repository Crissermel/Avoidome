"""
QSAR Test Script for Avoidome Bioactivity Data
==============================================

This script provides a simple test environment for QSAR modeling with limited samples
to quickly validate the modeling pipeline before running on full datasets.

Features:
- Single Random Forest model training
- Limited sample size for fast testing
- Built-in test predictions on sample molecules
- Simple performance visualization
- Quick validation of the modeling pipeline

Usage:
    python qsar_test.py

Input:
    - avoidome_bioactivity_profile.csv: Bioactivity data with SMILES and pChEMBL values

Output:
    - Single trained model for testing
    - Performance plot
    - Test predictions on sample molecules
    - Validation of the complete pipeline

Parameters:
    - target_id: Target protein ID (default: P05177 - CYP1A2)
    - max_samples: Maximum samples to use (default: 50)

Author: QSAR Modeling System
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import warnings
warnings.filterwarnings('ignore')

class SimpleQSARTest:
    """
    Simple QSAR Test Class for Quick Pipeline Validation
    
    This class provides a simplified QSAR modeling environment for testing purposes.
    It uses limited data and a single model to quickly validate the complete pipeline.
    
    Features:
    - Fast training with limited samples
    - Built-in test predictions
    - Simple performance evaluation
    - Quick validation of the modeling workflow
    
    Attributes:
        data_path (str): Path to the bioactivity data CSV file
        output_dir (str): Directory to save test results
    """
    
    def __init__(self, data_path, output_dir):
        """
        Initialize the Simple QSAR Test
        
        Args:
            data_path (str): Path to the bioactivity data CSV file
            output_dir (str): Directory to save test results
        """
        self.data_path = data_path
        self.output_dir = output_dir
        
    def load_data(self):
        """Load and preprocess the avoidome bioactivity data"""
        print("Loading data...")
        self.df = pd.read_csv(self.data_path)
        
        # Filter for rows with valid SMILES and pChEMBL values
        self.df = self.df.dropna(subset=['canonical_smiles', 'pchembl_value'])
        self.df = self.df[self.df['pchembl_value'] > 0]  # Remove invalid pChEMBL values
        
        print(f"Loaded {len(self.df)} valid bioactivity records")
        print(f"Number of unique targets: {self.df['UniProt ID'].nunique()}")
        
    def calculate_descriptors(self, smiles_list):
        """Calculate molecular descriptors from SMILES - SIMPLE VERSION"""
        descriptors = []
        valid_indices = []
        
        print(f"Calculating descriptors for {len(smiles_list)} molecules...")
        
        for i, smiles in enumerate(smiles_list):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    # Calculate basic descriptors only
                    desc_dict = {}
                    
                    # Basic descriptors
                    desc_dict['MolWt'] = Descriptors.MolWt(mol)
                    desc_dict['LogP'] = Descriptors.MolLogP(mol)
                    desc_dict['NumHDonors'] = Descriptors.NumHDonors(mol)
                    desc_dict['NumHAcceptors'] = Descriptors.NumHAcceptors(mol)
                    desc_dict['NumRotatableBonds'] = Descriptors.NumRotatableBonds(mol)
                    desc_dict['TPSA'] = Descriptors.TPSA(mol)
                    desc_dict['NumAtoms'] = mol.GetNumAtoms()
                    desc_dict['NumHeavyAtoms'] = mol.GetNumHeavyAtoms()
                    desc_dict['NumRings'] = Descriptors.RingCount(mol)
                    desc_dict['NumAromaticRings'] = Descriptors.NumAromaticRings(mol)
                    
                    # Small Morgan fingerprint (128 bits for speed)
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=128)
                    fp_array = np.array(fp)
                    
                    # Add fingerprint bits to descriptors
                    for j, bit in enumerate(fp_array):
                        desc_dict[f'FP_{j}'] = bit
                    
                    descriptors.append(desc_dict)
                    valid_indices.append(i)
                    
            except Exception as e:
                print(f"Error processing SMILES {i}: {smiles[:50]}... - {str(e)}")
                continue
        
        print(f"Successfully calculated descriptors for {len(descriptors)} molecules")
        print(f"Feature dimensionality: {len(descriptors[0]) if descriptors else 0}")
        return descriptors, valid_indices
    
    def prepare_target_data(self, target_id, max_samples=100):
        """Prepare data for a specific target - LIMITED SAMPLES"""
        target_data = self.df[self.df['UniProt ID'] == target_id].copy()
        
        if len(target_data) < 10:
            return None, None, None
        
        # Limit samples for testing
        if len(target_data) > max_samples:
            target_data = target_data.sample(n=max_samples, random_state=42)
        
        print(f"\nProcessing target {target_id} with {len(target_data)} samples (limited for testing)")
        
        # Calculate molecular descriptors
        descriptors, valid_indices = self.calculate_descriptors(target_data['canonical_smiles'].tolist())
        
        if len(descriptors) < 10:
            return None, None, None
        
        # Create feature matrix
        desc_df = pd.DataFrame(descriptors)
        X = desc_df.values
        y = target_data['pchembl_value'].iloc[valid_indices].values
        
        return X, y, desc_df.columns
    
    def train_single_model(self, X, y, feature_names):
        """Train a single Random Forest model"""
        print("Training Random Forest model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model with fast parameters
        model = RandomForestRegressor(n_estimators=25, max_depth=8, n_jobs=-1, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='r2')
        
        results = {
            'model': model,
            'feature_names': feature_names,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mae': mae,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'y_test': y_test,
            'y_pred': y_pred
        }
        
        print(f"  R²: {r2:.3f}, RMSE: {rmse:.3f}, CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        return results
    
    def save_model(self, target_id, model_info):
        """Save trained model and metadata"""
        model_dir = os.path.join(self.output_dir, target_id)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, "RandomForest_model.pkl")
        joblib.dump(model_info['model'], model_path)
        
        # Save metadata
        metadata = {
            'target_id': target_id,
            'model_name': 'RandomForest',
            'feature_names': model_info['feature_names'].tolist(),
            'performance': {
                'mse': model_info['mse'],
                'rmse': model_info['rmse'],
                'r2': model_info['r2'],
                'mae': model_info['mae'],
                'cv_r2_mean': model_info['cv_r2_mean'],
                'cv_r2_std': model_info['cv_r2_std']
            }
        }
        
        metadata_path = os.path.join(model_dir, "RandomForest_metadata.json")
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved to {model_path}")
    
    def create_simple_plot(self, target_id, results):
        """Create a simple performance plot"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'QSAR Model Performance for Target {target_id}', fontsize=14)
        
        # Predictions vs actual
        y_test = results['y_test']
        y_pred = results['y_pred']
        
        ax1.scatter(y_test, y_pred, alpha=0.6)
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax1.set_xlabel('Actual pChEMBL')
        ax1.set_ylabel('Predicted pChEMBL')
        ax1.set_title(f'Predictions vs Actual (R² = {results["r2"]:.3f})')
        
        # Residuals plot
        residuals = y_test - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.6)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Predicted pChEMBL')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residuals Plot')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, target_id, f"{target_id}_performance.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Performance plot saved to {plot_path}")
    
    def test_prediction(self, target_id, test_smiles):
        """Test prediction with a few sample molecules"""
        print(f"\n{'='*50}")
        print(f"TESTING PREDICTIONS FOR TARGET {target_id}")
        print(f"{'='*50}")
        
        # Load the trained model
        model_dir = os.path.join(self.output_dir, target_id)
        model_path = os.path.join(model_dir, "RandomForest_model.pkl")
        metadata_path = os.path.join(model_dir, "RandomForest_metadata.json")
        
        if not os.path.exists(model_path):
            print("Model not found. Please train the model first.")
            return
        
        # Load model and metadata
        model = joblib.load(model_path)
        with open(metadata_path, 'r') as f:
            import json
            metadata = json.load(f)
        
        feature_names = metadata['feature_names']
        
        # Test predictions
        for i, smiles in enumerate(test_smiles):
            try:
                # Calculate descriptors
                descriptors, _ = self.calculate_descriptors([smiles])
                if not descriptors:
                    print(f"Test {i+1}: Invalid SMILES - {smiles[:50]}...")
                    continue
                
                # Create feature vector
                desc_dict = descriptors[0]
                features = pd.DataFrame([desc_dict])
                X = features[feature_names].values
                
                # Make prediction
                prediction = model.predict(X)[0]
                print(f"Test {i+1}: SMILES = {smiles[:50]}...")
                print(f"         Predicted pChEMBL = {prediction:.3f}")
                
            except Exception as e:
                print(f"Test {i+1}: Error - {str(e)}")
    
    def run_test(self, target_id="P05177", max_samples=50):
        """Run a simple test with one target and limited samples"""
        print("="*60)
        print("SIMPLE QSAR TEST - SINGLE MODEL, LIMITED SAMPLES")
        print("="*60)
        
        self.load_data()
        
        # Prepare data
        X, y, feature_names = self.prepare_target_data(target_id, max_samples)
        
        if X is None:
            print(f"No valid data found for target {target_id}")
            return
        
        # Train model
        results = self.train_single_model(X, y, feature_names)
        
        # Save model
        self.save_model(target_id, results)
        
        # Create plot
        self.create_simple_plot(target_id, results)
        
        # Test predictions with sample molecules
        test_smiles = [
            "CCc1ccc(C(=O)N2Cc3ccccc3C[C@H]2CN2CCOCC2)c(-c2cc(C(=O)N(c3ccc(O)cc3)c3ccc4c(ccn4C)c3)c3n2CCCC3)c1",
            "COc1ccc(NS(=O)(=O)c2ccc(Br)cc2)cc1N1CCN(C)CC1",
            "Cc1nc2cc(OC[C@H](O)CN3CCN(CC(=O)Nc4cccc(-c5ccccc5)c4)CC3)ccc2s1"
        ]
        
        self.test_prediction(target_id, test_smiles)
        
        print(f"\n{'='*60}")
        print("TEST COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")

if __name__ == "__main__":
    """
    Main execution block for QSAR testing
    
    This section runs a simple test of the QSAR modeling pipeline with limited data
    to quickly validate that everything works correctly before running on full datasets.
    
    Configuration:
    - data_path: Path to the bioactivity data CSV file
    - output_dir: Directory to save test results
    - target_id: Target protein ID to test (default: P05177 - CYP1A2)
    - max_samples: Maximum samples to use for testing (default: 50)
    
    Output:
    - Single trained Random Forest model for testing
    - Performance plot for the test model
    - Test predictions on sample molecules
    - Validation that the complete pipeline works
    """
    
    # Initialize and run simple test
    data_path = "/home/serramelendezcsm/RA/Avoidome/processed_data/avoidome_bioactivity_profile.csv"
    output_dir = "/home/serramelendezcsm/RA/Avoidome/analyses/qsar_avoidome"
    
    qsar_test = SimpleQSARTest(data_path, output_dir)
    qsar_test.run_test(target_id="P05177", max_samples=50)  # Test with 50 samples 