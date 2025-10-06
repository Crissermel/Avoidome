"""
QSAR Modeling Fast Script for Avoidome Bioactivity Data
=======================================================

This script provides a faster version of QSAR modeling with optimized parameters
for quicker training while maintaining good performance.

Features:
- Multiple machine learning models (Random Forest, Gradient Boosting, Ridge, Lasso, SVR)
- Reduced feature dimensionality for speed (256-bit fingerprints)
- Optimized model parameters for faster training
- Limited target processing for testing
- Performance comparison between models

Usage:
    python qsar_modeling_fast.py

Input:
    - avoidome_bioactivity_profile.csv: Bioactivity data with SMILES and pChEMBL values

Output:
    - Multiple trained models for comparison
    - Performance plots and metrics
    - Model comparison summary
    - Saved models for each target

Parameters:
    - min_samples: Minimum samples per target (default: 50)
    - max_targets: Maximum targets to process (default: 3)

Author: QSAR Modeling System
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import os
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import warnings
warnings.filterwarnings('ignore')

class FastQSARModelBuilder:
    """
    Fast QSAR Model Builder for Quick Model Comparison
    
    This class provides a faster version of QSAR modeling with multiple algorithms
    for comparison. It uses optimized parameters and reduced feature dimensionality
    to speed up training while maintaining good performance.
    
    Features:
    - Multiple ML algorithms for comparison
    - Optimized parameters for speed
    - Reduced feature dimensionality
    - Performance comparison between models
    - Limited target processing for testing
    
    Attributes:
        data_path (str): Path to the bioactivity data CSV file
        output_dir (str): Directory to save models and results
        models (dict): Dictionary to store trained models
        results (dict): Dictionary to store model results
    """
    
    def __init__(self, data_path, output_dir):
        """
        Initialize the Fast QSAR Model Builder
        
        Args:
            data_path (str): Path to the bioactivity data CSV file
            output_dir (str): Directory to save models and results
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load and preprocess the avoidome bioactivity data"""
        print("Loading data...")
        self.df = pd.read_csv(self.data_path)
        
        # Filter for rows with valid SMILES and pChEMBL values
        self.df = self.df.dropna(subset=['canonical_smiles', 'pchembl_value'])
        self.df = self.df[self.df['pchembl_value'] > 0]  # Remove invalid pChEMBL values
        
        print(f"Loaded {len(self.df)} valid bioactivity records")
        print(f"Number of unique targets: {self.df['UniProt ID'].nunique()}")
        
    def calculate_molecular_descriptors(self, smiles_list):
        """Calculate molecular descriptors from SMILES - FAST VERSION"""
        descriptors = []
        valid_indices = []
        
        print(f"Calculating descriptors for {len(smiles_list)} molecules...")
        
        for i, smiles in enumerate(smiles_list):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    # Calculate basic descriptors only (no fingerprints for speed)
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
                    
                    # Add a few more useful descriptors
                    desc_dict['FractionCsp3'] = Descriptors.FractionCsp3(mol)
                    desc_dict['HeavyAtomCount'] = Descriptors.HeavyAtomCount(mol)
                    desc_dict['RingCount'] = Descriptors.RingCount(mol)
                    desc_dict['AromaticRings'] = Descriptors.NumAromaticRings(mol)
                    desc_dict['SaturatedRings'] = Descriptors.NumSaturatedRings(mol)
                    desc_dict['Heteroatoms'] = Descriptors.NumHeteroatoms(mol)
                    
                    # Small Morgan fingerprint (256 bits instead of 2048)
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=256)
                    fp_array = np.array(fp)
                    
                    # Add fingerprint bits to descriptors
                    for j, bit in enumerate(fp_array):
                        desc_dict[f'FP_{j}'] = bit
                    
                    descriptors.append(desc_dict)
                    valid_indices.append(i)
                    
            except Exception as e:
                if i < 5:  # Only print first few errors to avoid spam
                    print(f"Error processing SMILES {i}: {smiles[:50]}... - {str(e)}")
                continue
        
        print(f"Successfully calculated descriptors for {len(descriptors)} molecules")
        print(f"Feature dimensionality: {len(descriptors[0]) if descriptors else 0}")
        return descriptors, valid_indices
    
    def prepare_target_data(self, target_id, min_samples=50):
        """Prepare data for a specific target"""
        target_data = self.df[self.df['UniProt ID'] == target_id].copy()
        
        if len(target_data) < min_samples:
            return None, None, None
        
        print(f"\nProcessing target {target_id} with {len(target_data)} samples")
        
        # Calculate molecular descriptors
        descriptors, valid_indices = self.calculate_molecular_descriptors(target_data['canonical_smiles'].tolist())
        
        if len(descriptors) < min_samples:
            return None, None, None
        
        # Create feature matrix
        desc_df = pd.DataFrame(descriptors)
        X = desc_df.values
        y = target_data['pchembl_value'].iloc[valid_indices].values
        
        return X, y, desc_df.columns
    
    def train_models(self, X, y, feature_names):
        """Train multiple QSAR models - FAST VERSION"""
        models = {}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models with faster parameters
        model_configs = {
            'RandomForest': RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=50, max_depth=6, random_state=42),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.1, max_iter=1000),
            'SVR': SVR(kernel='rbf', C=1.0, gamma='scale', max_iter=1000)
        }
        
        results = {}
        
        for name, model in model_configs.items():
            print(f"Training {name}...")
            
            # Train model
            if name in ['Ridge', 'Lasso', 'SVR']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            # Cross-validation (3-fold for speed)
            if name in ['Ridge', 'Lasso', 'SVR']:
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring='r2')
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='r2')
            
            results[name] = {
                'model': model,
                'scaler': scaler if name in ['Ridge', 'Lasso', 'SVR'] else None,
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
    
    def save_model(self, target_id, model_name, model_info):
        """Save trained model and metadata"""
        model_dir = os.path.join(self.output_dir, target_id)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, f"{model_name}_model.pkl")
        joblib.dump(model_info['model'], model_path)
        
        # Save scaler if exists
        if model_info['scaler'] is not None:
            scaler_path = os.path.join(model_dir, f"{model_name}_scaler.pkl")
            joblib.dump(model_info['scaler'], scaler_path)
        
        # Save metadata
        metadata = {
            'target_id': target_id,
            'model_name': model_name,
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
        
        metadata_path = os.path.join(model_dir, f"{model_name}_metadata.json")
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def create_performance_plots(self, target_id, results):
        """Create performance visualization plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'QSAR Model Performance for Target {target_id}', fontsize=16)
        
        # Model comparison
        model_names = list(results.keys())
        r2_scores = [results[name]['r2'] for name in model_names]
        rmse_scores = [results[name]['rmse'] for name in model_names]
        
        # R² comparison
        axes[0, 0].bar(model_names, r2_scores)
        axes[0, 0].set_title('R² Scores Comparison')
        axes[0, 0].set_ylabel('R²')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # RMSE comparison
        axes[0, 1].bar(model_names, rmse_scores)
        axes[0, 1].set_title('RMSE Comparison')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Best model predictions vs actual
        best_model = max(results.keys(), key=lambda x: results[x]['r2'])
        y_test = results[best_model]['y_test']
        y_pred = results[best_model]['y_pred']
        
        axes[1, 0].scatter(y_test, y_pred, alpha=0.6)
        axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[1, 0].set_xlabel('Actual pChEMBL')
        axes[1, 0].set_ylabel('Predicted pChEMBL')
        axes[1, 0].set_title(f'Predictions vs Actual ({best_model})')
        
        # Residuals plot
        residuals = y_test - y_pred
        axes[1, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Predicted pChEMBL')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residuals Plot')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, target_id, f"{target_id}_performance.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_modeling(self, min_samples=50, max_targets=5):
        """Run the complete QSAR modeling pipeline - FAST VERSION"""
        self.load_data()
        
        # Get targets with sufficient data
        target_counts = self.df['UniProt ID'].value_counts()
        valid_targets = target_counts[target_counts >= min_samples].index
        
        print(f"\nFound {len(valid_targets)} targets with at least {min_samples} samples")
        print(f"Processing first {max_targets} targets for speed...")
        
        all_results = {}
        
        for i, target_id in enumerate(valid_targets[:max_targets]):
            try:
                print(f"\n{'='*50}")
                print(f"Processing target {i+1}/{max_targets}: {target_id}")
                print(f"{'='*50}")
                
                X, y, feature_names = self.prepare_target_data(target_id, min_samples)
                
                if X is None:
                    continue
                
                # Train models
                results = self.train_models(X, y, feature_names)
                
                # Save models
                for model_name, model_info in results.items():
                    self.save_model(target_id, model_name, model_info)
                
                # Create plots
                self.create_performance_plots(target_id, results)
                
                all_results[target_id] = results
                
                print(f"Completed modeling for {target_id}")
                
            except Exception as e:
                print(f"Error processing target {target_id}: {str(e)}")
                continue
        
        # Create summary report
        self.create_summary_report(all_results)
        
        return all_results
    
    def create_summary_report(self, all_results):
        """Create a summary report of all models"""
        summary_data = []
        
        for target_id, results in all_results.items():
            for model_name, model_info in results.items():
                summary_data.append({
                    'Target_ID': target_id,
                    'Model': model_name,
                    'R2': model_info['r2'],
                    'RMSE': model_info['rmse'],
                    'MAE': model_info['mae'],
                    'CV_R2_Mean': model_info['cv_r2_mean'],
                    'CV_R2_Std': model_info['cv_r2_std']
                })
        
        if not summary_data:
            print("\nNo models were successfully trained. Check the data quality and error messages above.")
            return
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(self.output_dir, 'model_performance_summary.csv'), index=False)
        
        print(f"\nSummary report saved to {os.path.join(self.output_dir, 'model_performance_summary.csv')}")
        
        # Print best models
        print("\nBest performing models:")
        best_models = summary_df.loc[summary_df.groupby('Target_ID')['R2'].idxmax()]
        print(best_models[['Target_ID', 'Model', 'R2', 'RMSE']].to_string(index=False))

if __name__ == "__main__":
    """
    Main execution block for Fast QSAR modeling
    
    This section runs a faster version of QSAR modeling with multiple algorithms
    for comparison. It uses optimized parameters and limited targets for quick testing.
    
    Configuration:
    - data_path: Path to the bioactivity data CSV file
    - output_dir: Directory to save models and results
    - min_samples: Minimum number of samples required per target (default: 50)
    - max_targets: Maximum number of targets to process (default: 3 for speed)
    
    Output:
    - Multiple trained models (Random Forest, Gradient Boosting, Ridge, Lasso, SVR)
    - Performance comparison between models
    - Performance plots and metrics
    - Model comparison summary
    
    Note: This version is optimized for speed and comparison rather than production use.
    """
    
    # Initialize and run QSAR modeling - FAST VERSION
    data_path = "/home/serramelendezcsm/RA/Avoidome/processed_data/avoidome_bioactivity_profile.csv"
    output_dir = "/home/serramelendezcsm/RA/Avoidome/analyses/qsar_avoidome"
    
    qsar_builder = FastQSARModelBuilder(data_path, output_dir)
    results = qsar_builder.run_modeling(min_samples=50, max_targets=3)  # Process only 3 targets for speed 