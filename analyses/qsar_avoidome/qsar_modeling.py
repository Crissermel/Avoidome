"""
QSAR Modeling Script for Avoidome Bioactivity Data
==================================================

This script trains Random Forest QSAR models to predict pChEMBL values from molecular structures
(SMILES) for single targets in the avoidome dataset.

Features:
- Single Random Forest model training (optimized for speed and performance)
- Comprehensive progress tracking and timing
- Detailed logging to files and console
- Performance visualization with plots
- Model persistence for reuse
- Support for multiple targets

Usage:
    python qsar_modeling.py

Input:
    - avoidome_bioactivity_profile.csv: Bioactivity data with SMILES and pChEMBL values

Output:
    - Trained models saved as .pkl files
    - Performance metadata as .json files
    - Performance plots as .png files
    - Summary report as .csv file
    - Detailed logs in logs/ directory


Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import warnings
import logging
import time
from datetime import datetime
warnings.filterwarnings('ignore')

class QSARModelBuilder:
    """
    QSAR Model Builder for Avoidome Bioactivity Data
    
    This class handles the complete QSAR modeling pipeline including:
    - Data loading and preprocessing
    - Molecular descriptor calculation
    - Random Forest model training
    - Performance evaluation
    - Model saving and visualization
    
    Attributes:
        data_path (str): Path to the bioactivity data CSV file
        output_dir (str): Directory to save models and results
        models (dict): Dictionary to store trained models
        results (dict): Dictionary to store model results
        logger: Logging object for progress tracking
    """
    
    def __init__(self, data_path, output_dir):
        """
        Initialize the QSAR Model Builder
        
        Args:
            data_path (str): Path to the bioactivity data CSV file
            output_dir (str): Directory to save models and results
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.models = {}
        self.results = {}
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = os.path.join(self.output_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f'qsar_modeling_{timestamp}.log')
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()  # Also print to console
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"QSAR Modeling started - Log file: {log_file}")
        
    def load_data(self):
        """Load and preprocess the avoidome bioactivity data"""
        self.logger.info("Loading data...")
        start_time = time.time()
        
        self.df = pd.read_csv(self.data_path)
        
        # Filter for rows with valid SMILES and pChEMBL values
        self.df = self.df.dropna(subset=['canonical_smiles', 'pchembl_value'])
        self.df = self.df[self.df['pchembl_value'] > 0]  # Remove invalid pChEMBL values
        
        load_time = time.time() - start_time
        self.logger.info(f"Loaded {len(self.df)} valid bioactivity records in {load_time:.2f}s")
        self.logger.info(f"Number of unique targets: {self.df['UniProt ID'].nunique()}")
        
    def calculate_molecular_descriptors(self, smiles_list, target_id=None, force_recalculate=False):
        """
        Calculate molecular descriptors from SMILES with global molecule caching
        
        Args:
            smiles_list: List of SMILES strings
            target_id: Target ID for logging (optional)
            force_recalculate: Force recalculation even if cached (default: False)
        
        Returns:
            descriptors: List of descriptor dictionaries
            valid_indices: List of valid molecule indices
        """
        descriptors = []
        valid_indices = []
        
        # Check global molecule cache first
        global_cache_file = os.path.join(self.output_dir, 'descriptor_cache', 'global_molecule_descriptors.pkl')
        
        if not force_recalculate and os.path.exists(global_cache_file):
            self.logger.info(f"Loading global molecule descriptor cache...")
            try:
                global_cache = joblib.load(global_cache_file)
                self.logger.info(f"Global cache contains {len(global_cache)} unique molecules")
                
                # Look up descriptors for each SMILES
                for i, smiles in enumerate(smiles_list):
                    if smiles in global_cache:
                        descriptors.append(global_cache[smiles])
                        valid_indices.append(i)
                    else:
                        self.logger.warning(f"SMILES not found in global cache: {smiles[:50]}...")
                
                if descriptors:
                    self.logger.info(f"Successfully loaded {len(descriptors)} descriptors from global cache")
                    return descriptors, valid_indices
                    
            except Exception as e:
                self.logger.warning(f"Failed to load global cache: {str(e)}")
        
        # Calculate descriptors if not in global cache or forced
        self.logger.info(f"Calculating descriptors for {len(smiles_list)} molecules...")
        start_time = time.time()
        
        # Load existing global cache if available
        global_cache = {}
        if os.path.exists(global_cache_file) and not force_recalculate:
            try:
                global_cache = joblib.load(global_cache_file)
                self.logger.info(f"Loaded existing global cache with {len(global_cache)} molecules")
            except:
                global_cache = {}
        
        new_molecules = 0
        for i, smiles in enumerate(smiles_list):
            try:
                # Check if already in global cache
                if smiles in global_cache and not force_recalculate:
                    descriptors.append(global_cache[smiles])
                    valid_indices.append(i)
                    continue
                
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    # Calculate basic descriptors
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
                    
                    # Morgan fingerprints (512 bits for speed)
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
                    fp_array = np.array(fp)
                    
                    # Add fingerprint bits to descriptors
                    for j, bit in enumerate(fp_array):
                        desc_dict[f'FP_{j}'] = bit
                    
                    descriptors.append(desc_dict)
                    valid_indices.append(i)
                    
                    # Add to global cache
                    global_cache[smiles] = desc_dict
                    new_molecules += 1
                    
            except Exception as e:
                if i < 5:  # Only log first few errors to avoid spam
                    self.logger.warning(f"Error processing SMILES {i}: {smiles[:50]}... - {str(e)}")
                continue
        
        calc_time = time.time() - start_time
        self.logger.info(f"Successfully calculated descriptors for {len(descriptors)} molecules in {calc_time:.2f}s")
        self.logger.info(f"Added {new_molecules} new molecules to global cache")
        self.logger.info(f"Global cache now contains {len(global_cache)} unique molecules")
        self.logger.info(f"Feature dimensionality: {len(descriptors[0]) if descriptors else 0}")
        
        # Save updated global cache
        if new_molecules > 0:
            self._save_global_cache(global_cache)
        
        return descriptors, valid_indices
    
    def _save_global_cache(self, global_cache):
        """Save global molecule descriptor cache to disk"""
        cache_dir = os.path.join(self.output_dir, 'descriptor_cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        cache_file = os.path.join(cache_dir, 'global_molecule_descriptors.pkl')
        cache_data = {
            'descriptors': global_cache,
            'molecule_count': len(global_cache),
            'timestamp': datetime.now().isoformat(),
            'feature_count': len(list(global_cache.values())[0]) if global_cache else 0
        }
        
        try:
            joblib.dump(global_cache, cache_file)
            self.logger.info(f"Global cache saved with {len(global_cache)} molecules")
        except Exception as e:
            self.logger.warning(f"Failed to save global cache: {str(e)}")
    
    def get_global_cache_info(self):
        """Get information about global molecule cache"""
        global_cache_file = os.path.join(self.output_dir, 'descriptor_cache', 'global_molecule_descriptors.pkl')
        
        if not os.path.exists(global_cache_file):
            return None
        
        try:
            global_cache = joblib.load(global_cache_file)
            return {
                'molecule_count': len(global_cache),
                'file_size_mb': os.path.getsize(global_cache_file) / (1024 * 1024),
                'feature_count': len(list(global_cache.values())[0]) if global_cache else 0,
                'sample_smiles': list(global_cache.keys())[:5] if global_cache else []
            }
        except Exception as e:
            self.logger.warning(f"Failed to read global cache: {str(e)}")
            return None
    
    def get_cached_targets(self):
        """Get list of targets with cached descriptors (legacy method)"""
        cache_dir = os.path.join(self.output_dir, 'descriptor_cache')
        if not os.path.exists(cache_dir):
            return []
        
        cached_targets = []
        for file in os.listdir(cache_dir):
            if file.endswith('_descriptors.pkl') and not file.startswith('global'):
                target_id = file.replace('_descriptors.pkl', '')
                cached_targets.append(target_id)
        
        return cached_targets
    
    def prepare_target_data(self, target_id, min_samples=50):
        """Prepare data for a specific target"""
        target_data = self.df[self.df['UniProt ID'] == target_id].copy()
        
        if len(target_data) < min_samples:
            self.logger.warning(f"Target {target_id} has insufficient data: {len(target_data)} < {min_samples}")
            return None, None, None
        
        self.logger.info(f"Processing target {target_id} with {len(target_data)} samples")
        
        # Calculate molecular descriptors (with caching)
        descriptors, valid_indices = self.calculate_molecular_descriptors(
            target_data['canonical_smiles'].tolist(), 
            target_id=target_id
        )
        
        if len(descriptors) < min_samples:
            self.logger.warning(f"Target {target_id} has insufficient valid descriptors: {len(descriptors)} < {min_samples}")
            return None, None, None
        
        # Create feature matrix
        desc_df = pd.DataFrame(descriptors)
        X = desc_df.values
        y = target_data['pchembl_value'].iloc[valid_indices].values
        
        self.logger.info(f"Feature matrix shape: {X.shape}, Target vector shape: {y.shape}")
        return X, y, desc_df.columns
    
    def train_models(self, X, y, feature_names):
        """Train Random Forest QSAR model"""
        self.logger.info("Training Random Forest model...")
        start_time = time.time()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.logger.info(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
        
        # Train Random Forest model
        model = RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        
        train_time = time.time() - start_time
        
        results = {
            'RandomForest': {
                'model': model,
                'scaler': None,  # Random Forest doesn't need scaling
                'feature_names': feature_names,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'mae': mae,
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std(),
                'y_test': y_test,
                'y_pred': y_pred,
                'training_time': train_time
            }
        }
        
        self.logger.info(f"Training completed in {train_time:.2f}s")
        self.logger.info(f"R²: {r2:.3f}, RMSE: {rmse:.3f}, CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        return results
    
    def save_model(self, target_id, model_name, model_info):
        """Save trained model and metadata"""
        model_dir = os.path.join(self.output_dir, target_id)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, f"{model_name}_model.pkl")
        joblib.dump(model_info['model'], model_path)
        self.logger.info(f"Model saved to {model_path}")
        
        # Save scaler if exists
        if model_info['scaler'] is not None:
            scaler_path = os.path.join(model_dir, f"{model_name}_scaler.pkl")
            joblib.dump(model_info['scaler'], scaler_path)
            self.logger.info(f"Scaler saved to {scaler_path}")
        
        # Save predictions for comparison analysis
        if 'y_test' in model_info and 'y_pred' in model_info:
            predictions_df = pd.DataFrame({
                'actual_pchembl': model_info['y_test'],
                'predicted_pchembl': model_info['y_pred'],
                'absolute_error': np.abs(model_info['y_test'] - model_info['y_pred']),
                'percentage_error': (np.abs(model_info['y_test'] - model_info['y_pred']) / model_info['y_test']) * 100
            })
            
            predictions_path = os.path.join(model_dir, 'predictions.csv')
            predictions_df.to_csv(predictions_path, index=False)
            self.logger.info(f"Predictions saved to {predictions_path}")
        
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
                'cv_r2_std': model_info['cv_r2_std'],
                'training_time': model_info.get('training_time', 0)
            }
        }
        
        metadata_path = os.path.join(model_dir, f"{model_name}_metadata.json")
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        self.logger.info(f"Metadata saved to {metadata_path}")
    
    def create_performance_plots(self, target_id, results):
        """Create performance visualization plots"""
        self.logger.info("Creating performance plots...")
        
        # Get Random Forest results
        rf_results = results['RandomForest']
        y_test = rf_results['y_test']
        y_pred = rf_results['y_pred']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Random Forest QSAR Model Performance for Target {target_id}', fontsize=14)
        
        # Predictions vs actual
        ax1.scatter(y_test, y_pred, alpha=0.6)
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax1.set_xlabel('Actual pChEMBL')
        ax1.set_ylabel('Predicted pChEMBL')
        ax1.set_title(f'Predictions vs Actual (R² = {rf_results["r2"]:.3f})')
        
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
        self.logger.info(f"Performance plot saved to {plot_path}")
    
    def run_modeling(self, min_samples=50, max_targets=None):
        """Run the complete QSAR modeling pipeline"""
        self.load_data()
        
        # Check for cached descriptors
        cached_targets = self.get_cached_targets()
        if cached_targets:
            self.logger.info(f"Found {len(cached_targets)} targets with cached descriptors: {cached_targets}")
        
        # Get targets with sufficient data
        target_counts = self.df['UniProt ID'].value_counts()
        valid_targets = target_counts[target_counts >= min_samples].index
        
        self.logger.info(f"Found {len(valid_targets)} targets with at least {min_samples} samples")
        
        if max_targets:
            valid_targets = valid_targets[:max_targets]
            self.logger.info(f"Processing first {max_targets} targets")
        
        all_results = {}
        total_start_time = time.time()
        
        for i, target_id in enumerate(valid_targets):
            try:
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"Processing target {i+1}/{len(valid_targets)}: {target_id}")
                self.logger.info(f"{'='*60}")
                
                target_start_time = time.time()
                
                X, y, feature_names = self.prepare_target_data(target_id, min_samples)
                
                if X is None:
                    continue
                
                # Train model
                results = self.train_models(X, y, feature_names)
                
                # Save model
                for model_name, model_info in results.items():
                    self.save_model(target_id, model_name, model_info)
                
                # Create plots
                self.create_performance_plots(target_id, results)
                
                all_results[target_id] = results
                
                
                target_time = time.time() - target_start_time
                self.logger.info(f"Completed modeling for {target_id} in {target_time:.2f}s")
                
            except Exception as e:
                self.logger.error(f"Error processing target {target_id}: {str(e)}")
                continue
        
        total_time = time.time() - total_start_time
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"TOTAL PROCESSING TIME: {total_time:.2f}s ({total_time/60:.1f} minutes)")
        self.logger.info(f"Successfully processed {len(all_results)} targets")
        self.logger.info(f"{'='*60}")
        
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
                    'CV_R2_Std': model_info['cv_r2_std'],
                    'Training_Time': model_info.get('training_time', 0)
                })
        
        if not summary_data:
            self.logger.warning("No models were successfully trained. Check the data quality and error messages above.")
            return
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(self.output_dir, 'model_performance_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        
        self.logger.info(f"Summary report saved to {summary_path}")
        
        # Print summary statistics
        self.logger.info(f"\nSummary Statistics:")
        self.logger.info(f"Total models trained: {len(summary_df)}")
        self.logger.info(f"Average R²: {summary_df['R2'].mean():.3f}")
        self.logger.info(f"Average RMSE: {summary_df['RMSE'].mean():.3f}")
        self.logger.info(f"Best R²: {summary_df['R2'].max():.3f}")
        self.logger.info(f"Worst R²: {summary_df['R2'].min():.3f}")
        
        # Show top performing models
        top_models = summary_df.nlargest(5, 'R2')[['Target_ID', 'R2', 'RMSE']]
        self.logger.info(f"\nTop 5 performing models:")
        for _, row in top_models.iterrows():
            self.logger.info(f"  {row['Target_ID']}: R² = {row['R2']:.3f}, RMSE = {row['RMSE']:.3f}")

if __name__ == "__main__":
    """
    Main execution block for QSAR modeling
    
    This section initializes the QSAR modeling pipeline and runs it on the avoidome dataset.
    It processes ALL targets with sufficient data and creates trained models with performance evaluation.
    
    Configuration:
    - data_path: Path to the bioactivity data CSV file
    - output_dir: Directory to save models and results
    - min_samples: Minimum number of samples required per target (default: 50)
    - max_targets: None (processes ALL targets with sufficient data)
    
    Output:
    - Trained Random Forest models for ALL targets (33+ targets)
    - Performance plots and metrics for each target
    - Comprehensive summary report with all results
    - Detailed logs of the entire process
    - Complete QSAR modeling system for avoidome dataset
    """
    
    # Initialize and run QSAR modeling
    data_path = "/home/serramelendezcsm/RA/Avoidome/processed_data/avoidome_bioactivity_profile.csv"
    output_dir = "/home/serramelendezcsm/RA/Avoidome/analyses/qsar_avoidome"
    
    qsar_builder = QSARModelBuilder(data_path, output_dir)
    results = qsar_builder.run_modeling(min_samples=50)  # Process ALL targets with at least 50 samples 