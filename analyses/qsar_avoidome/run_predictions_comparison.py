"""
QSAR Predictions Comparison Script for Avoidome Proteins
=======================================================

This script runs predictions on the Avoidome bioactivity data using trained QSAR models
and generates a comprehensive comparison table with actual vs predicted pChEMBL values.

Features:
- Loads bioactivity data for all Avoidome proteins
- Runs predictions using trained QSAR models (5CV)
- Uses global descriptor cache for fast predictions
- Generates comparison table with actual vs predicted values
- Calculates prediction errors and statistics
- Exports results to CSV files
- Provides detailed analysis and summary

Usage:
    python run_predictions_comparison.py

Input:
    - avoidome_bioactivity_profile.csv: Bioactivity data with SMILES and pChEMBL values
    - Trained QSAR models in qsar_avoidome directory
    - Global descriptor cache from training

Output:
    - actual_vs_predicted_summary.csv: Summary statistics per target
    - actual_vs_predicted_detailed.csv: Detailed comparison table
    - prediction_errors_analysis.csv: Error analysis and statistics
    - Console output with summary statistics

Author: QSAR Modeling System
Date: 2024
"""

import pandas as pd
import numpy as np
import os
import sys
import time
import joblib
from datetime import datetime
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Add the parent directory to the path to import the predictor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qsar_avoidome.predict import QSARPredictor

class CachedPredictionsComparison:
    """
    QSAR Predictions Comparison Class with Global Cache Support
    
    This class provides functionality to run predictions on Avoidome bioactivity data
    using trained QSAR models and the global descriptor cache for maximum efficiency.
    
    Features:
    - Load bioactivity data for all targets
    - Use global descriptor cache for fast predictions
    - Run predictions using trained models
    - Generate comparison tables
    - Calculate error statistics
    - Export results to CSV files
    
    Attributes:
        data_path (str): Path to the bioactivity data CSV file
        output_dir (str): Directory to save results
        predictor (QSARPredictor): QSAR predictor instance
        global_cache (dict): Global descriptor cache
    """
    
    def __init__(self, data_path, output_dir):
        """
        Initialize the Predictions Comparison
        
        Args:
            data_path (str): Path to the bioactivity data CSV file
            output_dir (str): Directory to save results
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.predictor = QSARPredictor()
        self.global_cache = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load global cache
        self.load_global_cache()
        
    def load_global_cache(self):
        """Load the global descriptor cache"""
        cache_file = os.path.join(self.predictor.models_dir, 'descriptor_cache', 'global_molecule_descriptors.pkl')
        
        if os.path.exists(cache_file):
            print(f"Loading global descriptor cache from: {cache_file}")
            try:
                self.global_cache = joblib.load(cache_file)
                print(f"Loaded {len(self.global_cache)} unique molecules from global cache")
            except Exception as e:
                print(f"Warning: Failed to load global cache: {str(e)}")
                self.global_cache = {}
        else:
            print("Warning: Global cache not found. Predictions will be slower.")
            self.global_cache = {}
    
    def get_cached_descriptors(self, smiles):
        """Get descriptors from cache or calculate if not available"""
        if smiles in self.global_cache:
            return self.global_cache[smiles]
        else:
            # Fallback to predictor's calculation method
            return self.predictor.calculate_descriptors(smiles)
    
    def predict_with_cache(self, smiles, target_id, model_name='RandomForest'):
        """Predict pChEMBL value using cached descriptors"""
        # Get descriptors from cache
        desc_dict = self.get_cached_descriptors(smiles)
        if desc_dict is None:
            return None, "Invalid SMILES or not in cache"
        
        # Load model
        try:
            model, scaler, metadata = self.predictor.load_model(target_id, model_name)
        except Exception as e:
            return None, str(e)
        
        # Create feature vector
        features = pd.DataFrame([desc_dict])
        feature_names = metadata['feature_names']
        
        # Check if all required features are present
        missing_features = set(feature_names) - set(features.columns)
        if missing_features:
            return None, f"Missing features: {missing_features}"
        
        X = features[feature_names].values
        
        # Scale if needed
        if scaler is not None:
            X = scaler.transform(X)
        
        # Make prediction
        try:
            prediction = model.predict(X)[0]
            return prediction, None
        except Exception as e:
            return None, f"Prediction error: {str(e)}"
    
    def load_bioactivity_data(self):
        """Load and preprocess the bioactivity data"""
        print("Loading bioactivity data...")
        start_time = time.time()
        
        # Load the data
        self.df = pd.read_csv(self.data_path)
        
        # Filter for rows with valid SMILES and pChEMBL values
        self.df = self.df.dropna(subset=['canonical_smiles', 'pchembl_value'])
        self.df = self.df[self.df['pchembl_value'] > 0]  # Remove invalid pChEMBL values
        
        # Get available targets with trained models
        available_targets = set(self.predictor.list_available_targets())
        
        # Filter data to only include targets with trained models
        self.df = self.df[self.df['UniProt ID'].isin(available_targets)]
        
        load_time = time.time() - start_time
        print(f"Loaded {len(self.df)} valid bioactivity records in {load_time:.2f}s")
        print(f"Number of unique targets: {self.df['UniProt ID'].nunique()}")
        print(f"Available targets with models: {len(available_targets)}")
        
        # Check cache coverage
        unique_smiles = set(self.df['canonical_smiles'].unique())
        cached_smiles = set(self.global_cache.keys())
        cache_coverage = len(unique_smiles.intersection(cached_smiles)) / len(unique_smiles) * 100
        print(f"Cache coverage: {cache_coverage:.1f}% ({len(unique_smiles.intersection(cached_smiles))}/{len(unique_smiles)} SMILES)")
        
        return self.df
    
    def run_predictions(self):
        """Run predictions for all targets using cached descriptors"""
        print("\nRunning predictions for all targets...")
        start_time = time.time()
        
        all_predictions = []
        target_stats = []
        
        # Group by target
        for target_id, target_data in self.df.groupby('UniProt ID'):
            print(f"Processing target: {target_id} ({len(target_data)} samples)")
            
            # Run predictions for this target
            predictions = []
            for idx, row in target_data.iterrows():
                smiles = row['canonical_smiles']
                actual_pchembl = row['pchembl_value']
                
                # Make prediction using cached descriptors
                predicted_pchembl, error = self.predict_with_cache(smiles, target_id)
                
                if predicted_pchembl is not None:
                    # Calculate errors
                    absolute_error = abs(actual_pchembl - predicted_pchembl)
                    percentage_error = (absolute_error / actual_pchembl) * 100
                    
                    # Categorize error
                    if absolute_error <= 0.5:
                        error_category = 'Excellent'
                    elif absolute_error <= 1.0:
                        error_category = 'Good'
                    elif absolute_error <= 1.5:
                        error_category = 'Moderate'
                    else:
                        error_category = 'Outlier'
                    
                    predictions.append({
                        'target_id': target_id,
                        'protein_name': row.get('Protein Name', ''),
                        'smiles': smiles,
                        'actual_pchembl': actual_pchembl,
                        'predicted_pchembl': predicted_pchembl,
                        'absolute_error': absolute_error,
                        'percentage_error': percentage_error,
                        'error_category': error_category,
                        'prediction_error': error
                    })
                else:
                    # Handle prediction errors
                    predictions.append({
                        'target_id': target_id,
                        'protein_name': row.get('Protein Name', ''),
                        'smiles': smiles,
                        'actual_pchembl': actual_pchembl,
                        'predicted_pchembl': None,
                        'absolute_error': None,
                        'percentage_error': None,
                        'error_category': 'Error',
                        'prediction_error': error
                    })
            
            # Calculate target statistics
            valid_predictions = [p for p in predictions if p['predicted_pchembl'] is not None]
            if valid_predictions:
                actual_values = [p['actual_pchembl'] for p in valid_predictions]
                predicted_values = [p['predicted_pchembl'] for p in valid_predictions]
                
                r2 = r2_score(actual_values, predicted_values)
                mae = mean_absolute_error(actual_values, predicted_values)
                rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
                
                target_stats.append({
                    'target_id': target_id,
                    'protein_name': valid_predictions[0]['protein_name'],
                    'total_samples': len(target_data),
                    'valid_predictions': len(valid_predictions),
                    'r2': r2,
                    'mae': mae,
                    'rmse': rmse,
                    'mean_absolute_error': np.mean([p['absolute_error'] for p in valid_predictions]),
                    'mean_percentage_error': np.mean([p['percentage_error'] for p in valid_predictions])
                })
            
            all_predictions.extend(predictions)
        
        prediction_time = time.time() - start_time
        print(f"Completed predictions in {prediction_time:.2f}s")
        
        return all_predictions, target_stats
    
    def generate_comparison_tables(self, all_predictions, target_stats):
        """Generate comparison tables and save to CSV files"""
        print("\nGenerating comparison tables...")
        
        # Create detailed comparison DataFrame
        detailed_df = pd.DataFrame(all_predictions)
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(target_stats)
        
        # Create error analysis DataFrame
        error_df = detailed_df[detailed_df['predicted_pchembl'].notna()].copy()
        
        # Save to CSV files
        detailed_file = os.path.join(self.output_dir, 'actual_vs_predicted_detailed.csv')
        summary_file = os.path.join(self.output_dir, 'actual_vs_predicted_summary.csv')
        error_file = os.path.join(self.output_dir, 'prediction_errors_analysis.csv')
        
        detailed_df.to_csv(detailed_file, index=False)
        summary_df.to_csv(summary_file, index=False)
        error_df.to_csv(error_file, index=False)
        
        print(f"Detailed comparison saved to: {detailed_file}")
        print(f"Summary statistics saved to: {summary_file}")
        print(f"Error analysis saved to: {error_file}")
        
        return detailed_df, summary_df, error_df
    
    def print_summary_statistics(self, detailed_df, summary_df):
        """Print summary statistics"""
        print("\n" + "="*60)
        print("PREDICTIONS COMPARISON SUMMARY")
        print("="*60)
        
        # Overall statistics
        total_predictions = len(detailed_df)
        valid_predictions = len(detailed_df[detailed_df['predicted_pchembl'].notna()])
        failed_predictions = total_predictions - valid_predictions
        
        print(f"Total predictions: {total_predictions}")
        print(f"Valid predictions: {valid_predictions} ({valid_predictions/total_predictions*100:.1f}%)")
        print(f"Failed predictions: {failed_predictions} ({failed_predictions/total_predictions*100:.1f}%)")
        
        if valid_predictions > 0:
            # Error statistics
            valid_data = detailed_df[detailed_df['predicted_pchembl'].notna()]
            mean_absolute_error = valid_data['absolute_error'].mean()
            mean_percentage_error = valid_data['percentage_error'].mean()
            max_error = valid_data['absolute_error'].max()
            
            print(f"\nError Statistics:")
            print(f"Mean Absolute Error: {mean_absolute_error:.3f}")
            print(f"Mean Percentage Error: {mean_percentage_error:.1f}%")
            print(f"Maximum Error: {max_error:.3f}")
            
            # Error category distribution
            error_categories = valid_data['error_category'].value_counts()
            print(f"\nError Category Distribution:")
            for category, count in error_categories.items():
                percentage = (count / len(valid_data)) * 100
                print(f"  {category}: {count} ({percentage:.1f}%)")
        
        # Target statistics
        if not summary_df.empty:
            print(f"\nTarget Performance Summary:")
            print(f"Number of targets: {len(summary_df)}")
            print(f"Average R²: {summary_df['r2'].mean():.3f}")
            print(f"Average MAE: {summary_df['mae'].mean():.3f}")
            print(f"Average RMSE: {summary_df['rmse'].mean():.3f}")
            
            # Best and worst performing targets
            best_target = summary_df.loc[summary_df['r2'].idxmax()]
            worst_target = summary_df.loc[summary_df['r2'].idxmin()]
            
            print(f"\nBest performing target: {best_target['target_id']} (R² = {best_target['r2']:.3f})")
            print(f"Worst performing target: {worst_target['target_id']} (R² = {worst_target['r2']:.3f})")
        
        print("="*60)
    
    def run_complete_analysis(self):
        """Run the complete predictions comparison analysis"""
        print("QSAR Predictions Comparison Analysis (with Global Cache)")
        print("="*60)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load data
        self.load_bioactivity_data()
        
        # Run predictions
        all_predictions, target_stats = self.run_predictions()
        
        # Generate comparison tables
        detailed_df, summary_df, error_df = self.generate_comparison_tables(all_predictions, target_stats)
        
        # Print summary statistics
        self.print_summary_statistics(detailed_df, summary_df)
        
        print(f"\nAnalysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("Results saved to:", self.output_dir)

def main():
    """
    Main function for QSAR predictions comparison
    
    This function initializes and runs the complete predictions comparison analysis
    for the Avoidome bioactivity dataset using the global descriptor cache.
    
    Configuration:
    - data_path: Path to the bioactivity data CSV file
    - output_dir: Directory to save results
    
    Output:
    - Detailed comparison table with actual vs predicted values
    - Summary statistics per target
    - Error analysis and statistics
    - Console output with comprehensive summary
    """
    
    # Configuration
    data_path = "/home/serramelendezcsm/RA/Avoidome/processed_data/avoidome_bioactivity_profile.csv"
    output_dir = "/home/serramelendezcsm/RA/Avoidome/analyses/qsar_avoidome/prediction_reports"
    
    # Create and run analysis
    analysis = CachedPredictionsComparison(data_path, output_dir)
    analysis.run_complete_analysis()

if __name__ == "__main__":
    main() 