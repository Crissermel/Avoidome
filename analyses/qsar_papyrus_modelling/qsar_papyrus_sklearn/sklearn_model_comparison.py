#!/usr/bin/env python3
"""
Scikit-learn Model Comparison for Papyrus QSAR Data
This script compares different regression models using scikit-learn instead of PyCaret.
"""

import pandas as pd
import numpy as np
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_modelling/qsar_papyrus_sklearn/sklearn_model_comparison.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SklearnModelComparison:
    """Scikit-learn based model comparison for Papyrus QSAR data"""
    
    def __init__(self, data_path: str = "/home/serramelendezcsm/RA/Avoidome/processed_data/papyrus_protein_check_results.csv"):
        self.data_path = data_path
        self.results_dir = Path("/home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_modelling/qsar_papyrus_sklearn/sklearn_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def load_and_prepare_data(self, protein_name: str = "CYP1A2") -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Load and prepare data for a specific protein using the existing Papyrus pipeline
        """
        logger.info(f"Loading data for protein: {protein_name}")
        
        # Import the existing Papyrus model to get the data
        import sys
        sys.path.append('/home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_modelling')
        from minimal_papyrus_prediction import PapyrusQSARModel
        
        # Initialize the existing model
        papyrus_model = PapyrusQSARModel()
        papyrus_model.load_data()
        
        # Get the protein row
        proteins_df = papyrus_model.proteins_df
        protein_row = proteins_df[proteins_df['name2_entry'] == protein_name]
        
        if len(protein_row) == 0:
            raise ValueError(f"Protein {protein_name} not found in dataset")
        
        # Get raw data directly from the existing pipeline
        human_id = protein_row.iloc[0]['human_uniprot_id']
        mouse_id = protein_row.iloc[0]['mouse_uniprot_id']
        rat_id = protein_row.iloc[0]['rat_uniprot_id']
        
        # Collect all activities
        all_activities = []
        for uniprot_id in [human_id, mouse_id, rat_id]:
            if pd.notna(uniprot_id) and uniprot_id:
                activities = papyrus_model.get_protein_activities(uniprot_id)
                if activities is not None and len(activities) > 0:
                    all_activities.append(activities)
        
        if not all_activities:
            raise ValueError(f"No bioactivity data available for {protein_name}")
        
        # Combine all activities
        combined_activities = pd.concat(all_activities, ignore_index=True)
        logger.info(f"Combined {len(combined_activities)} activities for {protein_name}")
        
        # Check if we have the required columns
        if 'pchembl_value' not in combined_activities.columns:
            raise ValueError(f"Missing required column 'pchembl_value' for {protein_name}")
        
        if 'SMILES' not in combined_activities.columns:
            raise ValueError(f"Missing required column 'SMILES' for {protein_name}")
        
        # Clean and prepare data
        combined_activities = combined_activities.dropna(subset=['pchembl_value'])
        combined_activities = combined_activities.drop_duplicates(subset=['SMILES', 'pchembl_value'])
        
        # Clean pchembl_value (handle semicolon-separated values)
        combined_activities['pchembl_value_numeric'] = combined_activities['pchembl_value'].apply(
            lambda x: float(str(x).split(';')[0]) if pd.notna(x) else None
        )
        combined_activities = combined_activities.dropna(subset=['pchembl_value_numeric'])
        
        if len(combined_activities) == 0:
            raise ValueError(f"No valid bioactivity data available for {protein_name}")
        
        # Generate Morgan fingerprints
        from rdkit import Chem
        from rdkit.Chem import AllChem
        
        fingerprints = []
        valid_indices = []
        
        for idx, row in combined_activities.iterrows():
            try:
                mol = Chem.MolFromSmiles(row['SMILES'])
                if mol is not None:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                    # Convert to dense array
                    fp_array = np.zeros(2048)
                    for bit in fp.GetOnBits():
                        fp_array[bit] = 1
                    fingerprints.append(fp_array)
                    valid_indices.append(idx)
            except:
                continue
        
        if len(fingerprints) == 0:
            raise ValueError(f"Could not generate fingerprints for {protein_name}")
        
        # Convert to numpy arrays
        X = np.array(fingerprints)
        y = combined_activities.iloc[valid_indices]['pchembl_value_numeric'].values
        
        logger.info(f"Prepared dataset with {len(X)} samples and {X.shape[1]} features")
        
        return X, y, protein_name
    
    def run_sklearn_comparison(self, X: np.ndarray, y: np.ndarray, protein_name: str) -> Dict:
        """
        Run scikit-learn model comparison
        """
        logger.info("Starting scikit-learn model comparison")
        
        try:
            from sklearn.model_selection import cross_val_score, KFold
            from sklearn.linear_model import LinearRegression
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.svm import SVR
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            import time
            
            # Initialize models
            models = {
                'Multilinear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'SVM': SVR(kernel='rbf', C=1.0, gamma='scale'),
            }
            
            # Setup cross-validation
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            
            models_results = {}
            
            for model_name, model in models.items():
                logger.info(f"Training {model_name}")
                try:
                    start_time = time.time()
                    
                    # Perform cross-validation
                    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
                    cv_rmse = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
                    cv_mae = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
                    
                    # Calculate metrics
                    mean_r2 = cv_scores.mean()
                    std_r2 = cv_scores.std()
                    mean_rmse = np.sqrt(cv_rmse.mean())
                    std_rmse = np.sqrt(cv_rmse.std())
                    mean_mae = cv_mae.mean()
                    std_mae = cv_mae.std()
                    
                    training_time = time.time() - start_time
                    
                    # Store results
                    models_results[model_name] = {
                        'model': model,
                        'results': {
                            'R2': mean_r2,
                            'R2_std': std_r2,
                            'RMSE': mean_rmse,
                            'RMSE_std': std_rmse,
                            'MAE': mean_mae,
                            'MAE_std': std_mae,
                            'TT (Sec)': training_time,
                            'cv_scores': cv_scores,
                            'cv_rmse': cv_rmse,
                            'cv_mae': cv_mae
                        }
                    }
                    
                    logger.info(f"{model_name} - R²: {mean_r2:.4f} ± {std_r2:.4f}, RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
                    
                except Exception as e:
                    logger.error(f"{model_name} failed: {e}")
                    models_results[model_name] = {
                        'model': None,
                        'results': None,
                        'error': str(e)
                    }
            
            return {
                'models_results': models_results,
                'protein_name': protein_name
            }
            
        except Exception as e:
            logger.error(f"Error in scikit-learn comparison: {e}")
            raise
    
    def save_results(self, results: Dict, protein_name: str):
        """
        Save comparison results to files
        """
        logger.info("Saving results")
        
        # Create summary DataFrame
        summary_data = []
        for model_name, model_data in results['models_results'].items():
            if model_data['results'] is not None:
                results_dict = model_data['results']
                summary_data.append({
                    'Model': model_name,
                    'R2': results_dict['R2'],
                    'R2_std': results_dict['R2_std'],
                    'RMSE': results_dict['RMSE'],
                    'RMSE_std': results_dict['RMSE_std'],
                    'MAE': results_dict['MAE'],
                    'MAE_std': results_dict['MAE_std'],
                    'TT (Sec)': results_dict['TT (Sec)']
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            
            # Save summary results
            summary_file = self.results_dir / f"{protein_name}_sklearn_model_summary.csv"
            summary_df.to_csv(summary_file, index=False)
            logger.info(f"Saved summary results to {summary_file}")
            
            # Save best model
            best_model_name = summary_df.loc[summary_df['R2'].idxmax(), 'Model']
            best_model = results['models_results'][best_model_name]['model']
            
            # Save best model using pickle
            import pickle
            model_file = self.results_dir / f"{protein_name}_best_sklearn_model.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(best_model, f)
            logger.info(f"Saved best model ({best_model_name}) to {model_file}")
        else:
            logger.warning("No successful models to save")
    
    def generate_report(self, results: Dict, protein_name: str):
        """
        Generate a scikit-learn model comparison report
        """
        logger.info("Generating model comparison report")
        
        report_file = self.results_dir / f"{protein_name}_sklearn_report.txt"
        
        with open(report_file, 'w') as f:
            f.write(f"Scikit-learn Model Comparison Report\n")
            f.write(f"====================================\n")
            f.write(f"Protein: {protein_name}\n")
            f.write(f"Date: {pd.Timestamp.now()}\n\n")
            
            f.write("MODEL PERFORMANCE RESULTS\n")
            f.write("=========================\n\n")
            
            successful_models = 0
            for model_name, model_data in results['models_results'].items():
                if model_data['results'] is not None:
                    result = model_data['results']
                    f.write(f"{model_name}:\n")
                    f.write(f"  R² Score: {result['R2']:.4f} ± {result['R2_std']:.4f}\n")
                    f.write(f"  RMSE: {result['RMSE']:.4f} ± {result['RMSE_std']:.4f}\n")
                    f.write(f"  MAE: {result['MAE']:.4f} ± {result['MAE_std']:.4f}\n")
                    f.write(f"  Training Time: {result['TT (Sec)']:.4f}s\n\n")
                    successful_models += 1
                else:
                    f.write(f"{model_name}: FAILED\n")
                    if 'error' in model_data:
                        f.write(f"  Error: {model_data['error']}\n")
                    f.write("\n")
            
            f.write(f"SUMMARY:\n")
            f.write(f"  Total Models Attempted: {len(results['models_results'])}\n")
            f.write(f"  Successful Models: {successful_models}\n")
            f.write(f"  Failed Models: {len(results['models_results']) - successful_models}\n")
        
        logger.info(f"Saved model report to {report_file}")
    
    def run_complete_analysis(self, protein_name: str = "CYP1A2"):
        """
        Run complete scikit-learn analysis for a protein
        """
        logger.info(f"Starting scikit-learn analysis for {protein_name}")
        
        try:
            # Load and prepare data
            X, y, protein_name = self.load_and_prepare_data(protein_name)
            
            # Run model comparison
            comparison_results = self.run_sklearn_comparison(X, y, protein_name)
            
            # Save results
            self.save_results(comparison_results, protein_name)
            
            # Generate report
            self.generate_report(comparison_results, protein_name)
            
            logger.info(f"Complete analysis finished for {protein_name}")
            
            return comparison_results
            
        except Exception as e:
            logger.error(f"Error in complete analysis: {e}")
            raise
    
    def get_all_proteins(self) -> List[str]:
        """
        Get list of all proteins from the dataset
        """
        import sys
        sys.path.append('/home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_modelling')
        from minimal_papyrus_prediction import PapyrusQSARModel
        
        papyrus_model = PapyrusQSARModel()
        papyrus_model.load_data()
        
        # Get all protein names
        proteins_df = papyrus_model.proteins_df
        protein_names = proteins_df['name2_entry'].tolist()
        
        return protein_names
    
    def create_overview_table(self, all_results: Dict[str, Dict]):
        """
        Create an overview table with best models for all proteins
        """
        logger.info("Creating overview table with best models")
        
        overview_data = []
        
        for protein_name, results in all_results.items():
            if results is None:
                continue
                
            # Find best model for this protein
            best_model_name = None
            best_r2 = -float('inf')
            best_rmse = float('inf')
            best_mae = float('inf')
            
            for model_name, model_data in results['models_results'].items():
                if model_data['results'] is not None:
                    r2 = model_data['results']['R2']
                    if r2 > best_r2:
                        best_r2 = r2
                        best_model_name = model_name
                        best_rmse = model_data['results']['RMSE']
                        best_mae = model_data['results']['MAE']
            
            if best_model_name:
                overview_data.append({
                    'Protein': protein_name,
                    'Best Model': best_model_name,
                    'R² Score': best_r2,
                    'RMSE': best_rmse,
                    'MAE': best_mae
                })
        
        if overview_data:
            overview_df = pd.DataFrame(overview_data)
            
            # Sort by R² score (descending)
            overview_df = overview_df.sort_values('R² Score', ascending=False)
            
            # Save overview table
            overview_file = self.results_dir / "all_proteins_best_models_overview.csv"
            overview_df.to_csv(overview_file, index=False)
            logger.info(f"Saved overview table to {overview_file}")
            
            # Generate overview report
            overview_report_file = self.results_dir / "all_proteins_best_models_report.txt"
            with open(overview_report_file, 'w') as f:
                f.write(f"All Proteins - Best Models Overview\n")
                f.write(f"==================================\n")
                f.write(f"Date: {pd.Timestamp.now()}\n\n")
                
                f.write("BEST MODEL FOR EACH PROTEIN:\n")
                f.write("=============================\n\n")
                
                for _, row in overview_df.iterrows():
                    f.write(f"{row['Protein']}:\n")
                    f.write(f"  Best Model: {row['Best Model']}\n")
                    f.write(f"  R² Score: {row['R² Score']:.4f}\n")
                    f.write(f"  RMSE: {row['RMSE']:.4f}\n")
                    f.write(f"  MAE: {row['MAE']:.4f}\n\n")
                
                f.write(f"SUMMARY:\n")
                f.write(f"  Total Proteins Analyzed: {len(overview_df)}\n")
                f.write(f"  Average R² Score: {overview_df['R² Score'].mean():.4f}\n")
                f.write(f"  Average RMSE: {overview_df['RMSE'].mean():.4f}\n")
                f.write(f"  Average MAE: {overview_df['MAE'].mean():.4f}\n")
            
            logger.info(f"Saved overview report to {overview_report_file}")
            
            return overview_df
        else:
            logger.warning("No successful results to create overview table")
            return None

def main():
    """Main function to run scikit-learn model comparison for all proteins"""
    
    model_comparison = SklearnModelComparison()
    
    # Get all proteins
    all_proteins = model_comparison.get_all_proteins()
    logger.info(f"Found {len(all_proteins)} proteins to analyze")
    
    # Store all results
    all_results = {}
    
    for protein in all_proteins:
        try:
            logger.info(f"\n{'='*50}")
            logger.info(f"ANALYZING PROTEIN: {protein}")
            logger.info(f"{'='*50}")
            
            results = model_comparison.run_complete_analysis(protein)
            all_results[protein] = results
            
            # Print summary
            print(f"\nResults for {protein}:")
            successful_models = 0
            best_model_name = None
            best_r2 = -float('inf')
            
            for model_name, model_data in results['models_results'].items():
                if model_data['results'] is not None:
                    result = model_data['results']
                    print(f"  {model_name}: R²={result['R2']:.4f}±{result['R2_std']:.4f}, RMSE={result['RMSE']:.4f}±{result['RMSE_std']:.4f}")
                    successful_models += 1
                    
                    if result['R2'] > best_r2:
                        best_r2 = result['R2']
                        best_model_name = model_name
                else:
                    print(f"  {model_name}: FAILED")
            
            print(f"  Successful models: {successful_models}/{len(results['models_results'])}")
            if best_model_name:
                print(f"  Best model: {best_model_name} (R²={best_r2:.4f})")
            
        except Exception as e:
            logger.error(f"Failed to analyze {protein}: {e}")
            all_results[protein] = None
            continue
    
    # Create overview table
    logger.info("\n" + "="*50)
    logger.info("CREATING OVERVIEW TABLE")
    logger.info("="*50)
    
    overview_df = model_comparison.create_overview_table(all_results)
    
    if overview_df is not None:
        print(f"\nOVERVIEW TABLE - Best Models for All Proteins:")
        print("="*50)
        print(overview_df.to_string(index=False))
        print(f"\nAverage R² Score: {overview_df['R² Score'].mean():.4f}")
        print(f"Average RMSE: {overview_df['RMSE'].mean():.4f}")
        print(f"Average MAE: {overview_df['MAE'].mean():.4f}")

if __name__ == "__main__":
    main() 