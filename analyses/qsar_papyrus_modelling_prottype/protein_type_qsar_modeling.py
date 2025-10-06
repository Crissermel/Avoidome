#!/usr/bin/env python3
"""
Protein Type QSAR Modeling Script

This script implements QSAR modeling for protein type groups by pooling bioactivity data
from proteins of the same functional family. It creates separate models for each animal
(human, mouse, rat) and each protein type group.

Features:
- Groups proteins by functional families (CYP, SLC, receptors, etc.)
- Pools bioactivity data separately for each animal (human, mouse, rat)
- Creates Morgan fingerprints for molecular representation
- Trains Random Forest models with 5-fold cross-validation
- Generates comprehensive performance reports
- Saves trained models for each protein type group

Usage:
    python protein_type_qsar_modeling.py

Input:
    - avoidome_prot_list_extended.csv: Protein list with prot_group column
    - Papyrus database for bioactivity data

Output:
    - Trained models for each protein type group
    - Performance reports and summaries
    - Model comparison analysis

Date: 2025-01-30
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import warnings
import joblib
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
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
        logging.FileHandler('/home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_modelling_prottype/protein_type_qsar_modeling.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProteinTypeQSARModel:
    """
    QSAR modeling for protein type groups using pooled bioactivity data
    """
    
    def __init__(self, protein_list_path: str = "/home/serramelendezcsm/RA/Avoidome/primary_data/avoidome_prot_list_extended.csv"):
        """
        Initialize the protein type QSAR model
        
        Args:
            protein_list_path: Path to the protein list CSV with prot_group column
        """
        self.protein_list_path = protein_list_path
        self.proteins_df = None
        self.papyrus_data = None
        self.papyrus_df = None
        self.protein_groups = {}
        self.grouped_data = {}
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load protein list and initialize Papyrus dataset"""
        logger.info("Loading protein list...")
        
        # Load protein list with protein groups
        self.proteins_df = pd.read_csv(self.protein_list_path)
        logger.info(f"Loaded {len(self.proteins_df)} proteins from {self.protein_list_path}")
        
        # Check if prot_group column exists
        if 'prot_group' not in self.proteins_df.columns:
            raise ValueError("prot_group column not found in protein list. Please add protein grouping first.")
        
        # Initialize Papyrus dataset
        logger.info("Initializing Papyrus dataset...")
        self.papyrus_data = PapyrusDataset(version='latest', plusplus=True)
        
        # Load full dataset into DataFrame for efficient filtering
        logger.info("Loading full Papyrus dataset into DataFrame...")
        self.papyrus_df = self.papyrus_data.to_dataframe()
        logger.info(f"Loaded {len(self.papyrus_df)} total activities from Papyrus")
        
    def analyze_protein_groups(self):
        """Analyze the distribution of proteins across groups"""
        logger.info("Analyzing protein group distribution...")
        
        # Count proteins per group
        group_counts = self.proteins_df['prot_group'].value_counts()
        logger.info("Protein group distribution:")
        for group, count in group_counts.items():
            logger.info(f"  {group}: {count} proteins")
        
        # Create protein groups dictionary
        for group in self.proteins_df['prot_group'].unique():
            group_proteins = self.proteins_df[self.proteins_df['prot_group'] == group]
            self.protein_groups[group] = group_proteins
        
        return group_counts
    
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
    
    def create_morgan_fingerprints(self, smiles_list: List[str], radius: int = 2, nBits: int = 2048) -> Tuple[np.ndarray, List[int]]:
        """
        Create Morgan fingerprints from SMILES strings
        
        Args:
            smiles_list: List of SMILES strings
            radius: Morgan fingerprint radius
            nBits: Number of bits in fingerprint
            
        Returns:
            Tuple of (fingerprints array, valid indices)
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
    
    def pool_group_data_by_animal(self, group_name: str, group_proteins: pd.DataFrame) -> Dict:
        """
        Pool bioactivity data for a protein group, separated by animal
        
        Args:
            group_name: Name of the protein group
            group_proteins: DataFrame containing proteins in the group
            
        Returns:
            Dictionary with pooled data for each animal
        """
        logger.info(f"Pooling data for protein group: {group_name}")
        
        # Initialize animal-specific data containers
        animal_data = {
            'human': [],
            'mouse': [],
            'rat': []
        }
        
        # Process each protein in the group
        for _, protein_row in group_proteins.iterrows():
            protein_name = protein_row['name2_entry']
            
            # Process human data
            if pd.notna(protein_row['human_uniprot_id']):
                human_data = self.get_protein_activities(protein_row['human_uniprot_id'])
                if human_data is not None:
                    human_data['source_protein'] = protein_name
                    human_data['organism'] = 'human'
                    animal_data['human'].append(human_data)
            
            # Process mouse data
            if pd.notna(protein_row['mouse_uniprot_id']):
                mouse_data = self.get_protein_activities(protein_row['mouse_uniprot_id'])
                if mouse_data is not None:
                    mouse_data['source_protein'] = protein_name
                    mouse_data['organism'] = 'mouse'
                    animal_data['mouse'].append(mouse_data)
            
            # Process rat data
            if pd.notna(protein_row['rat_uniprot_id']):
                rat_data = self.get_protein_activities(protein_row['rat_uniprot_id'])
                if rat_data is not None:
                    rat_data['source_protein'] = protein_name
                    rat_data['organism'] = 'rat'
                    animal_data['rat'].append(rat_data)
        
        # Combine data for each animal
        pooled_data = {}
        for animal, data_list in animal_data.items():
            if data_list:
                pooled_data[animal] = pd.concat(data_list, ignore_index=True)
                logger.info(f"  {animal}: {len(pooled_data[animal])} activities from {len(data_list)} proteins")
            else:
                pooled_data[animal] = pd.DataFrame()
                logger.info(f"  {animal}: No data available")
        
        return pooled_data
    
    def prepare_modeling_data(self, animal_data: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
        """
        Prepare data for QSAR modeling
        
        Args:
            animal_data: DataFrame with bioactivity data for a specific animal
            
        Returns:
            Tuple of (features, targets, feature_names)
        """
        if animal_data.empty:
            return None, None, []
        
        # Clean and prepare data
        clean_data = animal_data.dropna(subset=['SMILES', 'pchembl_value'])
        clean_data = clean_data.drop_duplicates(subset=['SMILES', 'pchembl_value'])
        
        if len(clean_data) < 10:
            logger.warning(f"Insufficient data for modeling: {len(clean_data)} samples")
            return None, None, []
        
        # Handle multi-value pchembl entries (e.g., '6.92;7.36')
        def process_pchembl_value(value):
            """Process pchembl value, handling multi-value entries"""
            if pd.isna(value):
                return None
            
            # Convert to string and split by semicolon if multiple values
            str_value = str(value).strip()
            if ';' in str_value:
                # Take the first value if multiple values exist
                first_value = str_value.split(';')[0].strip()
                try:
                    return float(first_value)
                except ValueError:
                    return None
            else:
                try:
                    return float(str_value)
                except ValueError:
                    return None
        
        # Count multi-value entries for logging
        multi_value_count = sum(clean_data['pchembl_value'].astype(str).str.contains(';', na=False))
        if multi_value_count > 0:
            logger.info(f"Found {multi_value_count} multi-value pchembl entries, processing...")
        
        # Apply pchembl processing
        clean_data['pchembl_value_processed'] = clean_data['pchembl_value'].apply(process_pchembl_value)
        
        # Remove rows with invalid pchembl values
        clean_data = clean_data.dropna(subset=['pchembl_value_processed'])
        
        logger.info(f"Data after pchembl processing: {len(clean_data)} samples (removed {len(animal_data) - len(clean_data)} invalid entries)")
        
        if len(clean_data) < 10:
            logger.warning(f"Insufficient valid data after pchembl processing: {len(clean_data)} samples")
            return None, None, []
        
        # Create Morgan fingerprints
        smiles_list = clean_data['SMILES'].tolist()
        fingerprints, valid_indices = self.create_morgan_fingerprints(smiles_list)
        
        if len(fingerprints) < 10:
            logger.warning(f"Insufficient valid fingerprints: {len(fingerprints)} samples")
            return None, None, []
        
        # Get target values (use processed pchembl values)
        targets = clean_data['pchembl_value_processed'].iloc[valid_indices].values
        
        # Shuffle the data to ensure unbiased CV splitting
        # This prevents data clustering by protein source
        shuffled_indices = np.random.permutation(len(fingerprints))
        fingerprints = fingerprints[shuffled_indices]
        targets = targets[shuffled_indices]
        
        logger.info(f"Shuffled data to ensure unbiased cross-validation splitting")
        
        # Feature names for Morgan fingerprints
        feature_names = [f'FP_{i}' for i in range(fingerprints.shape[1])]
        
        return fingerprints, targets, feature_names
    
    def create_predicted_vs_actual_plot(self, X: np.ndarray, y: np.ndarray, model, group_name: str, animal: str):
        """
        Create and save a predicted vs actual values scatter plot
        
        Args:
            X: Feature matrix
            y: Actual target values
            model: Trained model
            group_name: Name of the protein group
            animal: Animal type
        """
        try:
            # Get predictions
            y_pred = model.predict(X)
            
            # Create the plot
            plt.figure(figsize=(10, 8))
            
            # Create scatter plot
            plt.scatter(y, y_pred, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
            
            # Add perfect prediction line
            min_val = min(y.min(), y_pred.min())
            max_val = max(y.max(), y_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
            
            # Calculate R² for the plot
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            
            # Add statistics to plot
            plt.text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {rmse:.3f}', 
                    transform=plt.gca().transAxes, fontsize=12, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Labels and title
            plt.xlabel('Actual pChEMBL Values', fontsize=14)
            plt.ylabel('Predicted pChEMBL Values', fontsize=14)
            plt.title(f'{group_name} - {animal.upper()}\nPredicted vs Actual Values (n={len(y)})', fontsize=16)
            
            # Grid and legend
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Equal aspect ratio
            plt.axis('equal')
            
            # Ensure the group directory exists
            group_dir = Path(f"/home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_modelling_prottype/{group_name}")
            group_dir.mkdir(exist_ok=True)
            
            # Save the plot
            plot_filename = f"{group_name}_{animal}_predicted_vs_actual.png"
            plot_path = group_dir / plot_filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved predicted vs actual plot: {plot_path}")
            
        except Exception as e:
            logger.error(f"Error creating plot for {group_name} - {animal}: {e}")
    
    def train_rf_model(self, X: np.ndarray, y: np.ndarray, group_name: str, animal: str) -> Dict:
        """
        Train a Random Forest model with 5-fold CV
        
        Args:
            X: Feature matrix (Morgan fingerprints)
            y: Target values (bioactivity)
            group_name: Name of the protein group
            animal: Animal type (human, mouse, rat)
            
        Returns:
            Dictionary with CV results and trained model
        """
        logger.info(f"Training Random Forest model for {group_name} - {animal}")
        
        if len(X) < 10:
            logger.warning(f"Insufficient data for {group_name} - {animal}: {len(X)} samples")
            return {
                'group_name': group_name,
                'animal': animal,
                'n_samples': len(X),
                'status': 'insufficient_data',
                'cv_results': [],
                'model': None
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
            
            logger.info(f"  Fold {fold + 1} - RMSE: {rmse:.3f}, R²: {r2:.3f}")
        
        # Train final model on all data
        final_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        final_model.fit(X, y)
        
        # Create predicted vs actual plot for this model
        self.create_predicted_vs_actual_plot(X, y, final_model, group_name, animal)
        
        return {
            'group_name': group_name,
            'animal': animal,
            'n_samples': len(X),
            'status': 'success',
            'cv_results': cv_results,
            'model': final_model
        }
    
    def process_protein_group(self, group_name: str, group_proteins: pd.DataFrame):
        """Process a single protein group and train models for each animal"""
        logger.info(f"\nProcessing protein group: {group_name}")
        
        # Pool data by animal
        pooled_data = self.pool_group_data_by_animal(group_name, group_proteins)
        self.grouped_data[group_name] = pooled_data
        
        # Train models for each animal
        group_results = {}
        
        for animal, animal_data in pooled_data.items():
            if animal_data.empty:
                logger.info(f"No data available for {group_name} - {animal}")
                continue
            
            # Prepare modeling data
            X, y, feature_names = self.prepare_modeling_data(animal_data)
            
            if X is not None and y is not None:
                # Train model
                model_result = self.train_rf_model(X, y, group_name, animal)
                group_results[animal] = model_result
                
                # Save model
                self.save_model(group_name, animal, model_result, feature_names)
            else:
                logger.warning(f"Could not prepare data for {group_name} - {animal}")
        
        self.results[group_name] = group_results
    
    def save_model(self, group_name: str, animal: str, model_result: Dict, feature_names: List[str]):
        """Save trained model and metadata"""
        if model_result['status'] != 'success':
            return
        
        # Create output directory
        output_dir = Path(f"/home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_modelling_prottype/{group_name}")
        output_dir.mkdir(exist_ok=True)
        
        # Save model
        model_file = output_dir / f"{group_name}_{animal}_model.pkl"
        joblib.dump(model_result['model'], model_file)
        logger.info(f"Saved model to {model_file}")
        
        # Save metadata
        metadata = {
            'group_name': group_name,
            'animal': animal,
            'n_samples': model_result['n_samples'],
            'feature_names': feature_names,
            'cv_results': model_result['cv_results'],
            'training_date': datetime.now().isoformat()
        }
        
        metadata_file = output_dir / f"{group_name}_{animal}_metadata.json"
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save CV results
        cv_df = pd.DataFrame(model_result['cv_results'])
        cv_file = output_dir / f"{group_name}_{animal}_cv_results.csv"
        cv_df.to_csv(cv_file, index=False)
    
    def generate_group_summary(self, group_name: str, group_results: Dict):
        """Generate summary statistics for a protein group"""
        output_dir = Path(f"/home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_modelling_prottype/{group_name}")
        output_dir.mkdir(exist_ok=True)
        
        summary_stats = {
            'group_name': group_name,
            'total_proteins': len(self.protein_groups[group_name]),
            'animals_with_models': [],
            'total_samples': 0,
            'animal_breakdown': {}
        }
        
        for animal, result in group_results.items():
            if result['status'] == 'success':
                summary_stats['animals_with_models'].append(animal)
                summary_stats['animal_breakdown'][animal] = {
                    'samples': result['n_samples'],
                    'cv_r2_mean': np.mean([r['r2'] for r in result['cv_results']]),
                    'cv_r2_std': np.std([r['r2'] for r in result['cv_results']]),
                    'cv_rmse_mean': np.mean([r['rmse'] for r in result['cv_results']]),
                    'cv_rmse_std': np.std([r['rmse'] for r in result['cv_results']])
                }
                summary_stats['total_samples'] += result['n_samples']
        
        # Save summary
        summary_file = output_dir / f"{group_name}_modeling_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"PROTEIN GROUP QSAR MODELING SUMMARY: {group_name}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total proteins in group: {summary_stats['total_proteins']}\n")
            f.write(f"Animals with successful models: {', '.join(summary_stats['animals_with_models'])}\n")
            f.write(f"Total bioactivity samples: {summary_stats['total_samples']}\n\n")
            
            f.write("MODEL PERFORMANCE BY ANIMAL:\n")
            f.write("-" * 30 + "\n")
            for animal, stats in summary_stats['animal_breakdown'].items():
                f.write(f"{animal.upper()}:\n")
                f.write(f"  Samples: {stats['samples']}\n")
                f.write(f"  CV R²: {stats['cv_r2_mean']:.3f} ± {stats['cv_r2_std']:.3f}\n")
                f.write(f"  CV RMSE: {stats['cv_rmse_mean']:.3f} ± {stats['cv_rmse_std']:.3f}\n\n")
        
        logger.info(f"Generated summary for {group_name}")
    
    def generate_overall_report(self):
        """Generate overall modeling report"""
        logger.info("Generating overall modeling report...")
        
        overall_summary = []
        for group_name, group_results in self.results.items():
            group_summary = {
                'group_name': group_name,
                'total_proteins': len(self.protein_groups[group_name]),
                'animals_with_models': len([r for r in group_results.values() if r['status'] == 'success']),
                'total_samples': sum([r['n_samples'] for r in group_results.values() if r['status'] == 'success']),
                'avg_cv_r2': np.mean([
                    np.mean([fold['r2'] for fold in r['cv_results']]) 
                    for r in group_results.values() if r['status'] == 'success'
                ]) if any(r['status'] == 'success' for r in group_results.values()) else 0
            }
            overall_summary.append(group_summary)
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(overall_summary)
        
        # Save overall summary
        output_dir = Path("/home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_modelling_prottype")
        summary_file = output_dir / "overall_qsar_modeling_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        
        # Save detailed report
        report_file = output_dir / "detailed_qsar_modeling_report.txt"
        with open(report_file, 'w') as f:
            f.write("PROTEIN TYPE QSAR MODELING REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total protein groups: {len(summary_df)}\n")
            f.write(f"Total proteins: {summary_df['total_proteins'].sum()}\n")
            f.write(f"Total bioactivity samples: {summary_df['total_samples'].sum()}\n\n")
            
            f.write("PERFORMANCE SUMMARY:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Groups with successful models: {(summary_df['animals_with_models'] > 0).sum()}\n")
            f.write(f"Average CV R² across all models: {summary_df['avg_cv_r2'].mean():.3f}\n\n")
            
            f.write("TOP PERFORMING GROUPS:\n")
            f.write("-" * 20 + "\n")
            top_groups = summary_df.nlargest(5, 'avg_cv_r2')
            for _, group in top_groups.iterrows():
                f.write(f"{group['group_name']}: R² = {group['avg_cv_r2']:.3f}, Samples = {group['total_samples']}\n")
            
            f.write("\nDETAILED GROUP RESULTS:\n")
            f.write("-" * 20 + "\n")
            for _, row in summary_df.iterrows():
                f.write(f"\nGROUP: {row['group_name']}\n")
                f.write(f"  Proteins: {row['total_proteins']}\n")
                f.write(f"  Animals with models: {row['animals_with_models']}\n")
                f.write(f"  Total samples: {row['total_samples']}\n")
                f.write(f"  Average CV R²: {row['avg_cv_r2']:.3f}\n")
        
        logger.info(f"Overall report saved to {report_file}")
        return summary_df
    
    def run(self):
        """Run the complete protein type QSAR modeling pipeline"""
        logger.info("Starting protein type QSAR modeling pipeline...")
        
        try:
            # Load data
            self.load_data()
            
            # Analyze protein groups
            group_counts = self.analyze_protein_groups()
            
            # Process each protein group
            for group_name, group_proteins in self.protein_groups.items():
                self.process_protein_group(group_name, group_proteins)
                
                # Generate group summary
                if group_name in self.results:
                    self.generate_group_summary(group_name, self.results[group_name])
            
            # Generate overall report
            overall_summary = self.generate_overall_report()
            
            logger.info("Protein type QSAR modeling pipeline completed successfully!")
            logger.info(f"Processed {len(self.protein_groups)} protein groups")
            
            return overall_summary
            
        except Exception as e:
            logger.error(f"Error in protein type QSAR modeling pipeline: {e}")
            raise

def main():
    """Main execution function"""
    modeler = ProteinTypeQSARModel()
    summary = modeler.run()
    
    print("\n" + "="*60)
    print("PROTEIN TYPE QSAR MODELING COMPLETED")
    print("="*60)
    print(f"Total protein groups processed: {len(summary)}")
    print(f"Total proteins: {summary['total_proteins'].sum()}")
    print(f"Total bioactivity samples: {summary['total_samples'].sum()}")
    
    print("\nTop performing protein groups:")
    top_groups = summary.nlargest(5, 'avg_cv_r2')
    for _, group in top_groups.iterrows():
        print(f"  {group['group_name']}: R² = {group['avg_cv_r2']:.3f}, Samples = {group['total_samples']}")

if __name__ == "__main__":
    main() 