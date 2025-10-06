#!/usr/bin/env python3
"""
Comprehensive comparison of model performance differences between
standardized QSAR models and 04_qsar_model_creation.py for HTR2B
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

# Add paths
sys.path.append('/home/serramelendezcsm/RA/Avoidome/analyses')
from physicochemical_descriptors import calculate_physicochemical_descriptors

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_htr2b_data():
    """Load HTR2B data from Papyrus"""
    try:
        from papyrus_scripts import PapyrusDataset
        
        papyrus_data = PapyrusDataset(version='latest', plusplus=True)
        papyrus_df = papyrus_data.to_dataframe()
        
        htr2b_data = papyrus_df[papyrus_df['accession'] == 'P41595'].copy()
        valid_data = htr2b_data.dropna(subset=['SMILES', 'pchembl_value_Mean'])
        valid_data = valid_data[valid_data['SMILES'] != '']
        
        logger.info(f"Loaded {len(valid_data)} HTR2B records")
        return valid_data
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

def calculate_standardized_features(smiles_list):
    """Calculate features using standardized QSAR models approach"""
    features_list = []
    
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                # Morgan fingerprints (2048 bits)
                morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                morgan_array = np.array(morgan_fp)
                
                # Physicochemical descriptors (14 features)
                physico_desc = calculate_physicochemical_descriptors(smiles, include_sasa=False, verbose=False)
                physico_features = [
                    physico_desc['ALogP'],
                    physico_desc['Molecular_Weight'],
                    physico_desc['Num_H_Donors'],
                    physico_desc['Num_H_Acceptors'],
                    physico_desc['Num_Rotatable_Bonds'],
                    physico_desc['Num_Atoms'],
                    physico_desc['Num_Rings'],
                    physico_desc['Num_Aromatic_Rings'],
                    physico_desc['LogS'],
                    physico_desc['Molecular_Surface_Area'],
                    physico_desc['Molecular_Polar_Surface_Area'],
                    physico_desc['Num_Heavy_Atoms'],
                    physico_desc['Formal_Charge'],
                    physico_desc['Num_Saturated_Rings']
                ]
                
                # Combine features
                combined_features = np.concatenate([morgan_array, physico_features])
                features_list.append(combined_features)
            else:
                # Add zero vector for invalid SMILES
                features_list.append(np.zeros(2048 + 14))
        except Exception as e:
            logger.warning(f"Error processing {smiles}: {e}")
            features_list.append(np.zeros(2048 + 14))
    
    return np.array(features_list)

def calculate_04_features(smiles_list):
    """Calculate features using 04_qsar_model_creation.py approach"""
    features_list = []
    
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                # Morgan fingerprints (1024 bits)
                morgan_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                morgan_array = np.array(morgan_fp, dtype=np.float32)
                
                # Physicochemical descriptors (12 features)
                physico_desc = calculate_physicochemical_descriptors(smiles, include_sasa=False, verbose=False)
                physico_features = [
                    physico_desc['Molecular_Weight'],
                    physico_desc['ALogP'],
                    physico_desc['Molecular_Polar_Surface_Area'],
                    physico_desc['Num_H_Donors'],
                    physico_desc['Num_H_Acceptors'],
                    physico_desc['Num_Rotatable_Bonds'],
                    physico_desc['Num_Aromatic_Rings'],
                    physico_desc['Num_Heavy_Atoms'],
                    physico_desc['Num_Rings'],
                    physico_desc['Formal_Charge'],
                    physico_desc['LogS'],
                    physico_desc['Molecular_Surface_Area']
                ]
                
                # Combine features
                combined_features = np.concatenate([morgan_array, physico_features])
                features_list.append(combined_features)
            else:
                # Add zero vector for invalid SMILES
                features_list.append(np.zeros(1024 + 12))
        except Exception as e:
            logger.warning(f"Error processing {smiles}: {e}")
            features_list.append(np.zeros(1024 + 12))
    
    return np.array(features_list)

def train_and_evaluate_model(features, targets, model_name):
    """Train and evaluate a model"""
    logger.info(f"Training {model_name} model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    logger.info(f"{model_name} Results:")
    logger.info(f"  R²: {r2:.4f}")
    logger.info(f"  RMSE: {rmse:.4f}")
    logger.info(f"  MAE: {mae:.4f}")
    logger.info(f"  Training samples: {len(X_train)}")
    logger.info(f"  Test samples: {len(X_test)}")
    logger.info(f"  Features: {features.shape[1]}")
    
    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_features': features.shape[1]
    }

def main():
    """Main comparison function"""
    logger.info("Starting comprehensive HTR2B model comparison...")
    
    # Load data
    data = load_htr2b_data()
    if data is None:
        logger.error("Failed to load data")
        return
    
    smiles_list = data['SMILES'].tolist()
    targets = data['pchembl_value_Mean'].values
    
    logger.info(f"Data loaded: {len(smiles_list)} compounds, target range: {targets.min():.3f} - {targets.max():.3f}")
    
    # Calculate features using both approaches
    logger.info("Calculating standardized QSAR features...")
    features_std = calculate_standardized_features(smiles_list)
    
    logger.info("Calculating 04_qsar_model_creation.py features...")
    features_04 = calculate_04_features(smiles_list)
    
    logger.info(f"Standardized features shape: {features_std.shape}")
    logger.info(f"04_qsar_model_creation.py features shape: {features_04.shape}")
    
    # Train and evaluate both models
    logger.info("\n" + "="*60)
    logger.info("MODEL PERFORMANCE COMPARISON")
    logger.info("="*60)
    
    results_std = train_and_evaluate_model(features_std, targets, "Standardized QSAR")
    
    logger.info("\n" + "-"*40)
    
    results_04 = train_and_evaluate_model(features_04, targets, "04_qsar_model_creation.py")
    
    # Compare results
    logger.info("\n" + "="*60)
    logger.info("PERFORMANCE COMPARISON SUMMARY")
    logger.info("="*60)
    logger.info(f"R² difference: {results_std['r2'] - results_04['r2']:.4f}")
    logger.info(f"RMSE difference: {results_std['rmse'] - results_04['rmse']:.4f}")
    logger.info(f"MAE difference: {results_std['mae'] - results_04['mae']:.4f}")
    
    if results_std['r2'] > results_04['r2']:
        logger.info("Standardized QSAR model performs BETTER")
    else:
        logger.info("04_qsar_model_creation.py model performs BETTER")
    
    # Feature analysis
    logger.info("\n" + "="*60)
    logger.info("FEATURE ANALYSIS")
    logger.info("="*60)
    logger.info(f"Standardized features: {results_std['n_features']} (2048 Morgan + 14 physico)")
    logger.info(f"04_qsar_model_creation.py features: {results_04['n_features']} (1024 Morgan + 12 physico)")
    logger.info(f"Feature difference: {results_std['n_features'] - results_04['n_features']} features")
    
    # Check for NaN values
    logger.info(f"Standardized features NaN count: {np.isnan(features_std).sum()}")
    logger.info(f"04_qsar_model_creation.py features NaN count: {np.isnan(features_04).sum()}")

if __name__ == "__main__":
    main()
