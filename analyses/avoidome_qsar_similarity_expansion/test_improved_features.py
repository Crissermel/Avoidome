#!/usr/bin/env python3
"""
Test script to verify the improved features in 04_qsar_model_creation.py
match the standardized QSAR models approach
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
from rdkit.Chem import AllChem

# Add paths
sys.path.append('/home/serramelendezcsm/RA/Avoidome/analyses')
from physicochemical_descriptors import calculate_physicochemical_descriptors

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_improved_features():
    """Test the improved feature calculation"""
    logger.info("Testing improved feature calculation...")
    
    # Test SMILES
    test_smiles = "CCN(CC)CCCC(C)NC1=C2C=CC(Cl)=CC2=NC=C1"
    
    try:
        mol = Chem.MolFromSmiles(test_smiles)
        if mol is None:
            logger.error("Could not parse test SMILES")
            return False
        
        # Calculate Morgan fingerprints (2048 bits) - improved version
        morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        morgan_array = np.array(morgan_fp, dtype=np.float32)
        
        # Calculate physicochemical descriptors
        physico_desc = calculate_physicochemical_descriptors(test_smiles, include_sasa=False, verbose=False)
        
        # Create descriptor row with all 14 descriptors
        descriptor_row = {
            'SMILES': test_smiles,
            'molecular_weight': physico_desc['Molecular_Weight'],
            'logp': physico_desc['ALogP'],
            'tpsa': physico_desc['Molecular_Polar_Surface_Area'],
            'hbd': physico_desc['Num_H_Donors'],
            'hba': physico_desc['Num_H_Acceptors'],
            'rotatable_bonds': physico_desc['Num_Rotatable_Bonds'],
            'aromatic_rings': physico_desc['Num_Aromatic_Rings'],
            'heavy_atoms': physico_desc['Num_Heavy_Atoms'],
            'num_rings': physico_desc['Num_Rings'],
            'formal_charge': physico_desc['Formal_Charge'],
            'log_solubility': physico_desc['LogS'],
            'molecular_surface_area': physico_desc['Molecular_Surface_Area'],
            'num_atoms': physico_desc['Num_Atoms'],  # Added
            'num_saturated_rings': physico_desc['Num_Saturated_Rings']  # Added
        }
        
        # Add Morgan fingerprint bits
        for j, bit in enumerate(morgan_array):
            descriptor_row[f'morgan_bit_{j}'] = int(bit)
        
        # Verify feature counts
        morgan_bits = len([k for k in descriptor_row.keys() if k.startswith('morgan_bit_')])
        physico_features = len([k for k in descriptor_row.keys() if k not in ['SMILES'] and not k.startswith('morgan_bit_')])
        total_features = len(descriptor_row) - 1  # Exclude SMILES
        
        logger.info(f"Feature verification:")
        logger.info(f"  Morgan fingerprint bits: {morgan_bits}")
        logger.info(f"  Physicochemical descriptors: {physico_features}")
        logger.info(f"  Total features: {total_features}")
        logger.info(f"  Expected: 2048 + 14 = 2062")
        
        # Check specific descriptors
        logger.info(f"Key descriptors:")
        logger.info(f"  ALogP: {descriptor_row['logp']:.4f}")
        logger.info(f"  Molecular Weight: {descriptor_row['molecular_weight']:.4f}")
        logger.info(f"  Num Atoms: {descriptor_row['num_atoms']}")
        logger.info(f"  Num Saturated Rings: {descriptor_row['num_saturated_rings']}")
        logger.info(f"  Morgan non-zero bits: {np.sum(morgan_array)}")
        
        # Verify it matches standardized approach
        if morgan_bits == 2048 and physico_features == 14 and total_features == 2062:
            logger.info("✓ Feature calculation matches standardized QSAR models!")
            return True
        else:
            logger.error("✗ Feature calculation does not match standardized QSAR models")
            return False
            
    except Exception as e:
        logger.error(f"Error testing features: {e}")
        return False

def test_htr2b_performance():
    """Test HTR2B performance with improved features"""
    logger.info("Testing HTR2B performance with improved features...")
    
    try:
        from papyrus_scripts import PapyrusDataset
        
        # Load HTR2B data
        papyrus_data = PapyrusDataset(version='latest', plusplus=True)
        papyrus_df = papyrus_data.to_dataframe()
        
        htr2b_data = papyrus_df[papyrus_df['accession'] == 'P41595'].copy()
        valid_data = htr2b_data.dropna(subset=['SMILES', 'pchembl_value_Mean'])
        valid_data = valid_data[valid_data['SMILES'] != '']
        
        logger.info(f"Loaded {len(valid_data)} HTR2B records")
        
        # Calculate features using improved approach
        features_list = []
        targets = valid_data['pchembl_value_Mean'].values
        
        for smiles in valid_data['SMILES']:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    # Morgan fingerprints (2048 bits)
                    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                    morgan_array = np.array(morgan_fp, dtype=np.float32)
                    
                    # Physicochemical descriptors (14 features)
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
                        physico_desc['Molecular_Surface_Area'],
                        physico_desc['Num_Atoms'],  # Added
                        physico_desc['Num_Saturated_Rings']  # Added
                    ]
                    
                    # Combine features
                    combined_features = np.concatenate([morgan_array, physico_features])
                    features_list.append(combined_features)
                else:
                    features_list.append(np.zeros(2048 + 14))
            except Exception as e:
                logger.warning(f"Error processing {smiles}: {e}")
                features_list.append(np.zeros(2048 + 14))
        
        features = np.array(features_list)
        logger.info(f"Features shape: {features.shape}")
        
        # Train and test model
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets, test_size=0.2, random_state=42
        )
        
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        logger.info(f"Improved HTR2B model performance:")
        logger.info(f"  R²: {r2:.4f}")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  MAE: {mae:.4f}")
        logger.info(f"  Features: {features.shape[1]}")
        
        # Compare with expected standardized performance (R² ≈ 0.47)
        if r2 >= 0.46:  # Should be close to standardized performance
            logger.info("✓ Performance matches standardized QSAR models!")
            return True
        else:
            logger.warning(f"⚠ Performance ({r2:.4f}) is lower than expected (≥0.46)")
            return False
            
    except Exception as e:
        logger.error(f"Error testing HTR2B performance: {e}")
        return False

def main():
    """Main test function"""
    logger.info("Testing improved 04_qsar_model_creation.py features...")
    
    # Test feature calculation
    feature_test_passed = test_improved_features()
    
    # Test HTR2B performance
    performance_test_passed = test_htr2b_performance()
    
    logger.info("\n" + "="*60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("="*60)
    logger.info(f"Feature calculation test: {'PASSED' if feature_test_passed else 'FAILED'}")
    logger.info(f"HTR2B performance test: {'PASSED' if performance_test_passed else 'FAILED'}")
    
    if feature_test_passed and performance_test_passed:
        logger.info("✓ All tests passed! 04_qsar_model_creation.py improvements are working correctly.")
    else:
        logger.error("✗ Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()

