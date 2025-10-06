"""
QSAR Prediction Script for Avoidome Bioactivity Data
====================================================

This script provides a command-line interface for making predictions using trained QSAR models.
It can handle both single molecule predictions and batch predictions from CSV files.

Features:
- Single molecule prediction from SMILES string
- Batch prediction from CSV files
- Model performance inspection
- Available targets and models listing
- Error handling and validation

Usage Examples:
    # List available targets
    python predict.py --list-targets
    
    # Predict single molecule
    python predict.py --smiles "CCc1ccc(C(=O)N2Cc3ccccc3C[C@H]2CN2CCOCC2)c(-c2cc(C(=O)N(c3ccc(O)cc3)c3ccc4c(ccn4C)c3)c3n2CCCC3)c1" --target P05177
    
    # Batch prediction from CSV
    python predict.py --file compounds.csv --target P05177
    
    # Show model performance
    python predict.py --performance P05177

Command Line Arguments:
    --smiles: SMILES string for single prediction
    --target: Target ID (e.g., P05177)
    --model: Model name to use (default: RandomForest)
    --file: CSV file with SMILES column for batch prediction
    --list-targets: List all available targets
    --list-models: List available models for a target
    --performance: Show performance metrics for a target

Input:
    - Trained models in qsar_avoidome directory
    - SMILES strings (single or in CSV file)

Output:
    - Predicted pChEMBL values
    - CSV file with predictions (for batch mode)
    - Error messages for invalid inputs

Author: QSAR Modeling System
Date: 2024
"""

import joblib
import json
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import os
import argparse

class QSARPredictor:
    """
    QSAR Predictor Class for Making Predictions with Trained Models
    
    This class provides functionality to load trained QSAR models and make predictions
    on new molecules. It handles descriptor calculation, model loading, and prediction
    for both single molecules and batches.
    
    Features:
    - Load trained models from disk
    - Calculate molecular descriptors from SMILES
    - Make single and batch predictions
    - Handle errors and validation
    - Provide model performance information
    
    Attributes:
        models_dir (str): Directory containing trained models
        available_targets (list): List of available target IDs
    """
    
    def __init__(self, models_dir="/home/serramelendezcsm/RA/Avoidome/analyses/qsar_avoidome"):
        """
        Initialize the QSAR Predictor
        
        Args:
            models_dir (str): Directory containing trained models
        """
        self.models_dir = models_dir
        self.available_targets = self._get_available_targets()
        
    def _get_available_targets(self):
        """Get list of available targets with trained models"""
        targets = []
        if os.path.exists(self.models_dir):
            for item in os.listdir(self.models_dir):
                item_path = os.path.join(self.models_dir, item)
                if os.path.isdir(item_path) and any(item.endswith('_model.pkl') for item in os.listdir(item_path)):
                    targets.append(item)
        return targets
    
    def calculate_descriptors(self, smiles):
        """Calculate molecular descriptors from SMILES"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Calculate basic descriptors
        desc_dict = {
            'MolWt': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
            'TPSA': Descriptors.TPSA(mol),
            'NumAtoms': mol.GetNumAtoms(),
            'NumHeavyAtoms': mol.GetNumHeavyAtoms(),
            'NumRings': Descriptors.RingCount(mol),
            'NumAromaticRings': Descriptors.NumAromaticRings(mol)
        }
        
        # Add Morgan fingerprints
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        fp_array = np.array(fp)
        for j, bit in enumerate(fp_array):
            desc_dict[f'FP_{j}'] = bit
        
        return desc_dict
    
    def load_model(self, target_id, model_name='RandomForest'):
        """Load a trained QSAR model"""
        model_dir = os.path.join(self.models_dir, target_id)
        
        # Check if model exists
        model_path = os.path.join(model_dir, f"{model_name}_model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model {model_name} not found for target {target_id}")
        
        # Load model
        model = joblib.load(model_path)
        
        # Load scaler if exists
        scaler = None
        scaler_path = os.path.join(model_dir, f"{model_name}_scaler.pkl")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        
        # Load metadata
        metadata_path = os.path.join(model_dir, f"{model_name}_metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return model, scaler, metadata
    
    def predict_single(self, smiles, target_id, model_name='RandomForest'):
        """Predict pChEMBL value for a single SMILES and target"""
        # Calculate descriptors
        desc_dict = self.calculate_descriptors(smiles)
        if desc_dict is None:
            return None, "Invalid SMILES"
        
        # Load model
        try:
            model, scaler, metadata = self.load_model(target_id, model_name)
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
    
    def predict_batch(self, smiles_list, target_id, model_name='RandomForest'):
        """Predict pChEMBL values for a list of SMILES"""
        results = []
        
        for i, smiles in enumerate(smiles_list):
            prediction, error = self.predict_single(smiles, target_id, model_name)
            results.append({
                'index': i,
                'smiles': smiles,
                'prediction': prediction,
                'error': error
            })
        
        return results
    
    def get_model_performance(self, target_id, model_name='RandomForest'):
        """Get performance metrics for a model"""
        try:
            model, scaler, metadata = self.load_model(target_id, model_name)
            return metadata['performance']
        except Exception as e:
            return None
    
    def list_available_targets(self):
        """List all available targets with trained models"""
        return self.available_targets
    
    def list_available_models(self, target_id):
        """List available models for a specific target"""
        model_dir = os.path.join(self.models_dir, target_id)
        if not os.path.exists(model_dir):
            return []
        
        models = []
        for item in os.listdir(model_dir):
            if item.endswith('_model.pkl'):
                model_name = item.replace('_model.pkl', '')
                models.append(model_name)
        
        return models

def main():
    """
    Main function for QSAR prediction command-line interface
    
    This function handles command-line argument parsing and executes the appropriate
    prediction functionality based on the provided arguments.
    
    Command Line Arguments:
        --smiles: SMILES string for single molecule prediction
        --target: Target ID (e.g., P05177) for prediction
        --model: Model name to use (default: RandomForest)
        --file: CSV file with SMILES column for batch prediction
        --list-targets: List all available targets with trained models
        --list-models: List available models for a specific target
        --performance: Show performance metrics for a specific target
    
    Examples:
        python predict.py --list-targets
        python predict.py --smiles "CCc1ccc(C(=O)N2Cc3ccccc3C[C@H]2CN2CCOCC2)c(-c2cc(C(=O)N(c3ccc(O)cc3)c3ccc4c(ccn4C)c3)c3n2CCCC3)c1" --target P05177
        python predict.py --file compounds.csv --target P05177
        python predict.py --performance P05177
    """
    parser = argparse.ArgumentParser(description='QSAR Prediction Tool')
    parser.add_argument('--smiles', type=str, help='SMILES string to predict')
    parser.add_argument('--target', type=str, help='Target ID (e.g., P05177)')
    parser.add_argument('--model', type=str, default='RandomForest', help='Model name to use')
    parser.add_argument('--file', type=str, help='CSV file with SMILES column')
    parser.add_argument('--list-targets', action='store_true', help='List available targets')
    parser.add_argument('--list-models', type=str, help='List available models for target')
    parser.add_argument('--performance', type=str, help='Show performance for target')
    
    args = parser.parse_args()
    
    predictor = QSARPredictor()
    
    if args.list_targets:
        targets = predictor.list_available_targets()
        print("Available targets:")
        for target in targets:
            print(f"  {target}")
        return
    
    if args.list_models:
        models = predictor.list_available_models(args.list_models)
        print(f"Available models for {args.list_models}:")
        for model in models:
            print(f"  {model}")
        return
    
    if args.performance:
        perf = predictor.get_model_performance(args.performance, args.model)
        if perf:
            print(f"Performance for {args.performance} ({args.model}):")
            for metric, value in perf.items():
                print(f"  {metric}: {value}")
        else:
            print(f"No performance data found for {args.performance}")
        return
    
    if args.smiles and args.target:
        prediction, error = predictor.predict_single(args.smiles, args.target, args.model)
        if error:
            print(f"Error: {error}")
        else:
            print(f"Predicted pChEMBL: {prediction:.3f}")
        return
    
    if args.file and args.target:
        # Read CSV file
        try:
            df = pd.read_csv(args.file)
            if 'smiles' not in df.columns and 'SMILES' not in df.columns:
                print("Error: CSV file must contain 'smiles' or 'SMILES' column")
                return
            
            smiles_col = 'smiles' if 'smiles' in df.columns else 'SMILES'
            smiles_list = df[smiles_col].tolist()
            
            results = predictor.predict_batch(smiles_list, args.target, args.model)
            
            # Add predictions to dataframe
            df['predicted_pchembl'] = [r['prediction'] for r in results]
            df['prediction_error'] = [r['error'] for r in results]
            
            # Save results
            output_file = f"predictions_{args.target}_{args.model}.csv"
            df.to_csv(output_file, index=False)
            print(f"Predictions saved to {output_file}")
            
            # Print summary
            valid_predictions = [r for r in results if r['prediction'] is not None]
            print(f"Valid predictions: {len(valid_predictions)}/{len(results)}")
            if valid_predictions:
                predictions = [r['prediction'] for r in valid_predictions]
                print(f"Mean prediction: {np.mean(predictions):.3f}")
                print(f"Min prediction: {np.min(predictions):.3f}")
                print(f"Max prediction: {np.max(predictions):.3f}")
            
        except Exception as e:
            print(f"Error reading file: {str(e)}")
        return
    
    # If no arguments provided, show help
    parser.print_help()

if __name__ == "__main__":
    main() 