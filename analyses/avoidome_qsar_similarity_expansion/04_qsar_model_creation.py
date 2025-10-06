#!/usr/bin/env python3
"""
AQSE Workflow - Step 4: QSAR Model Creation (Optimized)

This script creates QSAR models for each avoidome target using different similarity 
threshold protein sets, with continuous pChEMBL value prediction.

OPTIMIZATION: Pre-computes molecular descriptors once and reuses them across all models.

Author: AQSE Pipeline
Date: 2025
"""

import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pickle
import json
from datetime import datetime
import hashlib
import tempfile

# Machine Learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Molecular descriptors
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from rdkit.Chem import Crippen, Lipinski

# Import custom descriptor functions
import sys
sys.path.append('/home/serramelendezcsm/RA/Avoidome/analyses')
from physicochemical_descriptors import calculate_physicochemical_descriptors
try:
    from run_esm_embeddings.single_esmc_embeddings import get_single_esmc_embedding
    ESM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ESM not available: {e}")
    ESM_AVAILABLE = False
    def get_single_esmc_embedding(protein_sequence, return_per_residue=False):
        """Dummy function when ESM is not available"""
        return np.zeros(1280)  # Return dummy embedding

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QSARModelCreation:
    """Handles QSAR model creation for AQSE workflow with optimized descriptor caching"""
    
    def __init__(self, similarity_results_dir: str, bioactivity_data_dir: str, 
                 output_dir: str, avoidome_file: str):
        """
        Initialize the QSAR model creation class
        
        Args:
            similarity_results_dir: Directory with similarity search results
            bioactivity_data_dir: Directory with bioactivity data
            output_dir: Output directory for results
            avoidome_file: Path to avoidome protein list
        """
        self.similarity_results_dir = Path(similarity_results_dir)
        self.bioactivity_data_dir = Path(bioactivity_data_dir)
        self.output_dir = Path(output_dir)
        self.avoidome_file = Path(avoidome_file)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.models_dir = self.output_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        self.features_dir = self.output_dir / "features"
        self.features_dir.mkdir(exist_ok=True)
        
        self.predictions_dir = self.output_dir / "predictions"
        self.predictions_dir.mkdir(exist_ok=True)
        
        self.metrics_dir = self.output_dir / "metrics"
        self.metrics_dir.mkdir(exist_ok=True)
        
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        # Descriptor cache directory
        self.descriptor_cache_dir = self.output_dir / "descriptor_cache"
        self.descriptor_cache_dir.mkdir(exist_ok=True)
        
        # Similarity thresholds
        self.thresholds = ['high', 'medium', 'low']
        
        # Load avoidome targets
        self.avoidome_targets = self.load_avoidome_targets()
        
        # Initialize results storage
        self.results = {}
        
        # Descriptor cache
        self.molecular_descriptor_cache = {}
        self.protein_descriptor_cache = {}
        
        # Set up temporary directory for ESM embeddings
        self.setup_temp_directory()
    
    def setup_temp_directory(self):
        """Set up temporary directory for ESM embeddings on zfsdata partition"""
        # Create a temporary directory on zfsdata partition
        temp_dir = Path("/zfsdata/data/cristina/temp_esm_embeddings")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Set environment variables for temporary directories
        os.environ['TMPDIR'] = str(temp_dir)
        os.environ['TMP'] = str(temp_dir)
        os.environ['TEMP'] = str(temp_dir)
        
        # Set tempfile default directory
        tempfile.tempdir = str(temp_dir)
        
        logger.info(f"Set temporary directory for ESM embeddings to: {temp_dir}")
        
    def load_avoidome_targets(self) -> List[Dict[str, str]]:
        """Load avoidome target protein information"""
        logger.info("Loading avoidome targets")
        
        try:
            df = pd.read_csv(self.avoidome_file)
            # Return list of dictionaries with both name and UniProt ID
            targets = []
            for _, row in df.iterrows():
                if pd.notna(row['UniProt ID']):
                    targets.append({
                        'name': row['Name_2'],
                        'uniprot_id': row['UniProt ID']
                    })
            logger.info(f"Loaded {len(targets)} avoidome targets")
            return targets
        except Exception as e:
            logger.error(f"Error loading avoidome targets: {e}")
            return []
    
    def load_similarity_results(self) -> Dict[str, Dict[str, List[str]]]:
        """Load similarity search results"""
        logger.info("Loading similarity search results")
        
        similarity_file = self.similarity_results_dir / "similarity_search_summary.csv"
        
        if not similarity_file.exists():
            logger.error(f"Similarity results not found: {similarity_file}")
            return {}
        
        try:
            df = pd.read_csv(similarity_file)
            
            # Convert to nested dictionary structure
            similarity_results = {}
            for _, row in df.iterrows():
                target = row['query_protein']
                threshold = row['threshold']
                num_similar = row['num_similar_proteins']
                
                if target not in similarity_results:
                    similarity_results[target] = {}
                
                # For now, we'll use all proteins from bioactivity data
                # In a real implementation, you'd extract the actual similar protein IDs
                similarity_results[target][threshold] = num_similar
            
            logger.info(f"Loaded similarity results for {len(similarity_results)} targets")
            return similarity_results
            
        except Exception as e:
            logger.error(f"Error loading similarity results: {e}")
            return {}
    
    def load_bioactivity_data(self, threshold: str) -> pd.DataFrame:
        """Load bioactivity data directly from Papyrus (like standardized QSAR)"""
        logger.info(f"Loading bioactivity data directly from Papyrus for {threshold} threshold")
        
        try:
            # Import Papyrus dataset
            from papyrus_scripts import PapyrusDataset
            
            # Load Papyrus dataset (same as standardized QSAR)
            logger.info("Loading Papyrus dataset...")
            papyrus_data = PapyrusDataset(version='latest', plusplus=True)
            papyrus_df = papyrus_data.to_dataframe()
            
            logger.info(f"Loaded {len(papyrus_df)} total activities from Papyrus")
            
            # Apply same filtering as standardized QSAR
            # Filter for valid SMILES and pchembl_value_Mean
            valid_data = papyrus_df.dropna(subset=['SMILES', 'pchembl_value_Mean'])
            valid_data = valid_data[valid_data['SMILES'] != '']
            
            logger.info(f"After filtering: {len(valid_data)} valid bioactivity records")
            return valid_data
            
        except Exception as e:
            logger.error(f"Error loading bioactivity data from Papyrus: {e}")
            return pd.DataFrame()
    
    def get_smiles_hash(self, smiles: str) -> str:
        """Generate a hash for SMILES string for caching"""
        return hashlib.md5(smiles.encode()).hexdigest()
    
    def get_protein_hash(self, protein_id: str) -> str:
        """Generate a hash for protein ID for caching"""
        return hashlib.md5(protein_id.encode()).hexdigest()
    
    def load_cached_molecular_descriptors(self, smiles_list: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """Load molecular descriptors from cache or compute if not cached"""
        logger.info(f"Loading molecular descriptors for {len(smiles_list)} compounds")
        
        cached_descriptors = []
        missing_smiles = []
        
        # Check cache for each SMILES
        for smiles in smiles_list:
            smiles_hash = self.get_smiles_hash(smiles)
            cache_file = self.descriptor_cache_dir / f"mol_desc_{smiles_hash}.pkl"
            
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        descriptor_row = pickle.load(f)
                    cached_descriptors.append(descriptor_row)
                except Exception as e:
                    logger.warning(f"Error loading cached descriptor for {smiles}: {e}")
                    missing_smiles.append(smiles)
            else:
                missing_smiles.append(smiles)
        
        logger.info(f"Found {len(cached_descriptors)} cached descriptors, need to compute {len(missing_smiles)}")
        return cached_descriptors, missing_smiles
    
    def compute_and_cache_molecular_descriptors(self, smiles_list: List[str]) -> List[Dict]:
        """Compute molecular descriptors and cache them"""
        logger.info(f"Computing molecular descriptors for {len(smiles_list)} compounds")
        
        descriptors = []
        
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    # Morgan fingerprints (ECFP4) - using full 2048 bits for better performance
                    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)  # Use full 2048 bits like standardized QSAR
                    morgan_array = np.array(morgan_fp, dtype=np.float32)  # Use float32 for memory efficiency
                    
                    # Use custom physicochemical descriptors
                    physico_desc = calculate_physicochemical_descriptors(smiles, include_sasa=False, verbose=False)
                    
                    descriptor_row = {
                        'SMILES': smiles,
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
                        'num_atoms': physico_desc['Num_Atoms'],  # Added missing descriptor
                        'num_saturated_rings': physico_desc['Num_Saturated_Rings']  # Added missing descriptor
                    }
                    
                    # Add Morgan fingerprint bits
                    for j, bit in enumerate(morgan_array):
                        descriptor_row[f'morgan_bit_{j}'] = int(bit)
                    
                    descriptors.append(descriptor_row)
                    
                    # Cache the descriptor
                    smiles_hash = self.get_smiles_hash(smiles)
                    cache_file = self.descriptor_cache_dir / f"mol_desc_{smiles_hash}.pkl"
                    with open(cache_file, 'wb') as f:
                        pickle.dump(descriptor_row, f)
                    
            except Exception as e:
                logger.warning(f"Error processing SMILES {smiles}: {e}")
                continue
        
        logger.info(f"Computed and cached descriptors for {len(descriptors)} valid compounds")
        return descriptors
    
    def generate_molecular_descriptors(self, smiles_list: List[str], chunk_size: int = 10000) -> pd.DataFrame:
        """Generate molecular descriptors with caching optimization and chunked processing"""
        logger.info(f"Generating molecular descriptors for {len(smiles_list)} compounds")
        
        # Process in chunks to manage memory
        all_descriptors = []
        
        for i in range(0, len(smiles_list), chunk_size):
            chunk_smiles = smiles_list[i:i+chunk_size]
            logger.info(f"Processing chunk {i//chunk_size + 1}/{(len(smiles_list)-1)//chunk_size + 1}: {len(chunk_smiles)} compounds")
            
            # Load cached descriptors for this chunk
            cached_descriptors, missing_smiles = self.load_cached_molecular_descriptors(chunk_smiles)
            
            # Compute missing descriptors for this chunk
            new_descriptors = []
            if missing_smiles:
                new_descriptors = self.compute_and_cache_molecular_descriptors(missing_smiles)
            
            # Combine cached and new descriptors for this chunk
            chunk_descriptors = cached_descriptors + new_descriptors
            all_descriptors.extend(chunk_descriptors)
            
            # Force garbage collection after each chunk
            import gc
            gc.collect()
        
        logger.info(f"Generated descriptors for {len(all_descriptors)} valid compounds")
        
        # Convert to DataFrame
        mol_desc_df = pd.DataFrame(all_descriptors)
        
        # Ensure 'SMILES' column exists (normalize from 'smiles' if needed)
        if 'smiles' in mol_desc_df.columns and 'SMILES' not in mol_desc_df.columns:
            mol_desc_df = mol_desc_df.rename(columns={'smiles': 'SMILES'})
        
        return mol_desc_df
    
    def load_cached_protein_descriptors(self, protein_ids: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """Load protein descriptors from cache or compute if not cached"""
        logger.info(f"Loading protein descriptors for {len(protein_ids)} proteins")
        
        cached_descriptors = []
        missing_proteins = []
        
        # Check cache for each protein
        for protein_id in protein_ids:
            protein_hash = self.get_protein_hash(protein_id)
            cache_file = self.descriptor_cache_dir / f"prot_desc_{protein_hash}.pkl"
            
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        descriptor_row = pickle.load(f)
                    cached_descriptors.append(descriptor_row)
                except Exception as e:
                    logger.warning(f"Error loading cached protein descriptor for {protein_id}: {e}")
                    missing_proteins.append(protein_id)
            else:
                missing_proteins.append(protein_id)
        
        logger.info(f"Found {len(cached_descriptors)} cached protein descriptors, need to compute {len(missing_proteins)}")
        return cached_descriptors, missing_proteins
    
    def compute_and_cache_protein_descriptors(self, protein_sequences: Dict[str, str]) -> List[Dict]:
        """Compute protein descriptors and cache them"""
        logger.info(f"Computing protein descriptors for {len(protein_sequences)} proteins")
        
        protein_descriptors = []
        
        for protein_id, sequence in protein_sequences.items():
            try:
                # Use ESM C embeddings
                esm_result = get_single_esmc_embedding(
                    protein_sequence=sequence,
                    model_name="esmc_300m",
                    return_per_residue=False,
                    verbose=False
                )
                
                # Extract embedding vector
                if isinstance(esm_result, dict):
                    esm_embedding = esm_result['embedding']
                else:
                    esm_embedding = esm_result
                
                # Ensure embedding is a 1D numpy array
                if hasattr(esm_embedding, 'shape') and len(esm_embedding.shape) > 1:
                    # If it's a 2D array, take the mean across the first dimension
                    esm_embedding = np.mean(esm_embedding, axis=0)
                elif hasattr(esm_embedding, 'flatten'):
                    # If it's not 1D, flatten it
                    esm_embedding = esm_embedding.flatten()
                
                # Convert to numpy array if it isn't already
                esm_embedding = np.array(esm_embedding)
                
                descriptor_row = {
                    'protein_id': protein_id,
                    'sequence_length': len(sequence)
                }
                
                # Add ESM embedding dimensions
                for i, val in enumerate(esm_embedding):
                    descriptor_row[f'esm_dim_{i}'] = float(val)
                
                protein_descriptors.append(descriptor_row)
                
                # Cache the descriptor
                protein_hash = self.get_protein_hash(protein_id)
                cache_file = self.descriptor_cache_dir / f"prot_desc_{protein_hash}.pkl"
                with open(cache_file, 'wb') as f:
                    pickle.dump(descriptor_row, f)
                
            except Exception as e:
                logger.warning(f"Error generating ESM embedding for {protein_id}: {e}")
                # Fallback to dummy embedding
                esm_embedding = np.random.randn(1280)
                descriptor_row = {
                    'protein_id': protein_id,
                    'sequence_length': len(sequence)
                }
                for i, val in enumerate(esm_embedding):
                    descriptor_row[f'esm_dim_{i}'] = float(val)
                protein_descriptors.append(descriptor_row)
                
                # Cache the fallback descriptor
                protein_hash = self.get_protein_hash(protein_id)
                cache_file = self.descriptor_cache_dir / f"prot_desc_{protein_hash}.pkl"
                with open(cache_file, 'wb') as f:
                    pickle.dump(descriptor_row, f)
        
        logger.info(f"Computed and cached protein descriptors for {len(protein_descriptors)} proteins")
        return protein_descriptors
    
    def generate_protein_descriptors(self, protein_sequences: Dict[str, str]) -> pd.DataFrame:
        """Generate protein descriptors with caching optimization"""
        logger.info(f"Generating protein descriptors for {len(protein_sequences)} proteins")
        
        protein_ids = list(protein_sequences.keys())
        
        # Load cached descriptors
        cached_descriptors, missing_proteins = self.load_cached_protein_descriptors(protein_ids)
        
        # Compute missing descriptors
        new_descriptors = []
        if missing_proteins:
            missing_sequences = {pid: protein_sequences[pid] for pid in missing_proteins}
            new_descriptors = self.compute_and_cache_protein_descriptors(missing_sequences)
        
        # Combine cached and new descriptors
        all_descriptors = cached_descriptors + new_descriptors
        
        logger.info(f"Generated protein descriptors for {len(all_descriptors)} proteins")
        return pd.DataFrame(all_descriptors)
    
    def load_protein_sequences(self, protein_ids: List[str]) -> Dict[str, str]:
        """Load protein sequences from Papyrus database"""
        logger.info(f"Loading protein sequences for {len(protein_ids)} proteins from Papyrus")
        
        sequences = {}
        
        try:
            # Import Papyrus dataset
            from papyrus_scripts import PapyrusDataset
            
            # Load Papyrus dataset
            logger.info("Loading Papyrus dataset to retrieve protein sequences...")
            papyrus_data = PapyrusDataset(version='latest', plusplus=True)
            
            # Get protein sequences from Papyrus
            proteins_df = papyrus_data.proteins().to_dataframe()
            
            # Create a mapping from target_id (UniProt ID + _WT) to Sequence
            protein_seq_map = dict(zip(proteins_df['target_id'], proteins_df['Sequence']))
            
            # Filter for the proteins we need
            for protein_id in protein_ids:
                # Add _WT suffix to match Papyrus target_id format
                papyrus_target_id = f"{protein_id}_WT"
                if papyrus_target_id in protein_seq_map:
                    sequences[protein_id] = protein_seq_map[papyrus_target_id]
                    logger.debug(f"Found sequence for {protein_id}: {len(protein_seq_map[papyrus_target_id])} amino acids")
                else:
                    logger.warning(f"Sequence not found in Papyrus for {protein_id} (looked for {papyrus_target_id})")
            
        except Exception as e:
            logger.error(f"Error loading sequences from Papyrus: {e}")
            # Fallback to Step 1 FASTA file if Papyrus fails
            logger.info("Falling back to Step 1 FASTA file...")
            step1_file = Path("/home/serramelendezcsm/RA/Avoidome/analyses/avoidome_qsar_similarity_expansion/01_input_preparation/avoidome_proteins_combined.fasta")
            
            if step1_file.exists():
                try:
                    from Bio import SeqIO
                    for record in SeqIO.parse(step1_file, "fasta"):
                        protein_id = record.id.split('|')[1] if '|' in record.id else record.id
                        if protein_id in protein_ids and protein_id not in sequences:
                            sequences[protein_id] = str(record.seq)
                except Exception as e2:
                    logger.warning(f"Error loading sequences from Step 1: {e2}")
        
        if len(sequences) < len(protein_ids):
            missing_proteins = set(protein_ids) - set(sequences.keys())
            logger.warning(f"Missing sequences for {len(missing_proteins)} proteins: {missing_proteins}")
            logger.warning("Only using proteins with available sequences")
        
        logger.info(f"Loaded sequences for {len(sequences)} proteins")
        return sequences
    
    def prepare_training_data(self, target: str, threshold: str, 
                            bioactivity_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare training and validation data for a target-threshold combination"""
        logger.info(f"Preparing data for {target} - {threshold} threshold")
        
        if bioactivity_data.empty:
            logger.warning(f"No bioactivity data for {threshold} threshold")
            return pd.DataFrame(), pd.DataFrame()
        
        # Get target protein UniProt ID
        target_uniprot = self.get_target_uniprot_id(target)
        if not target_uniprot:
            logger.warning(f"Could not find UniProt ID for target {target}")
            return pd.DataFrame(), pd.DataFrame()
        
        # Check if target protein has bioactivity data
        target_data = bioactivity_data[bioactivity_data['accession'] == target_uniprot]
        target_samples = len(target_data)
        
        logger.info(f"Target protein {target} ({target_uniprot}) has {target_samples} bioactivity samples")
        
        # Case 3: Target protein with less than 30 bioactivity points
        if target_samples < 30:
            logger.warning(f"Insufficient target protein data for {target}-{threshold}: {target_samples} samples (< 30 required)")
            return pd.DataFrame(), pd.DataFrame()
        
        # Case 1: Use only target protein data (like standardized QSAR single-protein models)
        # This ensures fair comparison with standardized QSAR models
        logger.info(f"Case 1: Using only target protein data - splitting 80/20 (like standardized QSAR)")
        from sklearn.model_selection import train_test_split
        
        train_data, val_data = train_test_split(
            target_data,
            test_size=0.2,
            random_state=42
        )
        
        logger.info(f"Training data: {len(train_data)} samples, Validation data: {len(val_data)} samples")
        logger.info(f"Both sets use only target protein: {target_uniprot}")
        
        return train_data, val_data
        
        # NOTE: Case 2 logic removed to ensure fair comparison with standardized QSAR models
        # All models now use only target protein data (Case 1) like standardized QSAR approach
    
    def get_target_uniprot_id(self, target: str) -> str:
        """Get UniProt ID for a target protein"""
        try:
            # Find target in the loaded avoidome targets
            for target_info in self.avoidome_targets:
                if target_info['name'] == target:
                    return target_info['uniprot_id']
            
            logger.warning(f"Target {target} not found in avoidome list")
            return None
        except Exception as e:
            logger.error(f"Error getting UniProt ID for {target}: {e}")
            return None
    
    def create_qsar_model(self, target: str, threshold: str, 
                         train_data: pd.DataFrame, val_data: pd.DataFrame) -> Dict[str, Any]:
        """Create QSAR model for a target-threshold combination using cached descriptors"""
        logger.info(f"Creating QSAR model for {target} - {threshold} threshold")
        
        try:
            # Get ONLY the proteins used in this specific model
            train_proteins = train_data['accession'].unique()
            val_proteins = val_data['accession'].unique()
            model_proteins = list(set(train_proteins) | set(val_proteins))
            
            # Print detailed model information
            print("\n" + "="*80)
            print(f"QSAR MODEL CREATION: {target} - {threshold.upper()} THRESHOLD")
            print("="*80)
            
            # Count datapoints per protein in training data
            train_protein_counts = train_data['accession'].value_counts().sort_index()
            print(f"\nTRAINING DATA:")
            print(f"  Total datapoints: {len(train_data)}")
            print(f"  Proteins used: {len(train_proteins)}")
            print(f"  Protein codes and datapoints:")
            for protein, count in train_protein_counts.items():
                print(f"    {protein}: {count} datapoints")
            
            # Count datapoints per protein in validation data
            val_protein_counts = val_data['accession'].value_counts().sort_index()
            print(f"\nVALIDATION DATA:")
            print(f"  Total datapoints: {len(val_data)}")
            print(f"  Proteins used: {len(val_proteins)}")
            print(f"  Protein codes and datapoints:")
            for protein, count in val_protein_counts.items():
                print(f"    {protein}: {count} datapoints")
            
            # Show overlap between training and validation proteins
            train_protein_set = set(train_proteins)
            val_protein_set = set(val_proteins)
            overlap = train_protein_set & val_protein_set
            train_only = train_protein_set - val_protein_set
            val_only = val_protein_set - train_protein_set
            
            print(f"\nPROTEIN OVERLAP ANALYSIS:")
            print(f"  Proteins in both train and validation: {len(overlap)}")
            if overlap:
                print(f"    {sorted(list(overlap))}")
            print(f"  Proteins only in training: {len(train_only)}")
            if train_only:
                print(f"    {sorted(list(train_only))}")
            print(f"  Proteins only in validation: {len(val_only)}")
            if val_only:
                print(f"    {sorted(list(val_only))}")
            
            logger.info(f"Computing protein descriptors for {len(model_proteins)} proteins used in {target}-{threshold} model")
            
            # Get protein sequences for ONLY the proteins used in this model
            model_protein_seqs = self.load_protein_sequences(model_proteins)
            
            # Filter data to only include proteins with available sequences BEFORE generating descriptors
            available_proteins = set(model_protein_seqs.keys())
            train_data = train_data[train_data['accession'].isin(available_proteins)]
            val_data = val_data[val_data['accession'].isin(available_proteins)]
            
            if train_data.empty or val_data.empty:
                logger.error(f"No data available for proteins with sequences in {target}-{threshold}")
                return {}
            
            # Update protein lists after filtering
            train_proteins = train_data['accession'].unique()
            val_proteins = val_data['accession'].unique()
            
            # Generate molecular descriptors AFTER filtering data
            train_smiles = train_data['SMILES'].tolist()
            val_smiles = val_data['SMILES'].tolist()
            
            train_mol_desc = self.generate_molecular_descriptors(train_smiles)
            val_mol_desc = self.generate_molecular_descriptors(val_smiles)
            
            if train_mol_desc.empty or val_mol_desc.empty:
                logger.error(f"Failed to generate molecular descriptors for {target}-{threshold}")
                return {}
            
            # Generate protein descriptors ONLY for these proteins
            model_prot_desc = self.generate_protein_descriptors(model_protein_seqs)
            
            # Split protein descriptors for train/val
            train_prot_desc = model_prot_desc[model_prot_desc['protein_id'].isin(train_proteins)]
            val_prot_desc = model_prot_desc[model_prot_desc['protein_id'].isin(val_proteins)]
            
            # Debug: Check sample sizes before combining features
            logger.info(f"Before combining features - Train: {len(train_data)} samples, Val: {len(val_data)} samples")
            logger.info(f"Molecular descriptors - Train: {len(train_mol_desc)} samples, Val: {len(val_mol_desc)} samples")
            logger.info(f"Protein descriptors - Train: {len(train_prot_desc)} samples, Val: {len(val_prot_desc)} samples")
            
            # Combine features and get targets
            train_features, y_train = self.combine_features(train_mol_desc, train_prot_desc, train_data)
            val_features, y_val = self.combine_features(val_mol_desc, val_prot_desc, val_data)
            
            if train_features.empty or val_features.empty:
                logger.error(f"Failed to combine features for {target}-{threshold}")
                return {}
            
            # Debug: Check sample sizes after combining features
            logger.info(f"After combining features - Train: {len(train_features)} samples, Val: {len(val_features)} samples")
            logger.info(f"Target variables - Train: {len(y_train)} samples, Val: {len(y_val)} samples")
            
            # Check for sample size consistency
            if len(train_features) != len(y_train):
                logger.error(f"Train features ({len(train_features)}) and targets ({len(y_train)}) size mismatch")
                return {}
            if len(val_features) != len(y_val):
                logger.error(f"Val features ({len(val_features)}) and targets ({len(y_val)}) size mismatch")
                return {}
            
            # Additional debugging for feature arrays
            logger.info(f"Train features shape: {train_features.shape}, dtype: {train_features.dtypes.iloc[0] if len(train_features) > 0 else 'empty'}")
            logger.info(f"Train targets shape: {y_train.shape}, dtype: {y_train.dtype}")
            logger.info(f"Val features shape: {val_features.shape}")
            logger.info(f"Val targets shape: {y_val.shape}")
            
            # Check for any NaN or infinite values
            if train_features.isnull().any().any():
                logger.warning("Train features contain NaN values")
            if val_features.isnull().any().any():
                logger.warning("Val features contain NaN values")
            if np.isnan(y_train).any():
                logger.warning("Train targets contain NaN values")
            if np.isnan(y_val).any():
                logger.warning("Val targets contain NaN values")
            
            # Train Random Forest model
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(train_features, y_train)
            
            # Make predictions
            y_train_pred = model.predict(train_features)
            y_val_pred = model.predict(val_features)
            
            # Calculate metrics
            train_r2 = r2_score(y_train, y_train_pred)
            val_r2 = r2_score(y_val, y_val_pred)
            val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
            val_mae = mean_absolute_error(y_val, y_val_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, train_features, y_train, cv=5, scoring='r2')
            q2 = cv_scores.mean()
            
            # Print detailed model performance information
            print(f"\nMODEL PERFORMANCE:")
            print(f"  Training R²: {train_r2:.4f}")
            print(f"  Validation R²: {val_r2:.4f}")
            print(f"  Validation RMSE: {val_rmse:.4f}")
            print(f"  Validation MAE: {val_mae:.4f}")
            print(f"  Cross-validation Q²: {q2:.4f}")
            print(f"  Q² Std Dev: {cv_scores.std():.4f}")
            
            # Print feature information
            print(f"\nFEATURE INFORMATION:")
            print(f"  Total features: {train_features.shape[1]}")
            print(f"  Molecular descriptors: {len([col for col in train_features.columns if col.startswith(('morgan_bit_', 'molecular_', 'logp', 'tpsa', 'hbd', 'hba', 'rotatable_', 'aromatic_', 'heavy_', 'num_', 'formal_', 'log_'))])}")
            print(f"  Protein descriptors: {len([col for col in train_features.columns if col.startswith('esm_dim_')])}")
            
            # Print data quality information
            print(f"\nDATA QUALITY:")
            print(f"  Training samples: {len(train_data)}")
            print(f"  Validation samples: {len(val_data)}")
            print(f"  Training features shape: {train_features.shape}")
            print(f"  Validation features shape: {val_features.shape}")
            print(f"  Missing values in training features: {train_features.isnull().sum().sum()}")
            print(f"  Missing values in validation features: {val_features.isnull().sum().sum()}")
            
            # Store results
            results = {
                'target': target,
                'threshold': threshold,
                'model': model,
                'train_r2': train_r2,
                'val_r2': val_r2,
                'val_rmse': val_rmse,
                'val_mae': val_mae,
                'q2': q2,
                'n_train': len(train_data),
                'n_val': len(val_data),
                'train_predictions': y_train_pred,
                'val_predictions': y_val_pred,
                'train_actual': y_train,
                'val_actual': y_val
            }
            
            print(f"\nMODEL STATUS: SUCCESS")
            print("="*80)
            
            logger.info(f"Model created for {target}-{threshold}: R²={val_r2:.3f}, RMSE={val_rmse:.3f}, Q²={q2:.3f}")
            
            return results
            
        except Exception as e:
            print(f"\nMODEL STATUS: FAILED")
            print(f"ERROR: {str(e)}")
            print("="*80)
            logger.error(f"Error creating model for {target}-{threshold}: {e}")
            return {}
    
    def combine_features(self, mol_desc: pd.DataFrame, prot_desc: pd.DataFrame, 
                        bioactivity_data: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Combine molecular and protein descriptors"""
        
        # Debug: Check input sizes
        logger.debug(f"combine_features input - bioactivity_data: {len(bioactivity_data)}, mol_desc: {len(mol_desc)}, prot_desc: {len(prot_desc)}")
        
        # Ensure molecular descriptors have 'SMILES' column (not 'smiles')
        if 'smiles' in mol_desc.columns and 'SMILES' not in mol_desc.columns:
            mol_desc = mol_desc.rename(columns={'smiles': 'SMILES'})
        
        # Merge molecular descriptors with bioactivity data
        combined = bioactivity_data.merge(mol_desc, on='SMILES', how='inner')
        logger.debug(f"After molecular merge: {len(combined)} samples")
        
        # Merge protein descriptors
        combined = combined.merge(prot_desc, left_on='accession', right_on='protein_id', how='inner')
        logger.debug(f"After protein merge: {len(combined)} samples")
        
        # Select feature columns (exclude metadata)
        ########## check???
        
        feature_cols = [col for col in combined.columns if col.startswith((
            'morgan_bit_', 'esm_dim_', 'molecular_', 'logp', 'tpsa', 'hbd', 'hba', 
            'rotatable_', 'aromatic_', 'heavy_', 'num_', 'formal_', 'log_'
        ))]
        
        result = combined[feature_cols]
        targets = combined['pchembl_value_Mean'].values
        
        logger.debug(f"Final features: {len(result)} samples with {len(feature_cols)} features")
        logger.debug(f"Final targets: {len(targets)} samples")
        
        return result, targets
    
    def save_model_results(self, results: Dict[str, Any]):
        """Save model results to files"""
        target = results['target']
        threshold = results['threshold']
        
        # Save model
        model_file = self.models_dir / f"{target}_{threshold}_model.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(results['model'], f)
        
        # Save predictions
        pred_data = pd.DataFrame({
            'actual': results['val_actual'],
            'predicted': results['val_predictions']
        })
        pred_file = self.predictions_dir / f"{target}_{threshold}_predictions.csv"
        pred_data.to_csv(pred_file, index=False)
        
        # Save metrics
        metrics = {
            'target': target,
            'threshold': threshold,
            'train_r2': results['train_r2'],
            'val_r2': results['val_r2'],
            'val_rmse': results['val_rmse'],
            'val_mae': results['val_mae'],
            'q2': results['q2'],
            'n_train': results['n_train'],
            'n_val': results['n_val']
        }
        
        metrics_file = self.metrics_dir / f"{target}_{threshold}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def create_summary_metrics(self):
        """Create summary metrics across all models"""
        logger.info("Creating summary metrics")
        
        all_metrics = []
        
        for target in self.avoidome_targets:
            for threshold in self.thresholds:
                metrics_file = self.metrics_dir / f"{target['name']}_{threshold}_{threshold}_metrics.json"
                if metrics_file.exists():
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                        all_metrics.append(metrics)
        
        if all_metrics:
            summary_df = pd.DataFrame(all_metrics)
            summary_file = self.metrics_dir / "model_performance_summary.csv"
            summary_df.to_csv(summary_file, index=False)
            
            logger.info(f"Saved summary metrics: {len(summary_df)} models")
            return summary_df
        else:
            logger.warning("No metrics found to summarize")
            return pd.DataFrame()
    
    def create_visualizations(self, summary_df: pd.DataFrame):
        """Create visualizations for model performance"""
        logger.info("Creating visualizations")
        
        if summary_df.empty:
            logger.warning("No data available for visualization")
            return
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Performance by threshold
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # R² by threshold
        sns.boxplot(data=summary_df, x='threshold', y='val_r2', ax=axes[0, 0])
        axes[0, 0].set_title('R² by Similarity Threshold')
        axes[0, 0].set_ylabel('R²')
        
        # RMSE by threshold
        sns.boxplot(data=summary_df, x='threshold', y='val_rmse', ax=axes[0, 1])
        axes[0, 1].set_title('RMSE by Similarity Threshold')
        axes[0, 1].set_ylabel('RMSE')
        
        # Q² by threshold
        sns.boxplot(data=summary_df, x='threshold', y='q2', ax=axes[1, 0])
        axes[1, 0].set_title('Q² by Similarity Threshold')
        axes[1, 0].set_ylabel('Q²')
        
        # Sample size by threshold
        sns.boxplot(data=summary_df, x='threshold', y='n_train', ax=axes[1, 1])
        axes[1, 1].set_title('Training Sample Size by Threshold')
        axes[1, 1].set_ylabel('Number of Samples')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'performance_by_threshold.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {self.plots_dir}")
    
    def cleanup_memory(self):
        """Clean up memory after processing each model"""
        import gc
        gc.collect()
        logger.debug("Memory cleanup completed")
    
    def get_similar_proteins_for_target(self, target: str, threshold: str) -> List[str]:
        """Get similar proteins for a specific target and threshold from similarity search summary"""
        # Load similarity search summary
        summary_file = self.similarity_results_dir / "similarity_search_summary.csv"
        
        if not summary_file.exists():
            logger.warning(f"Similarity search summary not found: {summary_file}")
            return []
        
        try:
            # Read similarity search summary
            summary_df = pd.read_csv(summary_file)
            
            # Get target protein UniProt ID
            target_uniprot = self.get_target_uniprot_id(target)
            if not target_uniprot:
                logger.warning(f"Could not find UniProt ID for target {target}")
                return []
            
            # Find the row for this target and threshold
            target_row = summary_df[
                (summary_df['query_protein'] == target_uniprot) & 
                (summary_df['threshold'] == threshold)
            ]
            
            if target_row.empty:
                logger.warning(f"No similarity data found for {target} ({target_uniprot}) with {threshold} threshold")
                return []
            
            # Parse similar proteins string
            similar_proteins_str = target_row.iloc[0]['similar_proteins']
            similar_proteins = []
            
            if pd.notna(similar_proteins_str) and similar_proteins_str.strip():
                # Extract protein IDs from string like "P05177_WT (100.0%), P20813_WT (78.0%)"
                for protein_entry in similar_proteins_str.split(', '):
                    if '(' in protein_entry:
                        protein_id = protein_entry.split('_WT')[0]  # Remove _WT suffix
                        # Include all proteins (including self) for the model
                        similar_proteins.append(protein_id)
            
            logger.info(f"Found {len(similar_proteins)} similar proteins for {target} ({target_uniprot}) with {threshold} threshold")
            return similar_proteins
            
        except Exception as e:
            logger.error(f"Error loading similar proteins for {target} from similarity search summary: {e}")
            return []
    
    def precompute_molecular_descriptors_only(self) -> Dict[str, Any]:
        """Pre-compute only molecular descriptors for all unique SMILES across thresholds"""
        logger.info("Pre-computing molecular descriptors for optimization")
        
        all_smiles = set()
        
        # Collect all unique SMILES across all thresholds
        for threshold in self.thresholds:
            bioactivity_data = self.load_bioactivity_data(threshold)
            if not bioactivity_data.empty:
                all_smiles.update(bioactivity_data['SMILES'].tolist())
        
        logger.info(f"Found {len(all_smiles)} unique SMILES across all thresholds")
        
        # Pre-compute molecular descriptors
        if all_smiles:
            logger.info("Pre-computing molecular descriptors")
            self.generate_molecular_descriptors(list(all_smiles))
        
        logger.info("Molecular descriptor pre-computation completed")
        return {
            'unique_smiles': len(all_smiles),
            'molecular_descriptors_cached': len(all_smiles)
        }
    
    def run_qsar_model_creation(self) -> Dict[str, Any]:
        """Run the complete QSAR model creation workflow with optimized descriptor caching"""
        logger.info("Starting AQSE QSAR model creation with optimized descriptor caching")
        
        # Pre-compute only molecular descriptors (protein descriptors computed per model)
        precompute_results = self.precompute_molecular_descriptors_only()
        logger.info(f"Molecular descriptor pre-computation results: {precompute_results}")
        
        # Process each target individually
        for target_info in self.avoidome_targets:
            target_name = target_info['name']
            target_uniprot = target_info['uniprot_id']
            
            logger.info(f"Processing target: {target_name} ({target_uniprot})")
            
            # Process each threshold for this target
            for threshold in self.thresholds:
                logger.info(f"Processing {target_name} - {threshold} threshold")
                
                # Get similar proteins for this target
                similar_proteins = self.get_similar_proteins_for_target(target_name, threshold)
                
                if not similar_proteins:
                    print(f"\n{'='*60}")
                    print(f"SKIPPING: {target_name} - {threshold.upper()} THRESHOLD")
                    print(f"REASON: No similar proteins found")
                    print(f"{'='*60}")
                    logger.warning(f"No similar proteins found for {target_name} - {threshold}")
                    continue
                
                # Load bioactivity data
                bioactivity_data = self.load_bioactivity_data(threshold)
                if bioactivity_data.empty:
                    print(f"\n{'='*60}")
                    print(f"SKIPPING: {target_name} - {threshold.upper()} THRESHOLD")
                    print(f"REASON: No bioactivity data for {threshold} threshold")
                    print(f"{'='*60}")
                    logger.warning(f"No bioactivity data for {threshold} threshold")
                    continue
                
                # Check if target protein has bioactivity data
                target_data = bioactivity_data[bioactivity_data['accession'] == target_uniprot]
                if target_data.empty:
                    print(f"\n{'='*60}")
                    print(f"SKIPPING: {target_name} - {threshold.upper()} THRESHOLD")
                    print(f"REASON: Target protein {target_name} ({target_uniprot}) not found in bioactivity data")
                    print(f"{'='*60}")
                    logger.warning(f"Target protein {target_name} ({target_uniprot}) not found in bioactivity data for {threshold}")
                    continue
                
                # Filter bioactivity data to include similar proteins + target protein
                all_relevant_proteins = similar_proteins + [target_uniprot]
                target_bioactivity_data = bioactivity_data[
                    bioactivity_data['accession'].isin(all_relevant_proteins)
                ].copy()
                
                if target_bioactivity_data.empty:
                    print(f"\n{'='*60}")
                    print(f"SKIPPING: {target_name} - {threshold.upper()} THRESHOLD")
                    print(f"REASON: No bioactivity data for relevant proteins")
                    print(f"{'='*60}")
                    logger.warning(f"No bioactivity data for relevant proteins of {target_name} - {threshold}")
                    continue
                
                # Print initial data summary
                print(f"\n{'='*60}")
                print(f"PROCESSING: {target_name} - {threshold.upper()} THRESHOLD")
                print(f"{'='*60}")
                print(f"Target protein: {target_name} ({target_uniprot})")
                print(f"Similar proteins found: {len(similar_proteins)}")
                print(f"Total bioactivity records: {len(target_bioactivity_data)}")
                print(f"Proteins in dataset: {sorted(target_bioactivity_data['accession'].unique())}")
                
                # Count datapoints per protein in the filtered dataset
                protein_counts = target_bioactivity_data['accession'].value_counts().sort_index()
                print(f"Datapoints per protein:")
                for protein, count in protein_counts.items():
                    print(f"  {protein}: {count} datapoints")
                
                logger.info(f"Found {len(target_bioactivity_data)} bioactivity records for {target_name} - {threshold}")
                
                # Prepare training data (using the corrected logic)
                train_data, val_data = self.prepare_training_data(target_name, threshold, target_bioactivity_data)
                
                if train_data.empty or val_data.empty:
                    print(f"\n{'='*60}")
                    print(f"SKIPPING: {target_name} - {threshold.upper()} THRESHOLD")
                    print(f"REASON: Insufficient data after train/val split")
                    print(f"  Training samples: {len(train_data)}")
                    print(f"  Validation samples: {len(val_data)}")
                    print(f"{'='*60}")
                    logger.warning(f"Insufficient data for {target_name} - {threshold}")
                    continue
                
                # Create QSAR model (protein descriptors computed only for this model's proteins)
                results = self.create_qsar_model(f"{target_name}_{threshold}", threshold, train_data, val_data)
                
                if results:
                    self.save_model_results(results)
                    self.results[f"{target_name}_{threshold}"] = results
                    logger.info(f"Successfully created model for {target_name} - {threshold}")
                
                # Clean up memory after each model
                self.cleanup_memory()
        
        # Create summary metrics
        summary_df = self.create_summary_metrics()
        
        # Create visualizations
        self.create_visualizations(summary_df)
        
        # Print final summary
        print(f"\n{'='*80}")
        print("QSAR MODEL CREATION COMPLETED")
        print(f"{'='*80}")
        print(f"Total models created: {len(self.results)}")
        print(f"Total avoidome targets processed: {len(self.avoidome_targets)}")
        print(f"Thresholds processed: {self.thresholds}")
        
        if self.results:
            print(f"\nSUCCESSFUL MODELS:")
            for model_key, model_results in self.results.items():
                target, threshold = model_key.rsplit('_', 1)
                print(f"  {target} - {threshold.upper()}: R²={model_results['val_r2']:.3f}, Q²={model_results['q2']:.3f}")
        
        if summary_df is not None and not summary_df.empty:
            print(f"\nPERFORMANCE SUMMARY:")
            print(f"  Average R²: {summary_df['val_r2'].mean():.3f} ± {summary_df['val_r2'].std():.3f}")
            print(f"  Average Q²: {summary_df['q2'].mean():.3f} ± {summary_df['q2'].std():.3f}")
            print(f"  Average RMSE: {summary_df['val_rmse'].mean():.3f} ± {summary_df['val_rmse'].std():.3f}")
            print(f"  Total training samples: {summary_df['n_train'].sum()}")
            print(f"  Total validation samples: {summary_df['n_val'].sum()}")
        
        print(f"{'='*80}")
        
        logger.info("QSAR model creation completed successfully")
        
        return {
            'models_created': len(self.results),
            'summary_metrics': summary_df,
            'results': self.results,
            'precompute_results': precompute_results
        }

def main():
    """Main function to run QSAR model creation"""
    
    # Set up paths - using /zfsdata/data/cristina/ for temporary storage
    similarity_results_dir = "/home/serramelendezcsm/RA/Avoidome/analyses/avoidome_qsar_similarity_expansion/02_similarity_search"
    bioactivity_data_dir = "/home/serramelendezcsm/RA/Avoidome/analyses/avoidome_qsar_similarity_expansion/03_data_collection"
    output_dir = "/zfsdata/data/cristina/avoidome_qsar_models_temp"
    avoidome_file = "/home/serramelendezcsm/RA/Avoidome/primary_data/avoidome_prot_list.csv"
    
    # Initialize QSAR model creation class
    qsar_creator = QSARModelCreation(similarity_results_dir, bioactivity_data_dir, output_dir, avoidome_file)
    
    # Run QSAR model creation
    results = qsar_creator.run_qsar_model_creation()
    
    if results:
        print("\n" + "="*60)
        print("AQSE QSAR MODEL CREATION COMPLETED (OPTIMIZED)")
        print("="*60)
        print(f"Models created: {results['models_created']}")
        print(f"Summary metrics: {len(results['summary_metrics'])} rows")
        print(f"Pre-computation results: {results['precompute_results']}")
        print("="*60)
    else:
        print("QSAR model creation failed - check logs for details")

if __name__ == "__main__":
    main()
