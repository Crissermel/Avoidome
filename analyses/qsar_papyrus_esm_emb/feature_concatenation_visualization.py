#!/usr/bin/env python3
"""
Feature Concatenation Visualization Script with Real Data

This script demonstrates the concatenation of Morgan fingerprints with ESM embeddings
using real data from Papyrus dataset and visualizes the shapes and distributions.

Date: 2025-01-30
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
import warnings
warnings.filterwarnings('ignore')

# Papyrus imports
from papyrus_scripts import PapyrusDataset

# Molecular fingerprinting
from rdkit import Chem
from rdkit.Chem import AllChem

# Setup plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class FeatureConcatenationVisualizer:
    """
    Visualize the concatenation of Morgan fingerprints with ESM embeddings using real data
    """
    
    def __init__(self, 
                 embeddings_path: str = "/home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_esm_emb/embeddings.npy",
                 targets_path: str = "/home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_esm_emb/targets_w_sequences.csv",
                 data_path: str = "/home/serramelendezcsm/RA/Avoidome/processed_data/papyrus_protein_check_results.csv"):
        """
        Initialize the visualizer
        
        Args:
            embeddings_path: Path to the ESM embeddings numpy file
            targets_path: Path to the targets with sequences CSV file
            data_path: Path to the protein check results CSV file
        """
        self.embeddings_path = embeddings_path
        self.targets_path = targets_path
        self.data_path = data_path
        self.embeddings = None
        self.targets_df = None
        self.proteins_df = None
        self.papyrus_df = None
        
    def load_data(self):
        """Load ESM embeddings, targets data, and Papyrus dataset"""
        print("Loading ESM embeddings, targets data, and Papyrus dataset...")
        
        # Load ESM embeddings
        self.embeddings = np.load(self.embeddings_path)
        print(f"Loaded ESM embeddings with shape: {self.embeddings.shape}")
        
        # Load targets with sequences
        self.targets_df = pd.read_csv(self.targets_path)
        print(f"Loaded {len(self.targets_df)} targets with sequences")
        
        # Load protein check results
        self.proteins_df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.proteins_df)} proteins from check results")
        
        # Initialize Papyrus dataset
        print("Loading Papyrus dataset...")
        papyrus_data = PapyrusDataset(version='latest', plusplus=True)
        self.papyrus_df = papyrus_data.to_dataframe()
        print(f"Loaded {len(self.papyrus_df)} total activities from Papyrus")
        
    def get_real_protein_compound_data(self, protein_name: str = "CYP1A2") -> tuple:
        """
        Get real protein-compound data from Papyrus
        
        Args:
            protein_name: Name of the protein to get data for
            
        Returns:
            Tuple of (morgan_fps, esm_embedding, protein_info)
        """
        print(f"Getting real data for protein: {protein_name}")
        
        # Find the protein in proteins_df
        protein_row = self.proteins_df[self.proteins_df['name2_entry'] == protein_name]
        if len(protein_row) == 0:
            print(f"Protein {protein_name} not found, using first available protein")
            protein_row = self.proteins_df.iloc[0:1]
            protein_name = protein_row.iloc[0]['name2_entry']
        
        # Get UniProt IDs
        human_id = protein_row.iloc[0]['human_uniprot_id']
        mouse_id = protein_row.iloc[0]['mouse_uniprot_id']
        rat_id = protein_row.iloc[0]['rat_uniprot_id']
        
        # Get activities for this protein
        all_activities = []
        for uniprot_id in [human_id, mouse_id, rat_id]:
            if pd.notna(uniprot_id):
                activities = self.papyrus_df[self.papyrus_df['accession'] == uniprot_id]
                if len(activities) > 0:
                    all_activities.append(activities)
        
        if not all_activities:
            print(f"No activities found for {protein_name}, using sample data")
            return self._create_sample_data()
        
        # Combine activities
        combined_activities = pd.concat(all_activities, ignore_index=True)
        print(f"Found {len(combined_activities)} activities for {protein_name}")
        
        # Get ESM embedding
        esm_embedding = None
        for uniprot_id in [human_id, mouse_id, rat_id]:
            if pd.notna(uniprot_id):
                protein_row_targets = self.targets_df[
                    (self.targets_df['human_uniprot_id'] == uniprot_id) |
                    (self.targets_df['mouse_uniprot_id'] == uniprot_id) |
                    (self.targets_df['rat_uniprot_id'] == uniprot_id)
                ]
                if len(protein_row_targets) > 0:
                    protein_idx = protein_row_targets.index[0]
                    esm_embedding = self.embeddings[protein_idx]
                    print(f"Retrieved ESM embedding for {protein_name} at index {protein_idx}")
                    break
        
        if esm_embedding is None:
            print(f"No ESM embedding found for {protein_name}, using sample data")
            return self._create_sample_data()
        
        # Create Morgan fingerprints from real SMILES
        smiles_list = combined_activities['SMILES'].tolist()
        morgan_fps, valid_indices = self.create_morgan_fingerprints(smiles_list)
        
        if len(morgan_fps) == 0:
            print(f"No valid fingerprints created for {protein_name}, using sample data")
            return self._create_sample_data()
        
        # Filter activities to only include valid fingerprints
        filtered_activities = combined_activities.iloc[valid_indices]
        
        print(f"Successfully created {len(morgan_fps)} Morgan fingerprints for {protein_name}")
        print(f"Using ESM embedding with shape: {esm_embedding.shape}")
        
        protein_info = {
            'name': protein_name,
            'n_activities': len(combined_activities),
            'n_valid_fingerprints': len(morgan_fps),
            'uniprot_ids': [human_id, mouse_id, rat_id]
        }
        
        return morgan_fps, esm_embedding, protein_info
    
    def _create_sample_data(self) -> tuple:
        """Create sample data if real data is not available"""
        print("Creating sample data for demonstration...")
        
        # Sample SMILES strings
        sample_smiles = [
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
            "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F",  # Celecoxib
            "CC1=C(C(C(=C(N1CCC(C(CC)CC)C)CC2=CC=C(C=C2)NS(=O)(=O)C(F)(F)F)CC3C=CC(=CC3=O)O)C(=O)O",  # Atorvastatin
        ]
        
        fingerprints = []
        for smiles in sample_smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                fingerprints.append(np.array(fp))
        
        morgan_fps = np.array(fingerprints)
        esm_embedding = self.embeddings[0]  # Use first protein's embedding
        
        protein_info = {
            'name': 'Sample Protein',
            'n_activities': len(sample_smiles),
            'n_valid_fingerprints': len(morgan_fps),
            'uniprot_ids': ['Sample']
        }
        
        return morgan_fps, esm_embedding, protein_info
    
    def create_morgan_fingerprints(self, smiles_list: list, radius: int = 2, nBits: int = 2048) -> tuple:
        """
        Create Morgan fingerprints from SMILES strings
        
        Args:
            smiles_list: List of SMILES strings
            radius: Morgan fingerprint radius
            nBits: Number of bits in fingerprint
            
        Returns:
            Tuple of (fingerprint array, valid indices)
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
                    print(f"Invalid SMILES: {smiles}")
            except Exception as e:
                print(f"Error creating fingerprint for {smiles}: {e}")
        
        if not fingerprints:
            return np.array([]), []
            
        return np.array(fingerprints), valid_indices
    
    def concatenate_features(self, morgan_fps: np.ndarray, esm_embedding: np.ndarray) -> np.ndarray:
        """
        Concatenate Morgan fingerprints with ESM embeddings
        
        Args:
            morgan_fps: Morgan fingerprints array
            esm_embedding: ESM embedding array
            
        Returns:
            Concatenated feature array
        """
        print(f"Concatenating features...")
        print(f"  Morgan fingerprints shape: {morgan_fps.shape}")
        print(f"  ESM embedding shape: {esm_embedding.shape}")
        
        # Repeat ESM embedding for each sample
        esm_features = np.tile(esm_embedding, (len(morgan_fps), 1))
        print(f"  Repeated ESM features shape: {esm_features.shape}")
        
        # Concatenate Morgan fingerprints with ESM features
        combined_features = np.concatenate([morgan_fps, esm_features], axis=1)
        print(f"  Combined features shape: {combined_features.shape}")
        
        return combined_features
    
    def visualize_feature_shapes(self, morgan_fps: np.ndarray, esm_embedding: np.ndarray, 
                                combined_features: np.ndarray, protein_info: dict):
        """
        Visualize the shapes of different feature types
        
        Args:
            morgan_fps: Morgan fingerprints array
            esm_embedding: ESM embedding array
            combined_features: Combined features array
            protein_info: Information about the protein
        """
        print("Creating feature shape visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Feature dimensions comparison
        feature_types = ['Morgan FPs', 'ESM Embedding', 'Combined Features']
        feature_dims = [morgan_fps.shape[1], esm_embedding.shape[0], combined_features.shape[1]]
        
        bars = axes[0, 0].bar(feature_types, feature_dims, color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[0, 0].set_title(f'Feature Dimensions Comparison\n{protein_info["name"]}', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Number of Features')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, dim in zip(bars, feature_dims):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                           str(dim), ha='center', va='bottom', fontweight='bold')
        
        # 2. Feature sparsity comparison
        morgan_sparsity = 1 - np.mean(morgan_fps)
        esm_sparsity = 0  # ESM embeddings are dense
        combined_sparsity = 1 - np.mean(combined_features)
        
        sparsity_data = [morgan_sparsity, esm_sparsity, combined_sparsity]
        sparsity_bars = axes[0, 1].bar(feature_types, sparsity_data, color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[0, 1].set_title('Feature Sparsity Comparison', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Sparsity (1 - Mean)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, sparsity in zip(sparsity_bars, sparsity_data):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{sparsity:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Morgan fingerprint distribution
        morgan_means = np.mean(morgan_fps, axis=0)
        axes[1, 0].hist(morgan_means, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].set_title('Morgan Fingerprint Bit Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Mean Bit Value')
        axes[1, 0].set_ylabel('Frequency')
        
        # 4. ESM embedding distribution
        axes[1, 1].hist(esm_embedding, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[1, 1].set_title('ESM Embedding Distribution', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Embedding Value')
        axes[1, 1].set_ylabel('Frequency')
        
        # Add protein information as text
        info_text = f"Protein: {protein_info['name']}\n"
        info_text += f"Activities: {protein_info['n_activities']}\n"
        info_text += f"Valid Fingerprints: {protein_info['n_valid_fingerprints']}\n"
        info_text += f"UniProt IDs: {', '.join([str(x) for x in protein_info['uniprot_ids'] if pd.notna(x)])}"
        
        fig.text(0.02, 0.02, info_text, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        plt.savefig('/home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_esm_emb/real_data_feature_shapes.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Feature shape visualization saved as 'real_data_feature_shapes.png'")
    
    def visualize_combined_features_pca(self, combined_features: np.ndarray, protein_info: dict):
        """
        Visualize combined features using PCA
        
        Args:
            combined_features: Combined features array
            protein_info: Information about the protein
        """
        print("Creating PCA visualization of combined features...")
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(combined_features)
        
        # Apply PCA
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features_scaled)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. PCA scatter plot
        scatter = ax1.scatter(features_pca[:, 0], features_pca[:, 1], 
                            alpha=0.6, s=50, c=range(len(features_pca)), 
                            cmap='viridis', edgecolors='white', linewidth=0.5)
        ax1.set_title(f'PCA of Combined Features (Morgan + ESM)\n{protein_info["name"]}', fontsize=14, fontweight='bold')
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax1.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Sample Index')
        
        # 2. Explained variance ratio
        n_components = min(50, combined_features.shape[1])
        pca_full = PCA(n_components=n_components)
        pca_full.fit(features_scaled)
        
        cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
        ax2.plot(range(1, n_components + 1), cumulative_variance, 'bo-', linewidth=2, markersize=6)
        ax2.set_title('Cumulative Explained Variance Ratio', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Explained Variance Ratio')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='95% variance')
        ax2.axhline(y=0.99, color='orange', linestyle='--', alpha=0.7, label='99% variance')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('/home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_esm_emb/real_data_combined_features_pca.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("PCA visualization saved as 'real_data_combined_features_pca.png'")
        
        # Print variance information
        print(f"\nPCA Information for {protein_info['name']}:")
        print(f"  Total features: {combined_features.shape[1]}")
        print(f"  Components needed for 95% variance: {np.argmax(cumulative_variance >= 0.95) + 1}")
        print(f"  Components needed for 99% variance: {np.argmax(cumulative_variance >= 0.99) + 1}")
    
    def visualize_umap_embedding(self, combined_features: np.ndarray, protein_info: dict):
        """
        Visualize combined features using UMAP
        
        Args:
            combined_features: Combined features array
            protein_info: Information about the protein
        """
        print("Creating UMAP visualization of combined features...")
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(combined_features)
        
        # Apply UMAP
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(combined_features)-1), min_dist=0.1)
        features_umap = reducer.fit_transform(features_scaled)
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        
        scatter = plt.scatter(features_umap[:, 0], features_umap[:, 1], 
                            alpha=0.7, s=60, c=range(len(features_umap)), 
                            cmap='plasma', edgecolors='white', linewidth=0.5)
        
        plt.title(f'UMAP of Combined Features (Morgan + ESM)\n{protein_info["name"]}', fontsize=16, fontweight='bold')
        plt.xlabel('UMAP-1', fontsize=12)
        plt.ylabel('UMAP-2', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Sample Index')
        
        plt.tight_layout()
        plt.savefig('/home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_esm_emb/real_data_combined_features_umap.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("UMAP visualization saved as 'real_data_combined_features_umap.png'")
    
    def run_visualization_pipeline(self, protein_name: str = "CYP1A2"):
        """Run the complete visualization pipeline with real data"""
        print(f"Starting feature concatenation visualization pipeline with real data...")
        print(f"Target protein: {protein_name}")
        
        # Load data
        self.load_data()
        
        # Get real protein-compound data
        morgan_fps, esm_embedding, protein_info = self.get_real_protein_compound_data(protein_name)
        
        # Concatenate features
        combined_features = self.concatenate_features(morgan_fps, esm_embedding)
        
        # Create visualizations
        self.visualize_feature_shapes(morgan_fps, esm_embedding, combined_features, protein_info)
        self.visualize_combined_features_pca(combined_features, protein_info)
        self.visualize_umap_embedding(combined_features, protein_info)
        
        print(f"\nVisualization pipeline completed for {protein_info['name']}!")
        print("Generated files:")
        print("  - real_data_feature_shapes.png")
        print("  - real_data_combined_features_pca.png")
        print("  - real_data_combined_features_umap.png")
        
        # Print summary statistics
        print(f"\nSummary for {protein_info['name']}:")
        print(f"  - Morgan fingerprints: {morgan_fps.shape[0]} compounds × {morgan_fps.shape[1]} features")
        print(f"  - ESM embedding: {esm_embedding.shape[0]} features")
        print(f"  - Combined features: {combined_features.shape[0]} compounds × {combined_features.shape[1]} features")
        print(f"  - Total activities found: {protein_info['n_activities']}")
        print(f"  - Valid fingerprints created: {protein_info['n_valid_fingerprints']}")

def main():
    """Main function to run the visualization pipeline with real data"""
    visualizer = FeatureConcatenationVisualizer()
    visualizer.run_visualization_pipeline("CYP1A2")  # You can change the protein name here

if __name__ == "__main__":
    main() 