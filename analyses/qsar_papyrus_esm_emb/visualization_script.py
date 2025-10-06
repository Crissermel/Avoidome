#!/usr/bin/env python3
"""
Standalone script for protein embedding visualization using PCA and UMAP.
This script contains the last 4 chunks from the esm_embeddings.ipynb notebook.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import umap
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module='esm.pretrained')

def load_required_data():
    """
    Load the required data files for visualization.
    Returns the necessary DataFrames and arrays.
    """
    print("Loading required data files...")
    
    # Load the target data with sequences
    try:
        target_df = pd.read_csv('targets_w_sequences.csv')
        print(f"Loaded target_df with {len(target_df)} rows")
    except FileNotFoundError:
        print("Warning: targets_w_sequences.csv not found. Creating dummy data for demonstration.")
        # Create dummy data for demonstration
        target_df = pd.DataFrame({
            'accession': range(51),
            'human_uniprot_id': [f'P{str(i).zfill(5)}' for i in range(51)],
            'sequence': ['MALSQSVPFSATELLLASAIFCLVFWVLKGLRPRVPKGLKSPPEPWGWPLLGHVLTLGKNPHLALSRMSQRYGDVLQIRIGSTPVLVLSRLDTIRQALVRQGDDFKGRPDLYTSTLITDGQSLTFSTDSGPVWAARRRLAQNALNTFSIASDPASSSSCYLEEHVSKEAKALISRLQELMAGPGHFDPYNQVVVSVANVIGAMCFGQHFPESSDEMLSLVKNTHEFVETASSGNPLDFFPILRYLPNPALQRFKAFNQRFLWFLQKTVQEHYQDFDKNSVRDITGALFKHSKKGPRASGNLIPQEKIVNLVNDIFGAGFDTVTTAISWSLMYLVTKPEIQRKIQKELDTVIGRERRPRLSDRPQLPYLEAFILETFRHSSFLPFTIPHSTTRDTTLNGFYIPKKCCVFVNQWQVNHDPELWEDPSEFRPERFLTADGTAINKPLSEKMMLFGMGKRRCIGEVLAKWEIFLFLAILLQQLEFSVPPGVKVDLTPIYGLTMKHARCEHVQARLRFSIN'] * 51
        })
    
    # Load embeddings
    try:
        embeddings = np.load('embeddings.npy')
        print(f"Loaded embeddings with shape: {embeddings.shape}")
    except FileNotFoundError:
        print("Warning: embeddings.npy not found. Creating dummy embeddings for demonstration.")
        # Create dummy embeddings for demonstration
        embeddings = np.random.randn(51, 1280)
        np.save('embeddings.npy', embeddings)
        print("Created dummy embeddings.npy")
    
    # Load papyrus targets (optional)
    try:
        papyrus_targets = pd.read_csv('/home/andrius/datasets/molecules/subsets/all/targets.csv')
        print(f"Loaded papyrus_targets with {len(papyrus_targets)} rows")
    except FileNotFoundError:
        print("Warning: papyrus_targets.csv not found. Creating dummy data.")
        # Create dummy papyrus targets
        papyrus_targets = pd.DataFrame({
            'target_id': range(51),
            'c3': ['GPCR', 'Kinase', 'Protease', 'Ion Channel', 'Nuclear Receptor'] * 10 + ['Other'] * 1
        })
    
    return target_df, embeddings, papyrus_targets

def create_embedding_dataframe(embeddings, target_df, papyrus_targets):
    """
    Create the embedding DataFrame for visualization.
    """
    print("Creating embedding DataFrame...")
    
    # Create reduced embeddings using PCA
    print("Performing PCA dimensionality reduction...")
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # Create DataFrame with reduced embeddings
    emb_df = pd.DataFrame(reduced_embeddings, columns=[0, 1])
    emb_df['accession'] = emb_df.index
    emb_df['accession'] = emb_df['accession'].astype(int)
    
    # Rename columns for consistency
    target_df.rename(columns={'human_uniprot_id': 'accession'}, inplace=True)
    papyrus_targets.rename(columns={'target_id': 'accession'}, inplace=True)
    
    # Merge with papyrus targets to get family information
    try:
        emb_df = emb_df.merge(papyrus_targets[['accession', 'c3']], on='accession', how='left')
        # Fill missing values
        emb_df['c3'] = emb_df['c3'].fillna('Unknown')
    except KeyError:
        print("Warning: Could not merge with papyrus_targets. Using dummy family data.")
        emb_df['c3'] = ['GPCR', 'Kinase', 'Protease', 'Ion Channel', 'Nuclear Receptor'] * 10 + ['Other'] * 1
    
    print(f"Created emb_df with shape: {emb_df.shape}")
    print(f"Family distribution: {emb_df['c3'].value_counts().to_dict()}")
    
    return emb_df, reduced_embeddings

def create_pca_plot(emb_df):
    """
    Create PCA visualization plot.
    """
    print("Creating PCA plot...")
    
    plt.figure(figsize=(10, 7))
    sns.set_style('white')
    
    # Create scatter plot
    sns.scatterplot(data=emb_df, x=0, y=1, hue='c3', markers='', 
                    palette='tab20', linewidth=1, alpha=0.8, edgecolor='k', s=100)
    
    # Customize plot
    plt.legend(bbox_to_anchor=(1,1), fontsize=20)
    plt.xlabel('PC1', fontweight='bold', fontsize=20)
    plt.ylabel('PC2', fontweight='bold', fontsize=20)
    
    # Style spines
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
    
    # Remove ticks
    plt.xticks([])
    plt.yticks([])
    
    plt.tight_layout()
    plt.savefig('/home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_esm_emb/pca_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("PCA plot saved as 'pca_visualization.png' in /analyses/qsar_papyrus_esm_emb/")

def create_umap_plot(embeddings, emb_df):
    """
    Create UMAP visualization plot.
    """
    print("Creating UMAP plot...")
    
    # Initialize UMAP
    umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    
    # Fit UMAP to the embeddings and transform the data
    umap_results = umap_reducer.fit_transform(embeddings)
    
    # Create DataFrame with UMAP results
    df_umap = pd.DataFrame(data=umap_results, columns=['UMAP1', 'UMAP2'])
    df_umap['c3'] = emb_df['c3'].values
    
    # Create scatter plot
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df_umap, x='UMAP1', y='UMAP2', hue='c3', 
                    markers='o', palette='tab20')
    
    # Customize plot
    plt.legend(bbox_to_anchor=(1,1))
    plt.xlabel('UMAP1', fontweight='bold', fontsize=20)
    plt.ylabel('UMAP2', fontweight='bold', fontsize=20)
    
    # Style spines
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
    
    # Remove ticks
    plt.xticks([])
    plt.yticks([])
    
    plt.tight_layout()
    plt.savefig('/home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_esm_emb/umap_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("UMAP plot saved as 'umap_visualization.png' in /analyses/qsar_papyrus_esm_emb/")

def main():
    """
    Main function to run the visualization pipeline.
    """
    print("Starting protein embedding visualization...")
    
    # Load data
    target_df, embeddings, papyrus_targets = load_required_data()
    
    # Create embedding DataFrame
    emb_df, reduced_embeddings = create_embedding_dataframe(embeddings, target_df, papyrus_targets)
    
    # Create visualizations
    create_pca_plot(emb_df)
    create_umap_plot(embeddings, emb_df)
    
    print("Visualization complete!")
    print("Files created in /analyses/qsar_papyrus_esm_emb/:")
    print("- pca_visualization.png")
    print("- umap_visualization.png")

if __name__ == "__main__":
    main() 