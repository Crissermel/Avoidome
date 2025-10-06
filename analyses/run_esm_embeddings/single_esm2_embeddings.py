#!/usr/bin/env python3
"""
Single ESM Embedding Function

This module provides a clean function to compute ESM embeddings for a single protein sequence.
Based on the ESM embeddings analysis script in this directory.


Date: 2025
"""

import torch
import esm
import numpy as np
import time
import warnings
from typing import Union, Dict, List, Optional, Tuple

# Suppress common warnings from ESM
warnings.filterwarnings("ignore", category=UserWarning, module='esm.pretrained')


def get_single_esm_embedding(
    protein_sequence: str,
    model_name: str = "esm2_t33_650M_UR50D",
    repr_layer: Optional[int] = None,
    include_bos_eos: bool = False,
    truncation_seq_length: Optional[int] = None,
    device: Optional[str] = None,
    return_per_residue: bool = False,
    verbose: bool = True
) -> Union[np.ndarray, Dict[str, Union[np.ndarray, str, int]]]:
    """
    Compute ESM embedding for a single protein sequence.

    Args:
        protein_sequence (str): Amino acid sequence in single-letter code format
        model_name (str): ESM model to use. Options:
            - "esm2_t6_8M_UR50D" (8M parameters, 6 layers)
            - "esm2_t12_35M_UR50D" (35M parameters, 12 layers) 
            - "esm2_t30_150M_UR50D" (150M parameters, 30 layers)
            - "esm2_t33_650M_UR50D" (650M parameters, 33 layers) - DEFAULT
            - "esm1b_t33_650M_UR50S" (ESM-1b, 650M parameters, 33 layers)
        repr_layer (int, optional): Layer to extract embeddings from. 
                                   If None, uses the last layer.
        include_bos_eos (bool): Whether to include BOS/EOS tokens in per-residue embeddings
        truncation_seq_length (int, optional): Maximum sequence length. Longer sequences will be truncated
        device (str, optional): "cuda" for GPU, "cpu" for CPU. If None, auto-detects
        return_per_residue (bool): If True, returns both sequence and per-residue embeddings
        verbose (bool): Whether to print progress information

    Returns:
        If return_per_residue=False:
            np.ndarray: Sequence-level embedding (1280-dimensional vector)
        
        If return_per_residue=True:
            dict: Dictionary containing:
                - 'sequence_embedding': np.ndarray (1280,)
                - 'per_residue_embedding': np.ndarray (seq_len, 1280)
                - 'model_name': str
                - 'representation_layer': int
                - 'sequence_length': int
                - 'processing_time': float

    Raises:
        ValueError: If protein_sequence is empty or invalid
        RuntimeError: If model loading or processing fails

    Example:
        >>> sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
        >>> embedding = get_single_esm_embedding(sequence)
        >>> print(f"Embedding shape: {embedding.shape}")  # (1280,)
        
        >>> # Get both sequence and per-residue embeddings
        >>> result = get_single_esm_embedding(sequence, return_per_residue=True)
        >>> print(f"Sequence embedding: {result['sequence_embedding'].shape}")
        >>> print(f"Per-residue embedding: {result['per_residue_embedding'].shape}")
    """
    
    # Input validation
    if not protein_sequence or not isinstance(protein_sequence, str):
        raise ValueError("protein_sequence must be a non-empty string")
    
    # Clean sequence (remove whitespace, convert to uppercase)
    protein_sequence = protein_sequence.strip().upper()
    if not protein_sequence:
        raise ValueError("protein_sequence cannot be empty after cleaning")
    
    # Validate amino acid characters
    valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
    if not all(aa in valid_aa for aa in protein_sequence):
        invalid_chars = set(protein_sequence) - valid_aa
        raise ValueError(f"Invalid amino acid characters found: {invalid_chars}")
    
    start_time = time.time()
    
    # Device selection
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if verbose:
        print(f"Computing ESM embedding for sequence of length {len(protein_sequence)}")
        print(f"Using device: {device}")
        print(f"Model: {model_name}")
    
    try:
        # Load ESM model
        if verbose:
            print("Loading ESM model...")
        
        model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        model = model.to(device)
        model.eval()
        
        # Set representation layer
        if repr_layer is None:
            repr_layer = model.num_layers
        
        if verbose:
            print(f"Extracting embeddings from layer: {repr_layer}")
            print(f"Model has {model.num_layers} layers and embedding dimension {model.embed_dim}")
        
        # Prepare sequence for processing
        if truncation_seq_length is not None and len(protein_sequence) > truncation_seq_length:
            if verbose:
                print(f"Truncating sequence from {len(protein_sequence)} to {truncation_seq_length} residues")
            protein_sequence = protein_sequence[:truncation_seq_length]
        
        # Convert to batch format
        batch_converter = alphabet.get_batch_converter()
        data_for_batching = [("protein_1", protein_sequence)]
        
        # Tokenize sequence
        batch_labels, batch_strs, batch_tokens = batch_converter(data_for_batching)
        batch_tokens = batch_tokens.to(device)
        
        # Extract embeddings
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)
            token_representations = results["representations"][repr_layer]
        
        # Process results
        protein_seq_str = batch_strs[0]
        token_repr = token_representations[0]  # Shape: (seq_len_with_bos_eos, embed_dim)
        
        # Per-residue embeddings
        if include_bos_eos:
            per_residue_emb = token_repr.cpu().numpy()
        else:
            # Exclude BOS and EOS tokens
            per_residue_emb = token_repr[1:len(protein_seq_str) + 1].cpu().numpy()
        
        # Sequence-level embedding (mean pooling over residues)
        sequence_emb = token_repr[1:len(protein_seq_str) + 1].mean(0).cpu().numpy()
        
        processing_time = time.time() - start_time
        
        if verbose:
            print(f"Processing completed in {processing_time:.2f} seconds")
            print(f"Sequence embedding shape: {sequence_emb.shape}")
            if return_per_residue:
                print(f"Per-residue embedding shape: {per_residue_emb.shape}")
        
        # Return results
        if return_per_residue:
            return {
                'sequence_embedding': sequence_emb,
                'per_residue_embedding': per_residue_emb,
                'model_name': model_name,
                'representation_layer': repr_layer,
                'sequence_length': len(protein_sequence),
                'processing_time': processing_time
            }
        else:
            return sequence_emb
            
    except Exception as e:
        error_msg = f"Error computing ESM embedding: {str(e)}"
        if verbose:
            print(error_msg)
        raise RuntimeError(error_msg) from e


def batch_esm_embeddings(
    protein_sequences: List[str],
    model_name: str = "esm2_t33_650M_UR50D",
    repr_layer: Optional[int] = None,
    include_bos_eos: bool = False,
    truncation_seq_length: Optional[int] = None,
    device: Optional[str] = None,
    return_per_residue: bool = False,
    verbose: bool = True
) -> Union[List[np.ndarray], List[Dict]]:
    """
    Compute ESM embeddings for multiple protein sequences in batch.
    
    Args:
        protein_sequences (List[str]): List of amino acid sequences
        model_name (str): ESM model to use
        repr_layer (int, optional): Layer to extract embeddings from
        include_bos_eos (bool): Whether to include BOS/EOS tokens
        truncation_seq_length (int, optional): Maximum sequence length
        device (str, optional): Device to use
        return_per_residue (bool): Whether to return per-residue embeddings
        verbose (bool): Whether to print progress
        
    Returns:
        List of embeddings (one per input sequence)
    """
    if not protein_sequences:
        return []
    
    if isinstance(protein_sequences, str):
        protein_sequences = [protein_sequences]
    
    # Device selection
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if verbose:
        print(f"Processing {len(protein_sequences)} sequences using {model_name}")
        print(f"Using device: {device}")
    
    try:
        # Load model once
        model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        model = model.to(device)
        model.eval()
        
        if repr_layer is None:
            repr_layer = model.num_layers
        
        batch_converter = alphabet.get_batch_converter()
        
        # Prepare data
        data_for_batching = []
        for i, seq in enumerate(protein_sequences):
            if truncation_seq_length is not None and len(seq) > truncation_seq_length:
                seq = seq[:truncation_seq_length]
            data_for_batching.append((f"protein_{i+1}", seq))
        
        # Process in batch
        batch_labels, batch_strs, batch_tokens = batch_converter(data_for_batching)
        batch_tokens = batch_tokens.to(device)
        
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)
            token_representations = results["representations"][repr_layer]
        
        # Process results
        embeddings = []
        for i, protein_seq_str in enumerate(batch_strs):
            token_repr = token_representations[i]
            
            # Per-residue embeddings
            if include_bos_eos:
                per_residue_emb = token_repr.cpu().numpy()
            else:
                per_residue_emb = token_repr[1:len(protein_seq_str) + 1].cpu().numpy()
            
            # Sequence-level embedding
            sequence_emb = token_repr[1:len(protein_seq_str) + 1].mean(0).cpu().numpy()
            
            if return_per_residue:
                embeddings.append({
                    'sequence_embedding': sequence_emb,
                    'per_residue_embedding': per_residue_emb,
                    'model_name': model_name,
                    'representation_layer': repr_layer,
                    'sequence_length': len(protein_seq_str)
                })
            else:
                embeddings.append(sequence_emb)
        
        if verbose:
            print(f"Successfully processed {len(embeddings)} sequences")
        
        return embeddings
        
    except Exception as e:
        error_msg = f"Error in batch processing: {str(e)}"
        if verbose:
            print(error_msg)
        raise RuntimeError(error_msg) from e


def main():
    """
    Example usage of the ESM embedding functions.
    """
    # Example protein sequence (CYP1A2)
    example_sequence = "MALSQSVPFSATELLLASAIFCLVFWVLKGLRPRVPKGLKSPPEPWGWPLLGHVLTLGKNPHLALSRMSQRYGDVLQIRIGSTPVLVLSRLDTIRQALVRQGDDFKGRPDLYTSTLITDGQSLTFSTDSGPVWAARRRLAQNALNTFSIASDPASSSSCYLEEHVSKEAKALISRLQELMAGPGHFDPYNQVVVSVANVIGAMCFGQHFPESSDEMLSLVKNTHEFVETASSGNPLDFFPILRYLPNPALQRFKAFNQRFLWFLQKTVQEHYQDFDKNSVRDITGALFKHSKKGPRASGNLIPQEKIVNLVNDIFGAGFDTVTTAISWSLMYLVTKPEIQRKIQKELDTVIGRERRPRLSDRPQLPYLEAFILETFRHSSFLPFTIPHSTTRDTTLNGFYIPKKCCVFVNQWQVNHDPELWEDPSEFRPERFLTADGTAINKPLSEKMMLFGMGKRRCIGEVLAKWEIFLFLAILLQQLEFSVPPGVKVDLTPIYGLTMKHARCEHVQARLRFSIN"
    
    print("=== Single ESM Embedding Example ===")
    
    # Single sequence embedding
    embedding = get_single_esm_embedding(example_sequence)
    print(f"Sequence embedding shape: {embedding.shape}")
    print(f"First 10 values: {embedding[:10]}")
    
    print("\n=== Per-residue Embedding Example ===")
    
    # Get both sequence and per-residue embeddings
    result = get_single_esm_embedding(
        example_sequence, 
        return_per_residue=True,
        model_name="esm2_t33_650M_UR50D"
    )
    
    print(f"Sequence embedding shape: {result['sequence_embedding'].shape}")
    print(f"Per-residue embedding shape: {result['per_residue_embedding'].shape}")
    print(f"Model used: {result['model_name']}")
    print(f"Processing time: {result['processing_time']:.2f} seconds")
    
    print("\n=== Batch Processing Example ===")
    
    # Multiple sequences
    sequences = [
        example_sequence,
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "MELSVLLFLALLTGLLLLLVQRHPNTHDRLPPGPRPLPLLGNLLQMDRRGLLKSFLRFREKYGDVFTVHLGPRPVVMLCGVEAIREALVDKAEAFSGRGKIAMVDPFFRGYGVIFANGNRWKVLRRFSVTTMRDFGMGKRSVEERIQEEAQCLIEELRKSKGALMDPTFLFQSITANIICSIVFGKRFHYQDQEFLKMLNLFYQTFSLISSVFGQLFELFSGFLKYFPGAHRQVYKNLQEINAYIGHSVEKHRETLDPSAPKDLIDTYLLHMEKEKSNAHSEFSHQNLNLNTLSLFFAGTETTSTTLRYGFLLMLKYPHVAERVYREIEQVIGPHRPPELHDRAKMPYTEAVIYEIQRFSDLLPMGVPHIVTQHTSFRGYIIPKDTEVFLILSTALHDPHYFEKPDAFNPDHFLDANGALKKTEAFIPFSLGKRICLGEGIARAELFLFFTTILQNFSMASPVAPEDIDLTPQECGVGKIPPTYQIRFLPR"
    ]
    
    batch_embeddings = batch_esm_embeddings(sequences, verbose=True)
    print(f"Processed {len(batch_embeddings)} sequences")
    for i, emb in enumerate(batch_embeddings):
        print(f"Sequence {i+1} embedding shape: {emb.shape}")


if __name__ == "__main__":
    main()