#!/usr/bin/env python3
"""
Single ESM C Embedding Function

This module provides a clean function to compute ESM C embeddings for a single protein sequence.
ESM C is a more efficient replacement for ESM2 with better performance and lower memory requirements.

Based on the ESM C API from https://pypi.org/project/esm/

REQUIREMENTS:
- Python 3.10 or higher
- ESM C package (pip install esm)
- The 'esmc' conda environment must be activated before running this script
  To activate: conda activate esmc

Author: Generated for ESM C embeddings
Date: 2024
"""

import torch
import numpy as np
import time
import warnings
from typing import Union, Dict, List, Optional, Tuple

# Suppress common warnings
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig
except ImportError:
    raise ImportError(
        "ESM C is not installed. Please install it with: pip install esm\n"
        "Make sure you are in the 'esmc' conda environment: conda activate esmc"
    )


def get_single_esmc_embedding(
    protein_sequence: str,
    model_name: str = "esmc_300m",
    device: Optional[str] = None,
    return_per_residue: bool = False,
    verbose: bool = True,
    use_flash_attn: bool = True
) -> Union[np.ndarray, Dict]:
    """
    Compute ESM C embedding for a single protein sequence.
    
    Args:
        protein_sequence (str): Protein sequence in single-letter amino acid code
        model_name (str): ESM C model name. Options: "esmc_300m", "esmc_600m"
        device (str, optional): Device to run on ("cuda", "cpu", or None for auto)
        return_per_residue (bool): If True, return per-residue embeddings
        verbose (bool): Print progress information
        use_flash_attn (bool): Use Flash Attention for faster computation
        
    Returns:
        Union[np.ndarray, Dict]: 
            - If return_per_residue=False: 1D array of sequence-level embedding
            - If return_per_residue=True: Dict with 'sequence_embedding' and 'per_residue_embeddings'
    """
    
    if verbose:
        print(f"Computing ESM C embedding for sequence of length {len(protein_sequence)}")
        print(f"Using model: {model_name}")
    
    # Input validation
    if not protein_sequence or not isinstance(protein_sequence, str):
        raise ValueError("Protein sequence must be a non-empty string")
    
    # Validate amino acid sequence
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    if not all(aa in valid_aa for aa in protein_sequence.upper()):
        raise ValueError("Sequence contains invalid amino acid characters")
    
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if verbose:
        print(f"Using device: {device}")
    
    start_time = time.time()
    
    try:
        # Load ESM C model
        if verbose:
            print("Loading ESM C model...")
        
        client = ESMC.from_pretrained(model_name).to(device)
        
        # Create ESMProtein object
        protein = ESMProtein(sequence=protein_sequence.upper())
        
        # Encode the protein
        if verbose:
            print("Encoding protein sequence...")
        
        protein_tensor = client.encode(protein)
        
        # Get logits and embeddings
        if verbose:
            print("Computing embeddings...")
        
        logits_config = LogitsConfig(
            sequence=True, 
            return_embeddings=True
        )
        
        # Disable flash attention if requested
        if not use_flash_attn:
            # Note: This would need to be passed to the model initialization
            # For now, we'll proceed with the default behavior
            pass
        
        logits_output = client.logits(protein_tensor, logits_config)
        
        # Extract embeddings
        embeddings = logits_output.embeddings
        
        if verbose:
            print(f"Embedding shape: {embeddings.shape}")
        
        # Convert to numpy
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        
        # Process embeddings based on return type
        if return_per_residue:
            # Return both sequence-level and per-residue embeddings
            sequence_embedding = np.mean(embeddings, axis=0)  # Average pooling for sequence-level
            
            result = {
                'sequence_embedding': sequence_embedding,
                'per_residue_embeddings': embeddings,
                'sequence_length': len(protein_sequence),
                'model_name': model_name,
                'embedding_dimension': embeddings.shape[-1],
                'computation_time': time.time() - start_time
            }
        else:
            # Return only sequence-level embedding (average pooled)
            sequence_embedding = np.mean(embeddings, axis=0)
            result = sequence_embedding
        
        if verbose:
            print(f"Computation completed in {time.time() - start_time:.2f} seconds")
        
        return result
        
    except Exception as e:
        raise RuntimeError(f"Error computing ESM C embedding: {str(e)}")


def batch_esmc_embeddings(
    protein_sequences: List[str],
    model_name: str = "esmc_300m",
    device: Optional[str] = None,
    batch_size: int = 1,
    verbose: bool = True,
    use_flash_attn: bool = True
) -> List[np.ndarray]:
    """
    Compute ESM C embeddings for multiple protein sequences.
    
    Args:
        protein_sequences (List[str]): List of protein sequences
        model_name (str): ESM C model name
        device (str, optional): Device to run on
        batch_size (int): Batch size for processing (currently limited to 1 for ESM C)
        verbose (bool): Print progress information
        use_flash_attn (bool): Use Flash Attention
        
    Returns:
        List[np.ndarray]: List of sequence-level embeddings
    """
    
    if verbose:
        print(f"Processing {len(protein_sequences)} sequences with ESM C")
    
    # ESM C currently processes one sequence at a time
    if batch_size != 1:
        if verbose:
            print("Warning: ESM C currently supports batch_size=1 only. Processing sequences individually.")
        batch_size = 1
    
    embeddings = []
    
    for i, sequence in enumerate(protein_sequences):
        if verbose:
            print(f"Processing sequence {i+1}/{len(protein_sequences)}")
        
        embedding = get_single_esmc_embedding(
            protein_sequence=sequence,
            model_name=model_name,
            device=device,
            return_per_residue=False,
            verbose=False,
            use_flash_attn=use_flash_attn
        )
        
        embeddings.append(embedding)
    
    return embeddings


def compare_esm2_vs_esmc(
    protein_sequence: str,
    esm2_model: str = "esm2_t33_650M_UR50D",
    esmc_model: str = "esmc_300m",
    device: Optional[str] = None,
    verbose: bool = True
) -> Dict:
    """
    Compare ESM2 and ESM C embeddings for the same sequence.
    
    Args:
        protein_sequence (str): Protein sequence to compare
        esm2_model (str): ESM2 model name
        esmc_model (str): ESM C model name
        device (str, optional): Device to run on
        verbose (bool): Print comparison results
        
    Returns:
        Dict: Comparison results including embeddings and performance metrics
    """
    
    if verbose:
        print("Comparing ESM2 vs ESM C embeddings...")
    
    # Import ESM2 function from the renamed file
    try:
        import sys
        import os
        sys.path.append(os.path.dirname(__file__))
        from single_esm2_embeddings import get_single_esm_embedding
    except ImportError:
        raise ImportError("Could not import ESM2 embedding function. Make sure single_esm2_embeddings.py exists.")
    
    # Compute ESM2 embedding
    if verbose:
        print("Computing ESM2 embedding...")
    
    esm2_start = time.time()
    esm2_result = get_single_esm_embedding(
        protein_sequence=protein_sequence,
        model_name=esm2_model,
        device=device,
        return_per_residue=True,
        verbose=False
    )
    esm2_time = time.time() - esm2_start
    
    # Compute ESM C embedding
    if verbose:
        print("Computing ESM C embedding...")
    
    esmc_start = time.time()
    esmc_result = get_single_esmc_embedding(
        protein_sequence=protein_sequence,
        model_name=esmc_model,
        device=device,
        return_per_residue=True,
        verbose=False
    )
    esmc_time = time.time() - esmc_start
    
    # Calculate similarity
    esm2_embedding = esm2_result['sequence_embedding']
    esmc_embedding = esmc_result['sequence_embedding']
    
    # Cosine similarity
    cosine_sim = np.dot(esm2_embedding, esmc_embedding) / (
        np.linalg.norm(esm2_embedding) * np.linalg.norm(esmc_embedding)
    )
    
    # Euclidean distance
    euclidean_dist = np.linalg.norm(esm2_embedding - esmc_embedding)
    
    comparison = {
        'sequence': protein_sequence,
        'sequence_length': len(protein_sequence),
        'esm2_model': esm2_model,
        'esmc_model': esmc_model,
        'esm2_embedding': esm2_embedding,
        'esmc_embedding': esmc_embedding,
        'esm2_embedding_dim': len(esm2_embedding),
        'esmc_embedding_dim': len(esmc_embedding),
        'cosine_similarity': cosine_sim,
        'euclidean_distance': euclidean_dist,
        'esm2_computation_time': esm2_time,
        'esmc_computation_time': esmc_time,
        'speedup_factor': esm2_time / esmc_time if esmc_time > 0 else float('inf')
    }
    
    if verbose:
        print(f"\nComparison Results:")
        print(f"Sequence length: {len(protein_sequence)}")
        print(f"ESM2 embedding dim: {len(esm2_embedding)}")
        print(f"ESM C embedding dim: {len(esmc_embedding)}")
        print(f"Cosine similarity: {cosine_sim:.4f}")
        print(f"Euclidean distance: {euclidean_dist:.4f}")
        print(f"ESM2 time: {esm2_time:.2f}s")
        print(f"ESM C time: {esmc_time:.2f}s")
        print(f"Speedup: {comparison['speedup_factor']:.2f}x")
    
    return comparison


if __name__ == "__main__":
    # Example usage
    print("ESM C Embedding Example")
    print("=" * 50)
    
    # Example protein sequence
    sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    
    try:
        # Single embedding
        print("\n1. Single ESM C Embedding:")
        embedding = get_single_esmc_embedding(sequence, verbose=True)
        print(f"Embedding shape: {embedding.shape}")
        print(f"First 10 values: {embedding[:10]}")
        
        # Per-residue embedding
        print("\n2. Per-residue ESM C Embedding:")
        result = get_single_esmc_embedding(sequence, return_per_residue=True, verbose=True)
        print(f"Sequence embedding shape: {result['sequence_embedding'].shape}")
        print(f"Per-residue embeddings shape: {result['per_residue_embeddings'].shape}")
        
        # Batch processing
        print("\n3. Batch ESM C Embeddings:")
        sequences = [sequence, "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"[:30]]
        batch_embeddings = batch_esmc_embeddings(sequences, verbose=True)
        print(f"Processed {len(batch_embeddings)} sequences")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure ESM C is properly installed: pip install esm")