"""
Descriptor Cache Management Script
=================================

This script provides utilities to manage the molecular descriptor cache,
including viewing cache status, clearing cache, and pre-calculating
descriptors for all targets.

Usage:
    python manage_cache.py --status          # Show cache status
    python manage_cache.py --clear           # Clear all cached descriptors
    python manage_cache.py --pre-calculate   # Pre-calculate descriptors for all targets
    python manage_cache.py --target P05177   # Show cache info for specific target

Author: QSAR Modeling System
Date: 2024
"""

import os
import joblib
import pandas as pd
import argparse
from datetime import datetime
import sys

# Add the parent directory to the path to import the QSAR modeling class
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from qsar_modeling import QSARModelBuilder

def get_cache_info(cache_dir):
    """Get information about cached descriptors"""
    if not os.path.exists(cache_dir):
        return []
    
    cache_info = []
    
    # Check for global cache
    global_cache_file = os.path.join(cache_dir, 'global_molecule_descriptors.pkl')
    if os.path.exists(global_cache_file):
        try:
            global_cache = joblib.load(global_cache_file)
            cache_info.append({
                'target_id': 'GLOBAL',
                'file_size_mb': os.path.getsize(global_cache_file) / (1024 * 1024),
                'molecule_count': len(global_cache),
                'feature_count': len(list(global_cache.values())[0]) if global_cache else 0,
                'timestamp': 'Global molecule cache',
                'is_global': True
            })
        except Exception as e:
            print(f"Error reading global cache: {str(e)}")
    
    # Check for legacy target-specific caches
    for file in os.listdir(cache_dir):
        if file.endswith('_descriptors.pkl') and not file.startswith('global'):
            target_id = file.replace('_descriptors.pkl', '')
            cache_file = os.path.join(cache_dir, file)
            
            try:
                cached_data = joblib.load(cache_file)
                cache_info.append({
                    'target_id': target_id,
                    'file_size_mb': os.path.getsize(cache_file) / (1024 * 1024),
                    'molecule_count': len(cached_data['descriptors']),
                    'feature_count': cached_data['feature_count'],
                    'timestamp': cached_data['timestamp'],
                    'is_global': False
                })
            except Exception as e:
                print(f"Error reading cache file {file}: {str(e)}")
    
    return cache_info

def show_cache_status(cache_dir):
    """Display cache status"""
    print("="*60)
    print("DESCRIPTOR CACHE STATUS")
    print("="*60)
    
    cache_info = get_cache_info(cache_dir)
    
    if not cache_info:
        print("No cached descriptors found.")
        return
    
    print(f"Found {len(cache_info)} cached targets:")
    print()
    
    total_size = 0
    total_molecules = 0
    
    # Show global cache first
    global_cache_info = [info for info in cache_info if info.get('is_global', False)]
    target_cache_info = [info for info in cache_info if not info.get('is_global', False)]
    
    if global_cache_info:
        info = global_cache_info[0]
        print("GLOBAL MOLECULE CACHE:")
        print(f"  Unique Molecules: {info['molecule_count']:,}")
        print(f"  Features per molecule: {info['feature_count']}")
        print(f"  File size: {info['file_size_mb']:.2f} MB")
        print(f"  Status: {info['timestamp']}")
        print()
        total_size += info['file_size_mb']
        total_molecules += info['molecule_count']
    
        if target_cache_info:
            print("TARGET-SPECIFIC CACHES (Legacy):")
            for info in target_cache_info:
                print(f"  Target: {info['target_id']}")
                print(f"    Molecules: {info['molecule_count']:,}")
                print(f"    Features: {info['feature_count']}")
                print(f"    File size: {info['file_size_mb']:.2f} MB")
                print(f"    Cached: {info['timestamp']}")
                print()
                
                total_size += info['file_size_mb']
                total_molecules += info['molecule_count']
    
    print(f"Total cached molecules: {total_molecules:,}")
    print(f"Total cache size: {total_size:.2f} MB")
    print("="*60)

def clear_cache(cache_dir):
    """Clear all cached descriptors"""
    if not os.path.exists(cache_dir):
        print("No cache directory found.")
        return
    
    cache_files = [f for f in os.listdir(cache_dir) if f.endswith('_descriptors.pkl')]
    
    if not cache_files:
        print("No cached descriptors found.")
        return
    
    print(f"Found {len(cache_files)} cached files.")
    response = input("Are you sure you want to delete all cached descriptors? (y/N): ")
    
    if response.lower() == 'y':
        for file in cache_files:
            file_path = os.path.join(cache_dir, file)
            os.remove(file_path)
            print(f"Deleted: {file}")
        print("Cache cleared successfully.")
    else:
        print("Cache clearing cancelled.")

def pre_calculate_descriptors(data_path, output_dir, min_samples=50):
    """Pre-calculate descriptors for all unique molecules (global cache)"""
    print("="*60)
    print("PRE-CALCULATING DESCRIPTORS FOR ALL UNIQUE MOLECULES")
    print("="*60)
    
    # Initialize QSAR builder
    qsar_builder = QSARModelBuilder(data_path, output_dir)
    qsar_builder.load_data()
    
    # Get all unique SMILES across all targets
    all_smiles = qsar_builder.df['canonical_smiles'].unique()
    print(f"Found {len(all_smiles):,} unique molecules across all targets")
    print()
    
    # Check existing global cache
    global_cache_info = qsar_builder.get_global_cache_info()
    if global_cache_info:
        print(f"Global cache already contains {global_cache_info['molecule_count']:,} molecules")
        print(f"Cache size: {global_cache_info['file_size_mb']:.2f} MB")
        print()
        
        # Calculate how many new molecules need processing
        new_molecules = len(all_smiles) - global_cache_info['molecule_count']
        if new_molecules <= 0:
            print("All molecules already cached! No new calculations needed.")
            show_cache_status(os.path.join(output_dir, 'descriptor_cache'))
            return
        else:
            print(f"Need to calculate descriptors for {new_molecules:,} new molecules")
            print()
    
    # Calculate descriptors for all unique molecules
    print("Calculating descriptors for all unique molecules...")
    print("This will create a global cache that can be reused across all targets.")
    print()
    
    try:
        # Calculate descriptors (this will automatically use and update global cache)
        descriptors, valid_indices = qsar_builder.calculate_molecular_descriptors(
            all_smiles.tolist(), force_recalculate=False
        )
        
        print(f"✓ Successfully processed {len(descriptors):,} molecules")
        
    except Exception as e:
        print(f"✗ Error during global calculation: {str(e)}")
        return
    
    print()
    print("Global molecule cache creation completed!")
    print("Now all targets can use the same cached descriptors.")
    show_cache_status(os.path.join(output_dir, 'descriptor_cache'))

def show_target_cache(cache_dir, target_id):
    """Show detailed cache information for a specific target"""
    cache_file = os.path.join(cache_dir, f'{target_id}_descriptors.pkl')
    
    if not os.path.exists(cache_file):
        print(f"No cached descriptors found for target {target_id}")
        return
    
    try:
        cached_data = joblib.load(cache_file)
        
        print(f"Cache information for target {target_id}:")
        print(f"  Molecules: {len(cached_data['descriptors']):,}")
        print(f"  Features: {cached_data['feature_count']}")
        print(f"  File size: {os.path.getsize(cache_file) / (1024 * 1024):.2f} MB")
        print(f"  Cached: {cached_data['timestamp']}")
        
        # Show sample descriptors
        if cached_data['descriptors']:
            sample_desc = cached_data['descriptors'][0]
            print(f"  Sample descriptors:")
            for key, value in list(sample_desc.items())[:10]:  # Show first 10
                print(f"    {key}: {value}")
            print(f"    ... and {len(sample_desc) - 10} more features")
        
    except Exception as e:
        print(f"Error reading cache file: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Descriptor Cache Management')
    parser.add_argument('--status', action='store_true', help='Show cache status')
    parser.add_argument('--clear', action='store_true', help='Clear all cached descriptors')
    parser.add_argument('--pre-calculate', action='store_true', help='Pre-calculate descriptors for all targets')
    parser.add_argument('--target', type=str, help='Show cache info for specific target')
    parser.add_argument('--data-path', type=str, 
                       default="/home/serramelendezcsm/RA/Avoidome/processed_data/avoidome_bioactivity_profile.csv",
                       help='Path to bioactivity data')
    parser.add_argument('--output-dir', type=str,
                       default="/home/serramelendezcsm/RA/Avoidome/analyses/qsar_avoidome",
                       help='Output directory')
    parser.add_argument('--min-samples', type=int, default=50,
                       help='Minimum samples required per target')
    
    args = parser.parse_args()
    
    cache_dir = os.path.join(args.output_dir, 'descriptor_cache')
    
    if args.status:
        show_cache_status(cache_dir)
    elif args.clear:
        clear_cache(cache_dir)
    elif args.pre_calculate:
        pre_calculate_descriptors(args.data_path, args.output_dir, args.min_samples)
    elif args.target:
        show_target_cache(cache_dir, args.target)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 