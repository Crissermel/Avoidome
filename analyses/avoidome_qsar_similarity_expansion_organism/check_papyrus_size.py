#!/usr/bin/env python3
"""
Check Papyrus database size
"""

import sys
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def check_papyrus_size():
    """Check the size of Papyrus database"""
    
    try:
        from papyrus_scripts import PapyrusDataset
        
        print("Loading Papyrus dataset...")
        papyrus_data = PapyrusDataset(version='05.7', plusplus=False)
        
        print("Getting protein data...")
        protein_data = papyrus_data.proteins()
        
        if hasattr(protein_data, 'to_dataframe'):
            protein_df = protein_data.to_dataframe()
        else:
            protein_df = protein_data
        
        print(f"Total proteins in Papyrus: {len(protein_df)}")
        print(f"Proteins with sequences: {len(protein_df.dropna(subset=['Sequence']))}")
        
        # Show first few proteins
        print("\nFirst 5 proteins:")
        for i, (_, row) in enumerate(protein_df.head().iterrows()):
            print(f"  {i+1}. {row['target_id']}: {len(str(row['Sequence']))} aa")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_papyrus_size()





