import os
import pandas as pd
import re

# Switch: Set to True to match all possible UniProt IDs in target_id, False to only use the first part
match_all_uniprot_ids = False

# Path to the CSV files
csv_path = 'primary_data/approved_drugs/02_processed/papyrus/papyrus_data.csv'
avoidome_path = 'primary_data/avoidome_prot_list.csv'

# explore csv papyrus data
# head -5 /home/serramelendezcsm/RA/Avoidome/primary_data/approved_drugs/02_processed/papyrus/papyrus_data.csv

def extract_uniprot_ids(target_id):
    # UniProt accessions are usually 6 characters, starting with P, Q, or O
    return [part for part in str(target_id).split('_') if re.match(r'^[OPQ][0-9A-Z]{5}$', part)]

def main():
    # Print file size
    file_size = os.path.getsize(csv_path)
    print(f"File size: {file_size / (1024 * 1024):.2f} MB")
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Data shape: {df.shape}")
    print("Column names:")
    print(df.columns.tolist())
    print("First 5 rows:")
    print(df.head())

    # Load avoidome protein list
    avoidome_df = pd.read_csv(avoidome_path)
    avoidome_uniprot_ids = set(avoidome_df['UniProt ID'].dropna().unique())
    print(f"\n=== AVOIDOME DATASET ANALYSIS ===")
    print(f"Loaded {len(avoidome_uniprot_ids)} unique avoidome UniProt IDs.")
    print(f"Total avoidome entries: {len(avoidome_df)}")
    print(f"Sample avoidome UniProt IDs: {list(avoidome_uniprot_ids)[:10]}")

    # Analyze Papyrus dataset targets
    print(f"\n=== PAPYRUS DATASET TARGET ANALYSIS ===")
    if match_all_uniprot_ids:
        # Extract all possible UniProt IDs from target_id
        df['uniprot_ids'] = df['target_id'].apply(extract_uniprot_ids)
        # Check for rows with more than one UniProt ID
        multi_id_rows = df[df['uniprot_ids'].apply(lambda ids: len(ids) > 1)]
        print(f"Rows with more than one UniProt ID in 'target_id': {multi_id_rows.shape[0]}")
        if not multi_id_rows.empty:
            print("Examples with multiple UniProt IDs:")
            print(multi_id_rows[['target_id', 'uniprot_ids']].head())
        
        # Get all unique UniProt IDs from Papyrus
        all_papyrus_uniprot_ids = set()
        for ids in df['uniprot_ids']:
            all_papyrus_uniprot_ids.update(ids)
        
        print(f"Total unique UniProt IDs in Papyrus dataset: {len(all_papyrus_uniprot_ids)}")
        print(f"Sample Papyrus UniProt IDs: {list(all_papyrus_uniprot_ids)[:10]}")
        
        # Filter Papyrus activities to those matching any avoidome UniProt ID
        mask = df['uniprot_ids'].apply(lambda ids: any(uid in avoidome_uniprot_ids for uid in ids))
    else:
        # Only use the first part as UniProt ID
        df['uniprot_id'] = df['target_id'].str.split('_').str[0]
        papyrus_uniprot_ids = set(df['uniprot_id'].dropna().unique())
        print(f"Total unique UniProt IDs in Papyrus dataset: {len(papyrus_uniprot_ids)}")
        print(f"Sample Papyrus UniProt IDs: {list(papyrus_uniprot_ids)[:10]}")
        mask = df['uniprot_id'].isin(avoidome_uniprot_ids)
    
    filtered_df = df[mask]
    print(f"\n=== OVERLAP ANALYSIS ===")
    print(f"Number of Papyrus activities matching avoidome proteins: {filtered_df.shape[0]}")
    print(f"Percentage of Papyrus activities matching avoidome proteins: {(filtered_df.shape[0] / df.shape[0]) * 100:.2f}%")
    
    # Calculate target overlap statistics
    if match_all_uniprot_ids:
        # Get unique targets that overlap
        overlapping_targets = set()
        for ids in filtered_df['uniprot_ids']:
            overlapping_targets.update([uid for uid in ids if uid in avoidome_uniprot_ids])
    else:
        overlapping_targets = set(filtered_df['uniprot_id'].unique())
    
    print(f"Number of unique targets that overlap between datasets: {len(overlapping_targets)}")
    print(f"Percentage of avoidome targets found in Papyrus: {(len(overlapping_targets) / len(avoidome_uniprot_ids)) * 100:.2f}%")
    
    # Show overlapping targets
    print(f"\nOverlapping targets: {sorted(list(overlapping_targets))}")
    
    # Show targets in avoidome but not in Papyrus
    if match_all_uniprot_ids:
        papyrus_targets = all_papyrus_uniprot_ids
    else:
        papyrus_targets = papyrus_uniprot_ids
    
    avoidome_only = avoidome_uniprot_ids - papyrus_targets
    papyrus_only = papyrus_targets - avoidome_uniprot_ids
    
    print(f"\nTargets in avoidome but not in Papyrus ({len(avoidome_only)}): {sorted(list(avoidome_only))}")
    print(f"Targets in Papyrus but not in avoidome ({len(papyrus_only)}): {sorted(list(papyrus_only))[:20]}...")  # Show first 20
    
    print(f"\nFirst 5 matching rows:")
    print(filtered_df.head())

if __name__ == "__main__":
    main() 