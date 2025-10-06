import json
import os
import time
import requests
import pandas as pd

# NOTE: Requires: pip install uniprot-id-mapper
try:
    from UniProtMapper import ProtMapper
except ImportError:
    raise ImportError("Please install the uniprot-id-mapper package: pip install uniprot-id-mapper")

json_path = os.path.join(os.path.dirname(__file__), '../primary_data/regex_extracted.json')

RETRY_LIMIT = 3
RETRY_DELAY = 5  # seconds

def extract_unique_proteins(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
        protein_dict = {}  # lower-case: original
        for entry in data:
            proteins = entry.get('proteins', [])
            if isinstance(proteins, list):
                for p in proteins:
                    p_clean = p.strip()
                    p_lower = p_clean.lower()
                    if p_lower not in protein_dict:
                        protein_dict[p_lower] = p_clean
        return list(protein_dict.values()), set(protein_dict.keys())  # originals, lower-case set

def safe_mapper_get(mapper, ids, from_db, to_db):
    for attempt in range(RETRY_LIMIT):
        try:
            result, failed = mapper.get(ids=ids, from_db=from_db, to_db=to_db)
            return result, failed
        except (requests.exceptions.RequestException, Exception) as e:
            print(f"API/network error for {ids} (attempt {attempt+1}/{RETRY_LIMIT}): {e}")
            if attempt < RETRY_LIMIT - 1:
                print(f"Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                # Return empty DataFrame and empty list on failure
                return pd.DataFrame(), []

def load_processed_proteins(mapped_path, unmapped_path):
    processed = set()
    if os.path.exists(mapped_path):
        with open(mapped_path, 'r') as f:
            for line in f:
                if line.strip():
                    processed.add(line.split('\t')[0].strip())
    if os.path.exists(unmapped_path):
        with open(unmapped_path, 'r') as f:
            for line in f:
                if line.strip():
                    processed.add(line.strip())
    return processed

def map_proteins_one_by_one(protein_list, mapped_path, unmapped_path, error_path):
    mapper = ProtMapper()
    mapped = {}
    unmapped = []
    processed = load_processed_proteins(mapped_path, unmapped_path)
    with open(mapped_path, 'a') as mapped_file, open(unmapped_path, 'a') as unmapped_file, open(error_path, 'a') as error_file:
        for p in protein_list:
            if p in processed:
                print(f"Skipping already processed: {p}")
                continue
            try:
                # Try as gene name
                print(f"Searching: {p} (as Gene_Name)")
                result, failed = safe_mapper_get(mapper, [p], "Gene_Name", "UniProtKB")
                print("Result DataFrame:")
                print(result)
                if not result.empty:
                    print(f"Columns: {result.columns}")
                    print(f"First row: {result.iloc[0]}")
                if not result.empty and result.iloc[0].get('Entry'):
                    uniprot_id = result.iloc[0]['Entry']
                    print(f"Result: {uniprot_id}")
                    mapped[p] = uniprot_id
                    mapped_file.write(f"{p}\t{uniprot_id}\n")
                    mapped_file.flush()
                else:
                    # Try as <prot_code>_HUMAN with from_db="UniProtKB"
                    p_human = f"{p}_HUMAN"
                    print(f"Searching: {p_human} (as UniProtKB)")
                    result2, failed2 = safe_mapper_get(mapper, [p_human], "UniProtKB", "UniProtKB")
                    print("Result2 DataFrame:")
                    print(result2)
                    if not result2.empty:
                        print(f"Columns: {result2.columns}")
                        print(f"First row: {result2.iloc[0]}")
                    if not result2.empty and result2.iloc[0].get('Entry'):
                        uniprot_id = result2.iloc[0]['Entry']
                        print(f"Result: {uniprot_id}")
                        mapped[p] = uniprot_id
                        mapped_file.write(f"{p}\t{uniprot_id}\n")
                        mapped_file.flush()
                    else:
                        print("Result: None")
                        unmapped.append(p)
                        unmapped_file.write(p + '\n')
                        unmapped_file.flush()
            except Exception as e:
                print(f"Error mapping {p}: {e}")
                error_file.write(f"{p}\t{str(e)}\n")
                error_file.flush()
    return mapped, unmapped

def main():
    unique_proteins, unique_proteins_lower = extract_unique_proteins(json_path)
    #unique_proteins = ["Bcl-2", "TP53", "EGFR"]
    print(f"Total unique proteins: {len(unique_proteins)}")
    # Save to file for mapping step
    out_path = os.path.join(os.path.dirname(__file__), 'unique_proteins_unprocessed.txt')
    with open(out_path, 'w') as out:
        for p in unique_proteins:
            out.write(p + '\n')
    print(f"\nUnique protein list saved to {out_path}")
    print("\nProceeding to mapping to UniProt using UniProtMapper (case-insensitive, with _HUMAN fallback, one by one, with resume support).\n")

    # --- Mapping step ---
    mapped_path = os.path.join(os.path.dirname(__file__), 'unique_proteins_mapped.txt')
    unmapped_path = os.path.join(os.path.dirname(__file__), 'unique_proteins_unmapped.txt')
    error_path = os.path.join(os.path.dirname(__file__), 'unique_proteins_errors.txt')

    # Do NOT truncate output files, so resume works
    # open(mapped_path, 'w').close()
    # open(unmapped_path, 'w').close()
    # open(error_path, 'w').close()

    mapped, unmapped = map_proteins_one_by_one(unique_proteins, mapped_path, unmapped_path, error_path)

    print(f"\nMapped proteins: {len(mapped)}")
    for p, uniprot_id in mapped.items():
        print(f"{p} -> UniProtKB: {uniprot_id}")
    print(f"\nUnmapped proteins: {len(unmapped)}")
    for p in unmapped:
        print(p)
    print(f"\nMapped protein list saved to {mapped_path}")
    print(f"Unmapped protein list saved to {unmapped_path}")
    print(f"Errors saved to {error_path}")

if __name__ == "__main__":
    main() 