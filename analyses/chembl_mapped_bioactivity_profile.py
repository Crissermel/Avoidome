import pandas as pd
import requests
import time
import os

# Read protein list
input_csv = os.path.join(os.path.dirname(__file__), '../analyses_mining/unique_proteins_mapped_with_chembl.csv')
prot_df = pd.read_csv(input_csv)

# Only keep rows with Chembl_code (not null or empty)
prot_df = prot_df.loc[pd.notna(prot_df['Chembl_code']) & (prot_df['Chembl_code'] != '')]

# Helper to get bioactivities for ChEMBL target
def get_bioactivities(chembl_id):
    all_activities = []
    offset = 0
    limit = 1000
    while True:
        url = f"https://www.ebi.ac.uk/chembl/api/data/activity?target_chembl_id={chembl_id}&limit={limit}&offset={offset}"
        r = requests.get(url, headers={"Accept": "application/json"})
        activities = r.json().get('activities', [])
        all_activities.extend(activities)
        if len(activities) < limit:
            break
        offset += limit
        time.sleep(0.2)  # Be gentle to the API
    return all_activities

output_csv = os.path.join(os.path.dirname(__file__), '../processed_data/chembl_mapped_bioactivity_profile.csv')
first_write = True
for idx, row in prot_df.iterrows():
    name = row['Protein']
    uniprot = row['UniProt_Accession']
    chembl_id = row['Chembl_code']
    if pd.isna(chembl_id) or chembl_id == '':
        print(f"No ChEMBL target for {name} ({uniprot})")
        continue
    print(f"{name} ({uniprot}) -> {chembl_id}")
    activities = get_bioactivities(chembl_id)
    print(f"  Found {len(activities)} activities")
    # Print all information types (keys) and their values for the first activity (if any)
    if activities:
        print(f"  Example field values for {chembl_id} (first activity):")
        for k, v in activities[0].items():
            print(f"    {k}: {v}")
    # Print all information types (keys) returned by the API for this target
    all_keys = set()
    for act in activities:
        all_keys.update(act.keys())
    print(f"  Information types (fields) in activities for {chembl_id}: {sorted(all_keys)}")
    rows = []
    for act in activities:
        organism = act.get('target_organism') or act.get('organism') or 'unknown'
        rows.append({
            "Protein Name": name,
            "UniProt ID": uniprot,
            "ChEMBL Target": chembl_id,
            "Organism": organism,
            "Bioactivity Type": act.get('standard_type'),
            "Value": act.get('standard_value'),
            "Units": act.get('standard_units'),
            "Compound": act.get('molecule_chembl_id'),
            "Assay Type": act.get('assay_type'),
            "Reference": act.get('document_chembl_id'),
            # Additional requested fields
            "activity_id": act.get('activity_id'),
            "assay_chembl_id": act.get('assay_chembl_id'),
            "assay_description": act.get('assay_description'),
            "assay_type": act.get('assay_type'),
            "canonical_smiles": act.get('canonical_smiles'),
            "document_chembl_id": act.get('document_chembl_id'),
            "document_journal": act.get('journal'),
            "document_year": act.get('year'),
            "ligand_efficiency": act.get('ligand_efficiency'),
            "molecule_chembl_id": act.get('molecule_chembl_id'),
            "pchembl_value": act.get('pchembl_value'),
            "potential_duplicate": act.get('potential_duplicate'),
            "target_chembl_id": act.get('target_chembl_id'),
            "value": act.get('value'),
        })
    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        out_df.to_csv(output_csv, index=False, mode='w' if first_write else 'a', header=first_write)
        first_write = False
    print(f"Saved {len(out_df)} rows for {name} ({chembl_id})")
print('All targets processed.') 