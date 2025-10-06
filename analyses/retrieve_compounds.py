import pandas as pd

def main():
    input_csv = 'processed_data/avoidome_bioactivity_profile.csv'
    output_csv = 'processed_data/compounds.csv'
    df = pd.read_csv(input_csv)
    if 'molecule_chembl_id' not in df.columns:
        print(f"Column 'molecule_chembl_id' not found in {input_csv}.")
        return
    unique_compounds = df['molecule_chembl_id'].dropna().drop_duplicates()
    unique_compounds = unique_compounds.to_frame()
    unique_compounds.to_csv(output_csv, index=False)
    print(f"Wrote {len(unique_compounds)} unique compounds to {output_csv}.")

if __name__ == '__main__':
    main() 
    # Wrote 168125 unique compounds to compounds.csv.