import pandas as pd
import requests
import time

def get_pubchem_cid(chembl_id):
    # Try searching by RegistryID (ChEMBL ID)
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/xref/RegistryID/{chembl_id}/cids/JSON"
    r = requests.get(url)
    if r.status_code == 200:
        cids = r.json().get('IdentifierList', {}).get('CID', [])
        if cids:
            return cids[0]
    # Fallback: try searching by name
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{chembl_id}/cids/JSON"
    r = requests.get(url)
    if r.status_code == 200:
        cids = r.json().get('IdentifierList', {}).get('CID', [])
        if cids:
            return cids[0]
    return None


def get_pubchem_consolidated_references(cid):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON/?heading=Literature"
    r = requests.get(url)
    if r.status_code != 200:
        print(f"Failed to fetch data for CID {cid}")
        return [], []
    data = r.json()
    try:
        sections = data['Record']['Section']
        for section in sections:
            if section.get('TOCHeading') == 'Literature':
                for sub in section.get('Section', []):
                    if sub.get('TOCHeading') == 'Consolidated References':
                        info_list = sub.get('Information', [])
                        dois = []
                        for info in info_list:
                            # DOI may be in info['Value']['DOI'] or info['Value'] (if string)
                            value = info.get('Value', {})
                            if isinstance(value, dict) and 'DOI' in value:
                                dois.append(value['DOI'])
                            elif isinstance(value, str) and value.startswith('10.'):
                                dois.append(value)
                        print(f"DOIs for CID {cid}: {dois}")
                        return info_list, dois
        print(f"No consolidated references found for CID {cid}.")
    except Exception as e:
        print(f"Error parsing literature section for CID {cid}: {e}")
    return [], []

def main():
    input_csv = 'processed_data/compounds.csv'
    output_csv = 'processed_data/compounds_with_pubchem_refs.csv'
    df = pd.read_csv(input_csv)
    results = []
    for chembl_id in df['molecule_chembl_id']:
        print(f"Processing {chembl_id}...")
        cid = get_pubchem_cid(chembl_id)
        if cid:
            info_list, _ = get_pubchem_consolidated_references(cid)
            if info_list and 'ReferenceNumber' in info_list[0]:
                ref_count = info_list[0]['ReferenceNumber']
            else:
                ref_count = 0
        else:
            print(f"CID not found for {chembl_id}")
            ref_count = 0
        results.append({
            'molecule_chembl_id': chembl_id,
            'pubchem_cid': cid,
            'consolidated_reference_count': ref_count
        })
        time.sleep(0.2)  # Be gentle to the API
    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)
    print(f"Wrote results to {output_csv}.")

if __name__ == '__main__':
    main() 