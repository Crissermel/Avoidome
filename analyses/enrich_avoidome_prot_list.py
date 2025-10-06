import csv
import requests
import time

INPUT_CSV = 'primary_data/avoidome_prot_list.csv'
OUTPUT_CSV = 'processed_data/avoidome_prot_list_enriched.csv'

# UniProt API endpoint for protein info
UNIPROT_API = 'https://rest.uniprot.org/uniprotkb/{}.json'

def fetch_uniprot_info(uniprot_id):
    try:
        url = UNIPROT_API.format(uniprot_id)
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return '', ''
        data = resp.json()
        # Function
        function = ''
        for comment in data.get('comments', []):
            if comment.get('commentType') == 'FUNCTION':
                function = ' '.join([t['value'] for t in comment.get('texts', [])])
                break
        # Pathways
        pathways = []
        for xref in data.get('uniProtKBCrossReferences', []):
            if xref.get('database') == 'Reactome':
                pathways.append(xref.get('id'))
        pathway_str = '; '.join(pathways)
        return function, pathway_str
    except Exception as e:
        return '', ''

def main():
    with open(INPUT_CSV, newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        if reader.fieldnames is not None:
            fieldnames = list(reader.fieldnames) + ['Function', 'Pathways']
        else:
            # Fallback: set default fieldnames
            fieldnames = ['Name', 'Alternative Names', 'UniProt ID', 'ChEMBL_target', '', 'source', 'Function', 'Pathways']
        rows = list(reader)

    for row in rows:
        uniprot_id = row.get('UniProt ID', '').strip()
        if uniprot_id:
            function, pathways = fetch_uniprot_info(uniprot_id)
            row['Function'] = function
            row['Pathways'] = pathways
            time.sleep(0.5)  # Be gentle to UniProt API
        else:
            row['Function'] = ''
            row['Pathways'] = ''

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

if __name__ == '__main__':
    main() 