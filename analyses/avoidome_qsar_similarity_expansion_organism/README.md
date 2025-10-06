# AQSE-org: Organism-Specific Avoidome QSAR Similarity Expansion

This directory contains the AQSE-org workflow, an organism-specific version of the Avoidome QSAR Similarity Expansion (AQSE) pipeline. The workflow processes avoidome proteins across different organisms (human, mouse, rat) and creates QSAR models using similar proteins from the Papyrus database.

## Overview

The AQSE-org workflow consists of three main steps:

1. **Organism-Specific Protein Mapping**: Maps avoidome proteins to their organism-specific UniProt IDs
2. **Organism-Specific Similarity Search**: Finds similar proteins in Papyrus database for each organism
3. **Organism-Specific QSAR Model Creation**: Creates QSAR models using similar proteins and organism-specific bioactivity data

## Workflow Steps

### Step 1: Organism-Specific Protein Mapping (`01_organism_protein_mapping.py`)

- Loads avoidome protein list from `avoidome_prot_list.csv`
- Maps proteins to organism-specific UniProt IDs using `prot_orgs_extended.csv`
- Fetches protein sequences from UniProt for each organism
- Creates FASTA files for similarity search
- Outputs organism-specific protein mappings and sequences

**Input Files:**
- `/home/serramelendezcsm/RA/Avoidome/primary_data/avoidome_prot_list.csv`
- `/home/serramelendezcsm/RA/Avoidome/primary_data/prot_orgs_extended.csv`

**Output Files:**
- `{organism}_protein_mappings.csv`: Protein mappings for each organism
- `{organism}_sequences.csv`: Protein sequences for each organism
- `{organism}_proteins_combined.fasta`: Combined FASTA files for each organism
- `organism_mappings_summary.csv`: Combined summary of all mappings

### Step 2: Organism-Specific Similarity Search (`02_organism_similarity_search.py`)

- Loads organism-specific protein sequences
- Searches for similar proteins in Papyrus database
- Creates similarity matrices for different thresholds (high: 70%, medium: 50%, low: 30%)
- Generates visualizations and summary statistics

**Input Files:**
- FASTA files from Step 1
- Papyrus database (via papyrus-scripts)

**Output Files:**
- `{organism}_similarity_matrix_{threshold}.csv`: Similarity matrices for each organism and threshold
- `{organism}_similarity_search_summary.csv`: Summary statistics for each organism
- `all_organisms_similarity_summary.csv`: Combined summary across all organisms
- Visualization plots in `plots/` directory

### Step 3: Organism-Specific QSAR Model Creation (`03_organism_qsar_model_creation.py`)

- Calculates Morgan fingerprints for all Papyrus compounds
- Creates QSAR models for proteins with similar proteins
- Uses similar proteins + target protein training data for training
- Uses target protein holdout data for testing
- Includes Morgan descriptors, physicochemical descriptors, and ESM C descriptors

**Input Files:**
- Organism mappings from Step 1
- Similarity search results from Step 2
- Papyrus bioactivity data

**Output Files:**
- `{organism}_{target}_{uniprot}_{threshold}_model.pkl`: Trained QSAR models
- `{organism}_{target}_{uniprot}_{threshold}_predictions.csv`: Test predictions
- `{organism}_{target}_{uniprot}_{threshold}_metrics.json`: Model metrics
- `organism_qsar_model_results.csv`: Combined results summary

## Usage

### Running the Complete Pipeline

```bash
cd /home/serramelendezcsm/RA/Avoidome/analyses/avoidome_qsar_similarity_expansion_organism
python run_aqse_org_pipeline.py
```

### Running Individual Steps

```bash
# Step 1: Organism-specific protein mapping
python run_aqse_org_pipeline.py --step 1

# Step 2: Organism-specific similarity search
python run_aqse_org_pipeline.py --step 2

# Step 3: Organism-specific QSAR model creation
python run_aqse_org_pipeline.py --step 3
```

### Custom Configuration

```bash
python run_aqse_org_pipeline.py --config config.json --base-dir /path/to/output --papyrus-version 05.7
```

## Configuration

The pipeline uses a default configuration that can be overridden:

```json
{
    "base_dir": "/home/serramelendezcsm/RA/Avoidome/analyses/avoidome_qsar_similarity_expansion_organism",
    "avoidome_file": "/home/serramelendezcsm/RA/Avoidome/primary_data/avoidome_prot_list.csv",
    "organism_mapping_file": "/home/serramelendezcsm/RA/Avoidome/primary_data/prot_orgs_extended.csv",
    "papyrus_version": "05.7",
    "similarity_thresholds": {
        "high": 70.0,
        "medium": 50.0,
        "low": 30.0
    }
}
```

## Dependencies

- Python 3.7+
- pandas
- numpy
- scikit-learn
- rdkit
- Bio
- matplotlib
- seaborn
- papyrus-scripts
- requests

## Output Structure

```
avoidome_qsar_similarity_expansion_organism/
├── 01_organism_mapping/
│   ├── fasta_files/
│   ├── human_protein_mappings.csv
│   ├── mouse_protein_mappings.csv
│   ├── rat_protein_mappings.csv
│   └── organism_mappings_summary.csv
├── 02_similarity_search/
│   ├── similarity_matrices/
│   ├── plots/
│   ├── human_similarity_search_summary.csv
│   ├── mouse_similarity_search_summary.csv
│   ├── rat_similarity_search_summary.csv
│   └── all_organisms_similarity_summary.csv
├── 03_qsar_models/
│   ├── morgan_fingerprints/
│   ├── qsar_models/
│   ├── results/
│   ├── descriptors_cache/
│   └── organism_qsar_model_results.csv
└── aqse_org_pipeline_summary.txt
```

## Key Features

1. **Organism-Specific Processing**: Handles human, mouse, and rat proteins separately
2. **Similarity-Based Expansion**: Uses similar proteins to expand training data
3. **Comprehensive Descriptors**: Combines Morgan fingerprints, physicochemical descriptors, and ESM C descriptors
4. **Robust Validation**: Uses holdout testing with target protein data
5. **Caching**: Caches computed descriptors to avoid recomputation
6. **Visualization**: Generates plots and summaries for analysis

## Notes

- The workflow skips proteins without similar proteins (no expansion possible)
- ESM C descriptors require the `run_esm_embeddings` module to be available
- Morgan fingerprints are calculated once and reused across all models
- The pipeline is designed to handle missing data gracefully
- All file paths use absolute paths as per user preference

## Troubleshooting

1. **Import Errors**: Ensure all dependencies are installed and paths are correct
2. **Memory Issues**: The Morgan fingerprint calculation can be memory-intensive for large datasets
3. **Network Issues**: UniProt API calls may fail; the pipeline includes retry logic
4. **ESM Errors**: If ESM is not available, dummy descriptors will be used

## Author

AQSE-org Pipeline
Date: 2025
