# AQSE Pipeline: Avoidome QSAR Similarity Expansion

This directory contains the implementation of the AQSE (Avoidome QSAR Similarity Expansion) pipeline for enriching the chemical space of avoidome proteins through similarity-based expansion using the Papyrus database.

## Overview

The AQSE pipeline implements a simplified version of the computational approach described in the proteochemometric modeling literature, specifically adapted for avoidome proteins. The pipeline consists of three main steps:

1. **Input Preparation**: Load avoidome protein sequences and prepare them for BLAST search
2. **Protein Similarity Search**: Use BLAST to find similar proteins in Papyrus database
3. **Data Collection Strategy**: Collect bioactivity data for similar proteins and create expanded datasets

## Pipeline Steps

### Step 1: Input Preparation (`01_input_preparation.py`)

**Purpose**: Prepare avoidome protein sequences for similarity search

**Inputs**:
- `avoidome_prot_list.csv`: List of avoidome proteins with UniProt IDs

**Processes**:
- Load and clean avoidome protein list
- Fetch protein sequences from UniProt database
- Create FASTA files for BLAST search
- Generate BLAST configuration file

**Outputs**:
- `avoidome_sequences.csv`: Protein sequences with metadata
- `avoidome_proteins_combined.fasta`: Combined FASTA file for BLAST
- Individual FASTA files for each protein
- `blast_config.txt`: BLAST configuration file
- `input_preparation_summary.txt`: Summary of preparation results

### Step 2: Protein Similarity Search (`02_protein_similarity_search.py`)

**Purpose**: Find proteins similar to avoidome proteins using BLAST

**Inputs**:
- FASTA files from Step 1
- Papyrus protein database (BLAST format)

**Processes**:
- Run BLAST search for each avoidome protein against Papyrus
- Parse BLAST results and create similarity matrices
- Identify similar proteins at different thresholds:
  - High similarity: ≥70% identity
  - Medium similarity: ≥50% identity  
  - Low similarity: ≥30% identity

**Outputs**:
- BLAST result files for each protein
- Similarity matrices for each threshold
- `similarity_search_summary.csv`: Summary of similarity results
- Visualization plots (distributions, heatmaps)

### Step 3: Data Collection Strategy (`03_data_collection_strategy.py`)

**Purpose**: Collect bioactivity data for similar proteins and create expanded datasets

**Inputs**:
- Similarity search results from Step 2
- Papyrus database (SQLite format)

**Processes**:
- Query Papyrus database for bioactivity data of similar proteins
- Standardize activity values to pIC50
- Filter by activity thresholds (pIC50 ≥ 1.0)
- Create expanded datasets for each similarity level

**Outputs**:
- Expanded datasets for each similarity threshold
- `dataset_statistics.csv`: Statistics for each dataset
- Visualization plots (dataset comparisons, activity distributions)

### Step 4: QSAR Model Creation (`04_2_qsar_model_creation.py`)

**Purpose**: Create QSAR models for each avoidome target using different similarity threshold protein sets

**Inputs**:
- Similarity search results from Step 2
- Papyrus database (direct queries, Step 3 bypassed)
- Avoidome protein sequences from Step 1

**Processes**:
- **For proteins WITH similar proteins**: Create threshold-specific models using similar proteins + target protein data
- **For proteins WITHOUT similar proteins**: Skip model creation (data identical to standard QSAR models)
- Generate molecular descriptors (Morgan fingerprints + physicochemical)
- Generate protein descriptors (ESM C embeddings)
- Train Random Forest models for each target-threshold combination
- Evaluate models with 5-fold cross-validation and external validation

**Outputs**:
- Trained Random Forest models (.pkl files) for proteins with similar proteins only
- Feature data (molecular + protein descriptors)
- Model predictions and performance metrics
- `aqse_model_results.csv`: Performance statistics
- Visualization plots (performance comparisons, feature importance)

**Optimization**: Standard AQSE models (proteins without similar proteins) are skipped as they would use identical data to standard QSAR models, saving computational resources.

## Usage

### Prerequisites

1. **Python Dependencies**:
   ```bash
   pip install pandas numpy matplotlib seaborn biopython requests papyrus-scripts rdkit scikit-learn
   ```

2. **ESM C Environment** (for protein embeddings):
   ```bash
   conda activate esmc
   pip install esm
   ```

3. **Papyrus-scripts Library**:
   - Handles automatic download and decompression of Papyrus data
   - No need for manual BLAST database creation
   - Provides integrated data filtering and querying

4. **Papyrus Database**:
   - Automatically downloaded by papyrus-scripts library
   - Uses full Papyrus database (not just Papyrus++ subset)
   - Available versions: 05.4, 05.5, 05.6, 05.7 (latest)
   - Contains ~15,000+ proteins and ~1.5M+ bioactivity records

### Running the Pipeline

#### Option 1: Run Complete Pipeline
```bash
python run_aqse_pipeline.py --papyrus-version 05.7
```

#### Option 2: Run Individual Steps
```bash
# Step 1 only
python run_aqse_pipeline.py --step 1

# Step 2 only (requires Step 1 completion)
python run_aqse_pipeline.py --step 2 --papyrus-version 05.7

# Step 3 only (requires Step 2 completion)
python run_aqse_pipeline.py --step 3 --papyrus-version 05.7

# Step 4 only (requires Step 3 completion)
python run_aqse_pipeline.py --step 4 --papyrus-version 05.7
```

#### Option 3: Custom Configuration
```bash
python run_aqse_pipeline.py --config config.json
```

### Configuration

Create a `config.json` file to customize pipeline parameters:

```json
{
    "base_dir": "/path/to/output/directory",
    "avoidome_file": "/path/to/avoidome_prot_list.csv",
    "papyrus_version": "05.7",
    "similarity_thresholds": {
        "high": 70.0,
        "medium": 50.0,
        "low": 30.0
    },
    "activity_thresholds": {
        "high_activity": 1.0,
        "medium_activity": 0.0,
        "low_activity": -1.0
    }
}
```

## Output Structure

```
avoidome_qsar_similarity_expansion/
├── 01_input_preparation/
│   ├── fasta_files/
│   ├── logs/
│   ├── avoidome_sequences.csv
│   ├── avoidome_proteins_combined.fasta
│   └── input_preparation_summary.txt
├── 02_similarity_search/
│   ├── blast_results/
│   ├── similarity_matrices/
│   ├── plots/
│   └── similarity_search_summary.csv
├── 03_data_collection/
│   ├── expanded_datasets/
│   ├── statistics/
│   ├── plots/
│   └── dataset_statistics.csv
├── run_aqse_pipeline.py
├── 01_input_preparation.py
├── 02_protein_similarity_search.py
├── 03_data_collection_strategy.py
├── README.md
└── aqse_pipeline.log
```

## Key Features

- **Modular Design**: Each step can be run independently
- **Comprehensive Logging**: Detailed logs for debugging and monitoring
- **Visualization**: Automatic generation of plots and summaries
- **Error Handling**: Robust error handling and recovery
- **Configurable**: Easy to modify thresholds and parameters
- **Scalable**: Can handle large protein datasets

## Expected Results

The pipeline will create expanded datasets with:
- **High similarity (≥70%)**: Most conservative, highest confidence
- **Medium similarity (50-70%)**: Balanced expansion vs. confidence
- **Low similarity (30-50%)**: Maximum expansion, lower confidence

Each dataset will contain:
- Protein sequences and metadata
- Bioactivity data (pIC50 values)
- Compound information (SMILES, molecular properties)
- Organism information
- Activity thresholds applied

## Troubleshooting

### Common Issues

1. **BLAST not found**: Ensure BLAST+ is installed and in PATH
2. **Database connection errors**: Check Papyrus database path and permissions
3. **Memory issues**: Reduce batch sizes or use more powerful hardware
4. **Network timeouts**: Increase timeout values for UniProt API calls

### Log Files

Check the following log files for detailed error information:
- `aqse_pipeline.log`: Main pipeline log
- `01_input_preparation/logs/`: Input preparation logs
- `02_similarity_search/logs/`: BLAST search logs
- `03_data_collection/logs/`: Data collection logs

## Next Steps

After running the AQSE pipeline, the expanded datasets can be used for:
- QSAR model training and validation
- Chemical space analysis
- Target prioritization
- Drug discovery applications

## Citation

If you use this pipeline in your research, please cite the original proteochemometric modeling approach and the Papyrus database.