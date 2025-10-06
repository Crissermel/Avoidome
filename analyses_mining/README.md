# ChEMBL-mapped Analysis Dashboard

This directory contains the ChEMBL-mapped analysis dashboard and related data files.

## Dashboard

### `chembl_dashboard.py`

A Streamlit dashboard specifically for analyzing ChEMBL-mapped bioactivity data.

**Usage:**
```bash
streamlit run analyses_mining/chembl_dashboard.py
```

**Features:**
- **ChEMBL-mapped Protein List**: Explore proteins successfully mapped to ChEMBL
- **Bioactivity Points per Protein**: Visualize distribution of bioactivity data across proteins
- **Bioactivity Type Distribution**: Analyze different types of bioactivity data
- **Interactive Data Exploration**: Explore the complete ChEMBL-mapped dataset

**Port:** The dashboard runs on port 8502 (different from the main dashboard on 8501)

## Data Files

### `unique_proteins_mapped_with_chembl.csv`
Contains the list of proteins that have been successfully mapped to ChEMBL with their corresponding ChEMBL target IDs.

### Other Files
- `unique_proteins_mapped.txt`: Text file with mapped proteins
- `unique_proteins_unmapped.txt`: Text file with unmapped proteins
- `unique_proteins_unprocessed.txt`: Text file with unprocessed proteins

## Analysis Scripts

### `convert_mapped_to_csv.py`
Converts the mapped protein data to CSV format for easier analysis.

### `process_regex_extracted.py`
Processes regex-extracted protein data.

### `protein_mining_C.py` and `protein_mining_L.py`
Protein mining scripts for different analysis approaches.

## Navigation

The dashboard provides a sidebar navigation with the following pages:

1. **Introduction**: Overview and usage instructions
2. **ChEMBL-mapped Protein List**: Complete protein mapping data
3. **ChEMBL-mapped Bioactivity Points per Protein**: Distribution analysis
4. **ChEMBL-mapped Bioactivity Type Distribution**: Type-specific analysis
5. **Explore ChEMBL-mapped Bioactivity Data**: Interactive data exploration

## Dependencies

The dashboard requires the same dependencies as the main dashboard:
- streamlit
- pandas
- matplotlib
- seaborn
- numpy

## Data Sources

The dashboard reads data from:
- `analyses_mining/unique_proteins_mapped_with_chembl.csv` (protein mapping data)
- ChEMBL-mapped bioactivity profiles (via `functions.data_loading`)

## Integration

This dashboard is separate from the main Avoidome dashboard to provide focused analysis of ChEMBL-mapped data. The main dashboard focuses on Avoidome-specific analyses, while this dashboard specializes in ChEMBL integration and mapping results. 