# QSAR Modeling Dashboard

A comprehensive Streamlit dashboard for visualizing and analyzing QSAR modeling results from the Avoidome project.

## Overview

This dashboard provides interactive visualization and analysis of QSAR modeling results, including:

- **Model Performance Metrics**: R² scores, RMSE, accuracy, F1 scores, and AUC
- **Data Distribution Analysis**: Sample sizes across proteins and organisms
- **Cross-Organism Comparison**: Performance differences between human, mouse, and rat models
- **Model Type Comparison**: Morgan vs ESM+Morgan model performance
- **Protein-Specific Analysis**: Detailed analysis for individual proteins

## Features

### Overview Section
- **Summary Statistics**: Total models, proteins, samples, and average performance
- **Data Distribution**: Visual distribution of models by organism and type
- **Performance Overview**: Heatmaps and top performer tables

### Protein Analysis
- **By Protein**: Detailed analysis for individual proteins
- **By Organism**: Organism-specific performance metrics
- **Cross-Organism Comparison**: Side-by-side performance comparison

### Model Details
- **Morgan Models**: Analysis of Morgan fingerprint-based models
- **ESM+Morgan Models**: Analysis of ESM+Morgan combined models
- **Model Comparison**: Statistical comparison between model types

### Performance Metrics
- **Regression Metrics**: R² and RMSE distribution analysis
- **Classification Metrics**: Accuracy, F1, and AUC distribution analysis
- **Feature Importance**: Analysis of molecular descriptor importance

## Data Sources

The dashboard reads data from the following files:

- `analyses/standardized_qsar_models/modeling_summary.csv` - Morgan models results
- `analyses/standardized_qsar_models/esm_morgan_modeling_summary.csv` - ESM+Morgan models results
- `processed_data/papyrus_protein_check_results.csv` - Protein list with UniProt IDs

## Installation

### Prerequisites

Ensure you have Python 3.7+ installed with the following packages:

```bash
pip install streamlit pandas matplotlib seaborn numpy plotly
```

### Quick Start

1. **Run the dashboard directly:**
   ```bash
   streamlit run qsar_modeling_dashboard.py
   ```

2. **Or use the launcher script:**
   ```bash
   python run_qsar_dashboard.py
   ```

3. **The dashboard will open in your browser at:** `http://localhost:8502`

## Usage

### Navigation

The dashboard is organized into four main sections accessible via the sidebar:

1. **Overview** - High-level statistics and distributions
2. **Protein Analysis** - Protein and organism-specific analysis
3. **Model Details** - Detailed model performance analysis
4. **Performance Metrics** - Statistical analysis of performance metrics

### Key Features

#### Interactive Visualizations
- **Hover Information**: Hover over charts for detailed information
- **Zoom and Pan**: Interactive zooming and panning in Plotly charts
- **Filtering**: Filter data by protein, organism, or model type

#### Data Tables
- **Sortable Columns**: Click column headers to sort data
- **Search Functionality**: Use the search box to find specific proteins
- **Export Options**: Copy data to clipboard or download as CSV

#### Performance Analysis
- **Heatmaps**: Visual representation of performance across proteins and organisms
- **Box Plots**: Distribution analysis of performance metrics
- **Scatter Plots**: Correlation analysis between different metrics

## Model Types

### Morgan Models
- **Morgan Regression**: Continuous pchembl_value prediction using Morgan fingerprints
- **Morgan Classification**: Binary activity classification using Morgan fingerprints

### ESM+Morgan Models
- **ESM+Morgan Regression**: Continuous prediction using ESM protein embeddings + Morgan fingerprints
- **ESM+Morgan Classification**: Binary classification using ESM protein embeddings + Morgan fingerprints

## Organisms

The dashboard analyzes models across three organisms:

- **Human** (Homo sapiens) - Primary focus with most data
- **Mouse** (Mus musculus) - Limited data availability
- **Rat** (Rattus norvegicus) - Moderate data availability

## Performance Metrics

### Regression Metrics
- **R² Score**: Coefficient of determination (0-1, higher is better)
- **RMSE**: Root Mean Square Error (lower is better)

### Classification Metrics
- **Accuracy**: Overall classification accuracy (0-1, higher is better)
- **F1 Score**: Harmonic mean of precision and recall (0-1, higher is better)
- **AUC**: Area Under the ROC Curve (0-1, higher is better)

## Troubleshooting

### Common Issues

1. **Missing Data Files**
   - Ensure QSAR models have been trained
   - Check that summary CSV files exist in the correct locations

2. **Package Import Errors**
   - Install missing packages using pip
   - Ensure you're using the correct Python environment

3. **Dashboard Won't Load**
   - Check that port 8502 is available
   - Try using a different port: `streamlit run qsar_modeling_dashboard.py --server.port 8503`

### Data Requirements

The dashboard requires the following data structure:

```
analyses/standardized_qsar_models/
├── modeling_summary.csv
├── esm_morgan_modeling_summary.csv
├── morgan_regression/
│   ├── human/
│   ├── mouse/
│   └── rat/
├── morgan_classification/
│   ├── human/
│   ├── mouse/
│   └── rat/
├── esm_morgan_regression/
│   ├── human/
│   ├── mouse/
│   └── rat/
└── esm_morgan_classification/
    ├── human/
    ├── mouse/
    └── rat/
```

## Customization

### Adding New Metrics
To add new performance metrics, modify the data loading functions and add new visualization functions.

### Changing Color Schemes
Update the `color_discrete_sequence` parameters in Plotly charts to change color schemes.

### Adding New Sections
Add new pages to the `PAGES` dictionary and implement corresponding functions.

## Technical Details

### Architecture
- **Frontend**: Streamlit web application
- **Visualization**: Plotly for interactive charts, Matplotlib/Seaborn for static plots
- **Data Processing**: Pandas for data manipulation
- **Caching**: Streamlit's `@st.cache_data` for performance optimization

### Performance
- **Data Caching**: All data loading functions are cached to improve performance
- **Lazy Loading**: Data is loaded only when needed
- **Efficient Visualizations**: Optimized Plotly charts for large datasets

## Contributing

To contribute to the dashboard:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This dashboard is part of the Avoidome project and follows the same licensing terms.

## Support

For issues or questions:

1. Check the troubleshooting section
2. Review the data requirements
3. Ensure all dependencies are installed
4. Check the console output for error messages

---

**Dashboard Version**: 1.0  
**Last Updated**: September 11, 2025  
**Compatible with**: QSAR Modeling Results v1.0