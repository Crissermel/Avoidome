# Protein Type QSAR Modeling

This directory contains the QSAR modeling system for protein type groups, which pools bioactivity data from proteins of the same functional family and creates separate models for each animal (human, mouse, rat).

## Overview

The protein type QSAR modeling system extends the individual protein QSAR approach by grouping related proteins and pooling their bioactivity data. This approach provides several advantages:

- **Increased data volume**: Pool data from multiple related proteins within each family
- **Better model robustness**: More training samples for more reliable models
- **Family-specific insights**: Understand structure-activity relationships within protein families
- **Cross-species analysis**: Separate models for human, mouse, and rat
- **Transfer learning potential**: Models can inform related proteins within the same family

## Architecture

### **Model Type**: Random Forest Regressor
- **Algorithm**: RandomForestRegressor from scikit-learn
- **Cross-validation**: 5-fold CV with shuffling
- **Hyperparameters**: n_estimators=100, random_state=42, n_jobs=-1
- **Minimum data requirement**: 10 samples per animal per protein group

### **Molecular Descriptors**: Morgan Fingerprints
- **Type**: Extended Connectivity Fingerprints (ECFP-like)
- **Radius**: 2 (captures local molecular environment)
- **Bits**: 2048 (high-resolution molecular representation)
- **Implementation**: RDKit Morgan fingerprint generation

### **Data Pooling Strategy**
- **Grouping**: Proteins grouped by functional families (CYP, SLC, receptors, etc.)
- **Animal separation**: Data pooled separately for human, mouse, and rat
- **Deduplication**: Removes duplicate SMILES-pchembl_value pairs
- **Quality control**: Filters out invalid SMILES and missing pchembl values

## Protein Groups

The system processes 33 protein groups based on functional families:

### **Metabolism Enzymes**
- **CYP**: Cytochrome P450 enzymes (7 proteins)
- **AOX**: Aldehyde oxidase enzymes (2 proteins)
- **MAO**: Monoamine oxidase enzymes (2 proteins)
- **ALDH**: Aldehyde dehydrogenase enzymes (1 protein)
- **ADH**: Alcohol dehydrogenase enzymes (2 proteins)
- **HSD**: Hydroxysteroid dehydrogenase enzymes (1 protein)
- **AKR**: Aldo-keto reductase enzymes (2 proteins)
- **FMO**: Flavin-containing monooxygenase enzymes (2 proteins)
- **SULT**: Sulfotransferase enzymes (2 proteins)
- **GST**: Glutathione S-transferase enzymes (2 proteins)

### **Transporters**
- **SLC**: Solute carrier transporters (3 proteins)
- **SLCO**: Solute carrier organic anion transporters (4 proteins)

### **Receptors**
- **HTR**: Serotonin receptors (2 proteins)
- **ADRA**: Alpha adrenergic receptors (2 proteins)
- **ADRB**: Beta adrenergic receptors (2 proteins)
- **CHRM**: Muscarinic acetylcholine receptors (3 proteins)
- **CHRN**: Nicotinic acetylcholine receptors (5 proteins)
- **CNR**: Cannabinoid receptors (2 proteins)
- **HRH**: Histamine receptors (1 protein)

### **Ion Channels**
- **KCN**: Potassium voltage-gated channels (2 proteins)
- **SCN**: Sodium voltage-gated channels (2 proteins)
- **CACN**: Calcium voltage-gated channels (2 proteins)

### **Other Proteins**
- **AHR**: Aryl hydrocarbon receptor and related (2 proteins)
- **NR**: Nuclear receptors (2 proteins)
- **XDH**: Xanthine dehydrogenase (1 protein)
- **ORM**: Orosomucoid proteins (2 proteins)
- **CAV**: Caveolin proteins (2 proteins)
- **SMPDL**: Sphingomyelin phosphodiesterase-like proteins (2 proteins)
- **GABP**: GA binding protein (1 protein)
- **NAT**: N-acetyltransferase (1 protein)
- **DIDO**: Death inducer-obliterator (1 protein)
- **OXA**: Oxidase assembly proteins (1 protein)
- **COX**: Cytochrome c oxidase assembly proteins (1 protein)

## Directory Structure

```
qsar_papyrus_modelling_prottype/
├── protein_type_qsar_modeling.py    # Main QSAR modeling script
├── requirements.txt                  # Python dependencies
├── README.md                         # This documentation
├── protein_type_qsar_modeling.log   # Execution log
├── overall_qsar_modeling_summary.csv # Summary across all groups
├── detailed_qsar_modeling_report.txt # Detailed text report
└── {GROUP_NAME}/                     # Directory for each protein group
    ├── {GROUP_NAME}_human_model.pkl
    ├── {GROUP_NAME}_human_metadata.json
    ├── {GROUP_NAME}_human_cv_results.csv
    ├── {GROUP_NAME}_mouse_model.pkl
    ├── {GROUP_NAME}_mouse_metadata.json
    ├── {GROUP_NAME}_mouse_cv_results.csv
    ├── {GROUP_NAME}_rat_model.pkl
    ├── {GROUP_NAME}_rat_metadata.json
    ├── {GROUP_NAME}_rat_cv_results.csv
    └── {GROUP_NAME}_modeling_summary.txt
```

## Usage

### 1. Install Dependencies

```bash
cd /home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_modelling_prottype
pip install -r requirements.txt
```

### 2. Run Protein Type QSAR Modeling

```bash
python protein_type_qsar_modeling.py
```

This will:
- Load the protein list with protein groups
- Initialize the Papyrus dataset
- Group proteins by functional families
- Pool bioactivity data separately for each animal
- Create Morgan fingerprints for molecular representation
- Train Random Forest models with 5-fold CV for each animal
- Save trained models and metadata
- Generate comprehensive performance reports

## Output Files

### **Model Files**
- `{GROUP_NAME}_{animal}_model.pkl`: Trained Random Forest model
- `{GROUP_NAME}_{animal}_metadata.json`: Model metadata and parameters
- `{GROUP_NAME}_{animal}_cv_results.csv`: Cross-validation results

### **Summary Files**
- `{GROUP_NAME}_modeling_summary.txt`: Group-specific performance summary
- `overall_qsar_modeling_summary.csv`: Summary across all groups
- `detailed_qsar_modeling_report.txt`: Comprehensive modeling report

### **Performance Metrics**
- **R² Score**: Coefficient of determination (0 to 1, higher is better)
- **RMSE**: Root mean square error (lower is better)
- **MAE**: Mean absolute error (lower is better)
- **Cross-validation**: 5-fold CV for robust performance estimates

## Data Flow

### **1. Data Loading**
- Load protein list with protein groups
- Initialize Papyrus database connection
- Load full bioactivity dataset

### **2. Protein Grouping**
- Analyze protein distribution across groups
- Create group-specific protein collections
- Validate group assignments

### **3. Data Pooling**
- Retrieve bioactivity data for each protein
- Pool data by animal (human, mouse, rat)
- Add source protein and organism annotations
- Remove duplicates and invalid entries

### **4. Molecular Representation**
- Convert SMILES to Morgan fingerprints
- Validate molecular structures
- Create feature matrices for modeling

### **5. Model Training**
- 5-fold cross-validation for each animal
- Random Forest model training
- Performance metric calculation
- Model persistence and metadata storage

### **6. Reporting**
- Group-specific performance summaries
- Overall modeling statistics
- Top performing groups identification
- Cross-species performance comparison

## Performance Analysis

### **Model Quality Assessment**
- **Excellent (R² > 0.7)**: High predictive performance
- **Good (R² 0.5-0.7)**: Moderate predictive performance
- **Moderate (R² 0.3-0.5)**: Basic predictive performance
- **Poor (R² < 0.3)**: Limited predictive performance

### **Data Volume Impact**
- **High volume (>1000 samples)**: Typically better model performance
- **Medium volume (100-1000 samples)**: Good model performance
- **Low volume (10-100 samples)**: Basic model performance
- **Insufficient (<10 samples)**: Cannot train reliable models

## Benefits for Drug Discovery

### **1. Increased Predictive Power**
- Larger training datasets improve model accuracy
- Family-specific patterns enhance predictions
- Cross-species data provides broader applicability

### **2. Better Understanding of Structure-Activity Relationships**
- Identify common binding patterns within protein families
- Understand family-specific molecular requirements
- Enable rational drug design within protein families

### **3. Improved Model Transferability**
- Models trained on one protein can inform related proteins
- Family-specific insights enable better predictions
- Reduced need for extensive experimental data for each protein

### **4. Cross-Species Analysis**
- Compare binding patterns across species
- Identify conserved vs. species-specific interactions
- Enable translational research applications

## Future Enhancements

### **1. Advanced Modeling Approaches**
- Deep learning models (neural networks)
- Ensemble methods (stacking, blending)
- Graph neural networks for molecular representation

### **2. Enhanced Molecular Descriptors**
- 3D molecular descriptors
- Pharmacophore-based features
- Protein-ligand interaction fingerprints

### **3. Model Interpretation**
- Feature importance analysis
- SHAP values for model explainability
- Molecular substructure identification

### **4. Active Learning**
- Iterative model improvement
- Uncertainty-based sample selection
- Targeted data collection strategies

## Notes

- The system requires the Papyrus database to be accessible
- Processing time depends on the size of the Papyrus dataset
- Each protein group creates its own subdirectory with models
- Log files track execution progress and any errors
- Models are saved in joblib format for easy loading and prediction

## Troubleshooting

### **Common Issues**
1. **Insufficient data**: Ensure protein groups have enough bioactivity data
2. **Invalid SMILES**: Check molecular structure validity
3. **Memory issues**: Large datasets may require increased memory allocation
4. **Papyrus connection**: Verify database accessibility and permissions

### **Performance Optimization**
1. **Parallel processing**: Models are trained with n_jobs=-1 for maximum CPU utilization
2. **Memory management**: Data is processed in chunks to manage memory usage
3. **Caching**: Consider implementing descriptor caching for large datasets 