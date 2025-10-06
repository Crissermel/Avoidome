# Dumbbell Plot Update Summary

## Changes Made
Updated the dumbbell plots to use **only human ESM+Morgan regression results** instead of the mixed organism results from the summary CSV file.

## What Was Changed

### **Function Updated**: `load_standard_qsar_results()`
- **Before**: Loaded from `esm_morgan_modeling_summary.csv` (included human, mouse, rat)
- **After**: Loads from individual JSON files in `/home/serramelendezcsm/RA/Avoidome/analyses/standardized_qsar_models/esm_morgan_regression/human/`

### **Data Source**
- **Location**: `analyses/standardized_qsar_models/esm_morgan_regression/human/`
- **Format**: Individual `results.json` files per protein
- **Organism**: Human only
- **Model Type**: ESM+Morgan regression only

## Results

### **Data Loaded Successfully**:
- **30 human ESM+Morgan regression models**
- **R² range**: -0.262 to 0.738
- **Sample sizes**: 150 to 4,492 compounds per protein

### **Proteins Included**:
CHRM1, SLC6A4, MAOB, ALDH1A1, KCNH2, CYP1A2, CYP2B6, CYP2C19, CYP2C9, CYP2D6, CYP3A4, HRH1, HSD11B1, HTR2B, MAOA, NR1I2, NR1I3, SCN5A, SLC6A2, SLC6A3, SLCO2B1, XDH, ADRA1A, ADRA2A, ADRB1, ADRB2, AHR, AKR7A3, AOX1, CACNB1, CAV1, CHRM2, CHRM3, CHRNA7, CNR2, CNRIP1, DIDO1, FMO1, GABPA, GSTA1, NAT8, ORM1, OXA1L, SMPDL3A, SULT1A1

## Impact on Dumbbell Plots

### **Both Dumbbell Plots Now Use**:
1. **Standard QSAR**: Human ESM+Morgan regression models only
2. **AQSE Models**: All available AQSE threshold models
3. **Consistent Comparison**: Human vs Human comparison for fair evaluation

### **Benefits**:
- **Consistent organism comparison** (human vs human)
- **More accurate performance comparison** 
- **Cleaner data** (no mixed organism results)
- **Better scientific validity** for the comparison

## Technical Details

### **Data Structure**:
```json
{
  "protein_name": "CYP1A2",
  "uniprot_id": "P05177", 
  "organism": "human",
  "n_samples": 329,
  "overall_metrics": {
    "cv_r2": 0.5087158739641404,
    "cv_rmse": 0.708302828687111,
    "cv_mae": 0.5386529966462774
  }
}
```

### **Dashboard Integration**:
- No changes needed to dumbbell plot functions
- Automatic loading of human-specific results
- Maintains all existing functionality and visualizations

## Verification
The updated function successfully loads 30 human ESM+Morgan regression models with proper R² scores, sample sizes, and protein identifiers. The dumbbell plots will now show a consistent human-to-human comparison between standard QSAR and AQSE models.



