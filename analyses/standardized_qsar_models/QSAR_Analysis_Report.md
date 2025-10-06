# QSAR Model Performance Analysis Report
============================================================

## Executive Summary

This report analyzes the performance of four QSAR model configurations:
1. **Morgan + Physicochemical** (50-sample threshold)
2. **ESM + Morgan + Physicochemical** (50-sample threshold)
3. **Morgan + Physicochemical** (30-sample threshold) - if available
4. **ESM + Morgan + Physicochemical** (30-sample threshold) - if available

## Model Performance Summary

   Model Type  Total Proteins  Completed  Skipped  Errors Completion Rate (%) Avg R² Avg RMSE Avg Accuracy Avg F1 Avg AUC Best R² Best Accuracy
    Morgan_50              52         27       22       3                51.9  0.557    0.709        0.853  0.709   0.882   0.736         0.988
ESM_Morgan_50              52         27       22       3                51.9  0.557    0.708        0.850  0.708   0.877   0.738         0.988

## Key Findings

### Best Regression Performance (R²)
- **Model**: ESM_Morgan_50
- **Average R²**: 0.557
- **Best R²**: 0.738

### Best Classification Performance (Accuracy)
- **Model**: Morgan_50
- **Average Accuracy**: 0.853
- **Best Accuracy**: 0.988

### Model Comparison Insights

- **ESM Addition Impact on R²**: +0.000
- **ESM Addition Impact on Accuracy**: -0.003

## Detailed Protein-by-Protein Comparison

The following table shows performance metrics for each protein across all model types:

Protein Morgan_50_R2 Morgan_50_RMSE Morgan_50_Accuracy Morgan_50_F1 Morgan_50_AUC  Morgan_50_Samples ESM_Morgan_50_R2 ESM_Morgan_50_RMSE ESM_Morgan_50_Accuracy ESM_Morgan_50_F1 ESM_Morgan_50_AUC  ESM_Morgan_50_Samples
 ADRA1A        0.520          0.797              0.809        0.860         0.882                566            0.523              0.796                  0.807            0.860             0.878                    566
 ADRA2A        0.489          0.883              0.810        0.776         0.887                526            0.489              0.884                  0.806            0.766             0.880                    526
  ADRB1        0.545          0.746              0.816        0.784         0.884                614            0.544              0.746                  0.813            0.780             0.881                    614
  ADRB2        0.697          0.884              0.857        0.895         0.917                860            0.697              0.885                  0.852            0.892             0.914                    860
ALDH1A1        0.490          0.678              0.833        0.699         0.889                150            0.500              0.671                  0.840            0.714             0.881                    150
  CHRM1        0.488          0.890              0.796        0.702         0.853               1103            0.488              0.890                  0.797            0.707             0.852                   1103
  CHRM2        0.600          0.876              0.809        0.805         0.892                721            0.600              0.877                  0.806            0.804             0.888                    721
  CHRM3        0.730          0.823              0.847        0.876         0.931               1209            0.730              0.823                  0.840            0.869             0.932                   1209
 CHRNA7        0.578          0.766              0.839        0.742         0.894                595            0.577              0.767                  0.834            0.733             0.890                    595
   CNR2        0.498          0.817              0.778        0.785         0.854               4295            0.498              0.818                  0.774            0.780             0.852                   4295
 CYP1A2        0.510          0.708              0.863        0.634         0.883                329            0.509              0.708                  0.866            0.633             0.871                    329
 CYP2C9        0.535          0.501              0.957        0.412         0.947                466            0.534              0.502                  0.959            0.457             0.952                    466
 CYP2D6        0.414          0.659              0.909        0.409         0.770                607            0.413              0.660                  0.908            0.417             0.791                    607
 CYP3A4        0.572          0.548              0.963        0.633         0.887               1387            0.573              0.548                  0.963            0.638             0.875                   1387
   HRH1        0.642          0.717              0.812        0.788         0.881                945            0.643              0.716                  0.813            0.791             0.878                    945
HSD11B1        0.630          0.632              0.825        0.859         0.903               2046            0.628              0.634                  0.822            0.856             0.901                   2046
  HTR2B        0.341          0.720              0.739        0.614         0.786               1180            0.339              0.721                  0.724            0.595             0.782                   1180
  KCNH2        0.501          0.667              0.945        0.643         0.920               4492            0.501              0.667                  0.943            0.630             0.914                   4492
   MAOA        0.560          0.793              0.925        0.632         0.897               1130            0.561              0.793                  0.919            0.616             0.902                   1130
   MAOB        0.586          0.831              0.841        0.743         0.901               1519            0.585              0.832                  0.837            0.738             0.903                   1519
  NR1I2        0.595          0.568              0.890        0.648         0.894                344            0.596              0.568                  0.887            0.642             0.895                    344
  NR1I3        0.562          0.678              0.881        0.788         0.943                 59            0.560              0.679                  0.898            0.812             0.938                     59
  SCN5A        0.436          0.516              0.988        0.333         0.744                329            0.435              0.517                  0.988            0.333             0.653                    329
 SLC6A2        0.577          0.671              0.798        0.785         0.869               2021            0.577              0.671                  0.797            0.786             0.866                   2021
 SLC6A3        0.623          0.640              0.832        0.743         0.890               1646            0.624              0.639                  0.817            0.718             0.884                   1646
 SLC6A4        0.579          0.734              0.807        0.823         0.888               2958            0.579              0.734                  0.807            0.824             0.886                   2958
    XDH        0.736          0.384              0.867        0.735         0.928                 98            0.738              0.383                  0.847            0.717             0.929                     98

## Recommendations

1. **Feature Engineering**: ESM embeddings provide additional protein-specific information
2. **Sample Size**: Lower thresholds (30 samples) may capture more proteins but with potentially lower reliability
3. **Model Selection**: Consider the trade-off between model complexity and performance
4. **Data Quality**: Focus on proteins with sufficient high-quality bioactivity data
