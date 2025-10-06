# QSAR Model Results Summary Report
==================================================

## Overall Statistics
- Total models created: 90
- Number of targets: 30
- Similarity thresholds: low, medium, high

## Performance Metrics Summary
- Average validation R²: 0.342 ± 0.172
- Average validation RMSE: 0.819 ± 0.206
- Average Q²: 0.258 ± 0.261

## Top 10 Best Performing Models (by Validation R²)
        target threshold   val_r2  val_rmse       q2  n_train  n_val
  NR1I3_medium    medium 0.602853  0.690081 0.512253       47     12
    NR1I3_high      high 0.602853  0.690081 0.512253       47     12
     NR1I3_low       low 0.602853  0.690081 0.512253       47     12
      CNR2_low       low 0.593913  0.730917 0.360033     3993    859
     CNR2_high      high 0.593913  0.730917 0.360033     3993    859
   CNR2_medium    medium 0.593913  0.730917 0.360033     3993    859
    SLC6A3_low       low 0.591088  0.687798 0.586752     2485    330
   HSD11B1_low       low 0.590879  0.687320 0.317744     2249    410
HSD11B1_medium    medium 0.590879  0.687320 0.317744     2249    410
  HSD11B1_high      high 0.590879  0.687320 0.317744     2249    410

## Model Quality Distribution
- Poor (R² < 0.3): 33 models (36.7%)
- Fair (0.3 ≤ R² < 0.5): 32 models (35.6%)
- Good (0.5 ≤ R² < 0.7): 25 models (27.8%)

## Performance by Similarity Threshold
### HIGH Threshold
- Average R²: 0.338 ± 0.172
- Average RMSE: 0.822 ± 0.208
- Average Q²: 0.254 ± 0.260

### MEDIUM Threshold
- Average R²: 0.338 ± 0.172
- Average RMSE: 0.821 ± 0.208
- Average Q²: 0.252 ± 0.261

### LOW Threshold
- Average R²: 0.348 ± 0.177
- Average RMSE: 0.814 ± 0.210
- Average Q²: 0.268 ± 0.269

## Top 10 Best Performing Targets (Average across thresholds)
target_name  val_r2_mean  val_r2_std  val_rmse_mean  q2_mean
      NR1I3        0.603       0.000          0.690    0.512
       CNR2        0.594       0.000          0.731    0.360
    HSD11B1        0.591       0.000          0.687    0.318
     SLC6A4        0.581       0.000          0.740    0.508
      CHRM3        0.575       0.000          0.998    0.571
      ADRB2        0.574       0.000          0.994    0.377
        XDH        0.526       0.020          0.642    0.505
    CYP2C19        0.519       0.000          0.392    0.422
     CYP2B6        0.463       0.000          0.543    0.105
     SLC6A3        0.424       0.145          0.812    0.267

## Recommendations
- 0 models achieved excellent performance (R² ≥ 0.7)
- 25 models achieved good performance (0.5 ≤ R² < 0.7)
- Consider additional feature engineering for good performing models
- 33 models showed poor performance (R² < 0.3) - consider alternative approaches

## Files Generated
- `performance_overview.png`: Overall performance metrics
- `target_performance_analysis.png`: Detailed target analysis
- `prediction_quality_analysis.png`: Prediction quality examples
- `threshold_comparison.png`: Similarity threshold comparison
- `model_quality_assessment.png`: Model quality distribution
- `target_performance_summary.csv`: Detailed target statistics
- `threshold_summary_stats.csv`: Threshold comparison statistics
- `model_quality_summary.csv`: Quality category statistics