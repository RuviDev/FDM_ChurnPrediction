# Model Card - Credit Card Customer Churn

**Intended use:** Prioritize retention actions for customers with elevated churn risk. Not for credit underwriting decisions.

## Data
- Source: Kaggle BankChurners (public)
- Size: ~10k rows, ~23 columns (mix of numeric & categorical)
- Target: `Attrition_Flag` -> `Churn` (1 if Attrited Customer)
- Preprocessing: drop `CLIENTNUM` and the two helper `Naive_Bayes_Classifier_*` columns

## Pipeline
- Imputation: numeric=median, categorical=most_frequent
- Scaling: StandardScaler (numeric)
- Encoding: OneHotEncoder(handle_unknown='ignore')
- Model: XGBoost (class imbalance via `scale_pos_weight`)
- Calibration: Isotonic (CalibratedClassifierCV)

## Metrics (holdout)
See app/metrics.json for exact numbers and the Insights tab for ROC/PR/Calibration plots.
- We optimize for decision usefulness (PR-AUC & thresholded F1), not just ROC.

## Threshold & ROI
- Choose threshold via Expected Value = `assumed treatment effectiveness` x `P(churn)` x `value_retained` - `contact_cost`

## Fairness & Governance
- Sensitive attributes: consider excluding `Gender`
- Documented training pipeline, calibration, and validation
- Monitor drift; review threshold & calibration quarterly
