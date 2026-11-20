# Methodology audit

Audit date: 2026-11-20.

| Procedure | Implemented definition and evidence |
|---|---|
| Outcome | Explicit configured positive label becomes internal 1. Reversing the configured label reverses the outcome in an executable test. |
| Missing data | MICE uses training-only initialization, ignored assessment rows, PMM donors and no outcome predictor. Perturbing assessment values left training imputations unchanged in the synthetic test. |
| Multiple imputations | Holdout fitting iterates over every completed dataset. A five-imputation test produced ten LR/DT fits. Performance mean/min/max are descriptive, not confidence intervals. |
| LR pooling | Rubin scalar pooling with a finite complete-data degrees-of-freedom adjustment is conditional on identical selected designs across all imputations. Different designs are explicitly not pooled. |
| Selection | LASSO uses stratified CV, binomial deviance and a configurable lambda rule. Boruta uses confirmed attributes with optional rough fix. Intersection equality was tested. No predictor is forced into an empty intersection. |
| Scaling | SVM/ANN scalers are fitted on training predictors only. Prediction does not update them. |
| Nested validation | Holdout cases are excluded. Imputation, feature selection and scaling run within each inner/outer training partition. Candidate ranking uses mean inner AUC. Repeated OOF observations are not independent participants. |
| Calibration | Offset score root is solved stably; slope is a separate logistic calibration fit. Constant predictions and separation are not assigned favorable slopes. Offset score residual and interval ordering were tested. |
| Thresholds/DCA | Classification uses probability >= threshold. DCA accepts only thresholds strictly between zero and one, and saves that exact array. |
| SHAP | The game explains the positive-class probability with an empirical background and identity link. Exact enumeration and kernel approximation agreed on a small tested problem; additivity was checked. |

Limits: structural tests do not establish predictive validity, convergence on a new dataset, missing-at-random assumptions or clinical utility. Outcome-free imputation is a specified deployment-compatible design choice, not a universal optimal imputation model. Users must inspect chain diagnostics and logged events. Coefficient pooling after data-driven selection does not remove selection uncertainty. Frozen-prediction bootstrap intervals omit development uncertainty. The nested default uses one imputation per partition unless configured otherwise. Exact SHAP cost grows exponentially; background substitution is not a causal interpretation.
