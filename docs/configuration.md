# Configuration contract

`schema` selects the outcome column, its exact positive label, ordered continuous/binary categorical predictors and recognized missing tokens. Outcome mapping is applied before stratified splitting; outcome values never enter imputation predictors. All model arrays then use internal 0/1 with 1 equal to the configured positive label.

`split.seed`, `imputation.seed`, `selection.seed`, `training.seed`, `validation.seed`, `evaluation.bootstrap_seed` and `interpretation.seed` are separate controls. Nested partition imputation/selection seeds derive deterministically from the configured validation seed and global outer-fold/inner-fold indices. Each completed dataset adds its own selector seed offset. Exact SHAP does not sample coalitions; kernel/display operations use the configured interpretation seed where applicable.

`models` may contain any subset of the seven registered algorithms. `params` controls holdout fitting, while `candidates` is the ordered list considered by inner CV. There is no hidden tuning against the holdout cohort. Selection may be disabled or fixed explicitly. If the intersection is empty, the model is an explicit training-prior predictor; variables are never forced into the model to improve its AUC.

The joint MICE/selection interface requires at least two configured predictors. LASSO requires at least three folds and sufficient outcome-class counts in the current training partition. A prior-only model has no feature explanation plot; the CLI rejects that request explicitly.

All I/O paths are CLI arguments or `HAO_RSCRIPT`/`R_LIBS_USER` environment settings. No local account paths or credentials are embedded. Trusted optional passive MICE formulas run within R; do not accept untrusted configuration files as executable research input.
