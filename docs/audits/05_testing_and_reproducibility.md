# Testing and reproducibility audit

Audit date: 2026-11-20. Local Python 3.12 and R 4.5.1 were used. Direct Python versions are pinned in `requirements.txt` and their exact releases were verified against the official package index. `renv.lock` records the observed R dependency closure for mice, glmnet, Boruta, ranger and jsonlite. A clean cross-platform restoration of the entire R closure has not been tested.

The full local test command enabled R integration with `HAO_TEST_R=1`. **15 tests ran and passed**, including all four R integration tests. Only generic synthetic data were used; no study analyses were rerun. R source parsing and lockfile parsing passed. Python source parsing and the CLI help entry point are checked separately by the source audit and release checks.

Coverage includes column/outcome mapping, binary category encoding, split seeds, seven algorithm fits with finite positive-class probabilities, training-only scaling, explicit prior-only fallback, stable offset calibration, thresholds, DCA boundary rejection, seeded bootstrap, Rubin pooling, exact/kernel SHAP agreement, additivity and four explanation plot exports. R integration covers two completed datasets with observed-value preservation, assessment perturbation isolation, executable LASSO/Boruta intersection, configured passive formulas, five-imputation LR/DT training and small nested CV with OOF coverage checks.

The small nested test used two outer folds, one repeat and two inner folds; selectors remained inside partitions. This is an execution test, not a performance estimate or a test of the much larger default grid. The test suite does not run all seven models across five imputations or full default nested resampling.

An initial invalid two-fold LASSO test exposed glmnet's minimum-fold requirement. Validation and the test were corrected to require at least three folds, then the complete suite passed. A deliberately short ANN structural fit emitted a convergence warning; finite output and configuration were tested, not optimization adequacy. SHAP emitted upstream plotting deprecation warnings. These warnings are not suppressed as proof of convergence.

The default CI command runs Python tests and explicitly **skips** R integration unless enabled with R dependencies available. Such skips are not passes. Local private test logs are retained outside the repository; no logs or result artifacts are uploaded. Reproduction on a new platform may require compiler/OpenMP setup and additional convergence checks.
