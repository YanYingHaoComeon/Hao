# Configuration audit

Audit date: 2026-11-20.

Executable tests verified renamed outcome/predictor columns, explicit positive-label reversal, deterministic split seeds, changed-seed split differences, model subset selection and all seven algorithm constructors. Model parameters and random states were inspected after construction. Training-only category encoding rejects unseen assessment categories and multilevel categorical inputs.

The example uses generic column names and contains no observations. Input paths, output paths and configuration paths are required CLI arguments. Rscript and custom R libraries can be selected through `HAO_RSCRIPT` and `R_LIBS_USER`.

Separate seeds control splitting, imputation, selection, fitting, validation, bootstrap and interpretation. Nested partition seeds are derived deterministically from the configured validation seed. Each imputation has its own selector offset. Exact SHAP enumerates coalitions; kernel SHAP uses the configured seed and sample budget.

Required researcher decisions remain explicit: outcome meaning, predictor types and units, recognized missing tokens, passive formulas, sample sizes/fold feasibility, model grids, iteration budgets, decision thresholds, background population and assessment plan. The framework requires at least two predictors, binary categorical predictors and a fully observed binary outcome. LASSO requires at least three folds with adequate class counts. No clinical defaults, study-specific sample counts or selected variable lists are distributed.

Outcome: tested configuration contracts passed. Arbitrary configurations and external datasets have not been exhaustively validated.
