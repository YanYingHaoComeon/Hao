# Hao

Hao is a disease-independent binary prediction framework with real, modular Python (`.py`) and R (`.R`) sources. It supports training-restricted MICE, LASSO/Boruta feature intersection, seven model families, repeated nested cross-validation, probability calibration, threshold analysis, decision curves and SHAP explanations. It contains no research records, pretrained models or study-specific results.

## Seven modules

| Module | Implementation |
|---|---|
| `01_data_preprocessing/` | Configured schema, positive-class mapping, stratified split and training-only category encoding |
| `02_missing_data_imputation/` | R MICE program and Python bridge; assessment rows are ignored when fitting imputation models |
| `03_feature_selection/` | Standalone R LASSO and Boruta procedures, called within each training partition |
| `04_model_development/` | Separate LR, DT, RF, XGBoost, LightGBM, SVM and ANN source files |
| `05_model_validation/` | Repeated nested CV, including partition-specific imputation, selection and scaling |
| `06_model_evaluation/` | ROC-AUC, average precision, Brier score, calibration, classification thresholds, DCA and pooling |
| `07_model_interpretation/` | Exact empirical-background and Kernel SHAP; generic importance, beeswarm, dependence and waterfall plots |

## Installation

Python 3.12 and R 4.5 were used for local structural tests. Install Python packages in an isolated environment:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Restore the R dependency record with an installed `renv`, or install the recorded versions in an isolated R library. Required direct R packages are mice, glmnet, Boruta, ranger and jsonlite. `renv.lock` records their observed dependency closure. No package installation runs implicitly inside the analysis programs.

```sh
Rscript -e 'renv::restore(lockfile="renv.lock", prompt=FALSE)'
```

When using a custom R library, export `R_LIBS_USER` before invoking Python. Set `HAO_RSCRIPT` if the Rscript executable is not on PATH. Compatible compiler/OpenMP libraries may be needed for the platform-specific packages.

## Configuration and private inputs

Copy and edit `config/example.json`. Configure your CSV column names, explicit positive label, predictor types, split fraction, all random seeds, MICE settings, feature-selection mode, chosen models, candidate grids and evaluation thresholds. Nothing infers the positive class from alphabetical order. The example column names are generic and no sample patient rows are distributed.

Use numeric continuous predictors and binary categorical predictors. Training categories are encoded from training data only; unseen assessment categories are rejected. For multilevel nominal features, supply a predeclared encoding rather than assuming that integer category codes are metric quantities. The outcome must be fully observed with exactly two labels. Numeric sentinel values are not automatically interpreted as missing. No automatic outlier removal is performed.

Inputs and all generated artifacts must stay private. The command-line entry point requires an output directory outside this source repository. Only load fitted pickle objects you created and trust.

## Execute

```sh
python run.py train --config config/example.json --data YOUR_PRIVATE_CSV --output YOUR_PRIVATE_OUTPUT
python run.py nested --config config/example.json --data YOUR_PRIVATE_CSV --output YOUR_PRIVATE_NESTED_OUTPUT
python run.py explain --config config/example.json --model-dir YOUR_TRUSTED_MODEL_DIRECTORY --output YOUR_PRIVATE_EXPLANATIONS
```

The uppercase arguments describe user-supplied local paths; they are not provided datasets. `train` fits each configured model in **every completed dataset**, producing separate performance estimates and descriptive mean/min/max summaries. LR coefficients are Rubin-pooled only when the selected designs match across all completed datasets. Otherwise the designs remain separate. Across-imputation ranges are not confidence intervals.

The R program is also an executable source entry point. Its arguments are predictor-only training CSV, predictor-only assessment CSV, training-label CSV (column `target`, internal 0/1), JSON configuration, private output directory and repository root:

```sh
Rscript --vanilla 02_missing_data_imputation/01_mice.R TRAIN_CSV ASSESSMENT_CSV TRAIN_LABEL_CSV CONFIG_JSON PRIVATE_OUTPUT REPOSITORY_ROOT
```

The Python bridge constructs these inputs within each appropriate training partition. Do not pass assessment outcomes to this R interface. LASSO and Boruta are separate reusable R functions sourced by this entry point.

## Statistical definitions

- LR uses unpenalized maximum likelihood. SVM and ANN scaling is fitted only in their training partition. Random seeds and constructor parameters are configured explicitly.
- MICE uses training-only initialization, PMM with configurable donors and iteration/dataset counts, ignored assessment rows and no outcome predictor. Optional passive predictor formulas must be supplied by the researcher in trusted configuration; none are invented by the framework.
- Feature selection supports `intersection`, `fixed` (set `selection.predictors`) or `none`. LASSO uses stratified binomial-deviance CV and a configured lambda rule. Boruta uses confirmed attributes after optional TentativeRoughFix.
- Nested CV is restricted to the configured development split. Preprocessing and selection occur inside inner and outer training partitions. Candidate selection uses mean inner ROC-AUC with a first-listed-candidate tie rule. `validation.imputations` defaults to one per partition; larger values average predictions within each assessment fold. Repeated OOF records reuse participants and are not independent cases.
- Calibration-in-the-large fixes the logit-prediction coefficient at one and solves the offset score equation stably. Calibration slope is a separate two-parameter logistic fit. Constant predictions are marked unidentifiable; separation warnings are retained rather than replaced with favorable slopes.
- Classification uses probability >= threshold. DCA calculation and plotting must use the same saved threshold array, strictly inside (0,1). Frozen-prediction percentile bootstrap intervals do not include model-development uncertainty.
- SHAP explains the internally encoded positive outcome probability, on the identity-link empirical-background game. Exact enumeration is limited by the configured feature budget. Kernel mode is a separately configurable approximation; explanation plots do not fit a new predictive model.

## Testing and audits

```sh
python -m unittest discover -s tests -v
HAO_TEST_R=1 python -m unittest discover -s tests -v
python scripts/static_audit.py
```

All tests generate generic synthetic observations in temporary directories. The first command explicitly skips R integration unless enabled; skipped tests are not described as passed. The second executes real MICE, LASSO, Boruta, five-imputation training and small nested CV when the R dependencies are installed. See the five reports in `docs/audits/` for the exact local test scope and limitations. Structural testing is not a clinical benchmark, and optimization warnings should be monitored with real inputs.

## Source provenance and license

The framework is newly organized generic source using the documented APIs of independently installed scientific packages. Prior project-specific scripts, commercial-template text, original research Git history, contacts, variables and artifacts are **not redistributed**. Algorithms are expressed as reusable library calls, not as a relabeled copy of restricted code. Dependencies retain their own licenses; no dependency source is vendored.

Hao's project-authored source is licensed under **MIT**, chosen for straightforward academic and software reuse with preserved copyright/license notices. `LICENSE`, `CITATION.cff` and `.zenodo.json` specify the same license.

## Citation and archived release

[Version 0.1.0](https://github.com/YanYingHaoComeon/Hao/releases/tag/v0.1.0) is publicly archived on [Zenodo](https://zenodo.org/records/24393539). The registered DOI resolves to the published record, and the archive's 53 files were verified against the release. The DOI was added to the repository citation metadata after archiving; the release tag remains unchanged.

YanYingHaoComeon. (2026). *Hao: Configurable binary prediction workflows* (Version 0.1.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.24393539
