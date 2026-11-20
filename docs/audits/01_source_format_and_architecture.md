# Source format and architecture audit

Audit date: 2026-11-20. Scope: the new Hao source tree only.

All executable procedures are actual `.py` or `.R` files. The seven numbered modules contain preprocessing, imputation, feature selection, model development, nested validation, evaluation and interpretation. The Python CLI coordinates the R imputation/selection entry point; individual algorithms have separate source files. Markdown documents describe execution and do not substitute for executable source.

The source is newly authored generic orchestration around installed library APIs. Earlier study scripts, commercial templates and their Git history are excluded. Algorithmic definitions include training-restricted PMM MICE, binomial-deviance LASSO, confirmed Boruta intersection, seven model families, partition-specific nested validation and positive-probability SHAP. This is reusable software, not an archive that reproduces an earlier study's particular results or parameters.

Dependencies flow from `run.py` through `hao/workflow.py` to the numbered modules. Numeric folder names are loaded through `importlib`, which was exercised in the tests. The R bridge passes repository-relative source locations and researcher-configured private input/output locations.

Outcome: source-format and module dependency checks passed. No research files are needed for the structural tests.
