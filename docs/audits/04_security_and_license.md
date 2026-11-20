# Security and license audit

Audit date: 2026-11-20.

The publication candidate was inspected as UTF-8 source, configuration and documentation. The static audit rejects data/model/figure/log artifacts, source symlinks, binary content, personal account paths and credential literals. Python and JSON files are parsed. Hidden publication metadata and workflow files are included in the inspection. Generated test observations and outputs remain in temporary/private directories and are not distributed.

Excluded categories: patient records, intermediate/completed datasets, fitted models, individual predictions, OOF/SHAP results, research figures, run logs, original scripts and original Git history. The CLI refuses analysis outputs inside the public source tree. Pickle loading is limited by documentation to trusted user-generated model artifacts; it is not safe for untrusted files. Optional passive R formulas also require trusted configuration.

MIT was chosen for the newly authored Hao source because it permits academic citation, maintenance and reuse while retaining the copyright and license notice. `LICENSE`, `CITATION.cff` and `.zenodo.json` agree. Restricted commercial-template source and promotional/contact text were excluded, not relicensed. Separately installed dependencies retain their own terms; no dependency implementation or font is vendored.

The GitHub workflow uses only its repository-scoped built-in token with job-specific contents permission. No personal token is included. Release creation is gated on the quality job. Publication and Zenodo archiving must be verified separately; source metadata contains no guessed DOI.

Outcome: local source audit passed. Automated pattern checks reduce accidental disclosure risk and are supplemented by source review; they are not a proof that every future contribution is safe.
