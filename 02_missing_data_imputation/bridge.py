# SPDX-License-Identifier: MIT
from pathlib import Path
import json, os, subprocess
import pandas as pd
from hao.common import ROOT, module, write_json

def prepare(training, assessment, y, cfg, directory):
    path = Path(directory); path.mkdir(parents=True, exist_ok=True)
    a, b, mappings = module("01_data_preprocessing", "schema").encode_partition(
        training, assessment, cfg["schema"]["categorical"])
    a.to_csv(path / "training_input.csv", index=False)
    b.to_csv(path / "assessment_input.csv", index=False)
    pd.DataFrame({"target": y}).to_csv(path / "training_labels.csv", index=False)
    write_json(path / "config.json", cfg)
    command = [os.environ.get("HAO_RSCRIPT", "Rscript"), "--vanilla",
        str(ROOT / "02_missing_data_imputation/01_mice.R"),
        str(path / "training_input.csv"), str(path / "assessment_input.csv"),
        str(path / "training_labels.csv"), str(path / "config.json"), str(path), str(ROOT)]
    try:
        with (path / "imputation.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    except subprocess.CalledProcessError as error:
        details = (path / "imputation.log").read_text(errors="replace")[-3000:]
        raise RuntimeError("R imputation/selection failed; private diagnostics:\n" + details) from error
    choices = json.loads((path / "selection.json").read_text())
    batches = []
    for i in range(1, cfg["imputation"]["m"] + 1):
        batches.append((pd.read_csv(path / f"train_{i}.csv"),
            pd.read_csv(path / f"assessment_{i}.csv"), choices[i-1]["retained"]))
    return batches, mappings
