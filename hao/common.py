# SPDX-License-Identifier: MIT
"""Configuration, module loading, and private-output helpers."""
from pathlib import Path
import importlib, json, os
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def module(folder, name):
    return importlib.import_module(folder + "." + name)

def load_config(path):
    cfg = json.loads(Path(path).read_text())
    schema = cfg["schema"]
    features = schema["continuous"] + schema["categorical"]
    if len(features) < 2 or len(features) != len(set(features)) or schema["outcome"] in features:
        raise ValueError("Specify unique predictors, separate from the outcome.")
    if not 0 < cfg["split"]["validation_fraction"] < 1:
        raise ValueError("validation_fraction must lie between zero and one.")
    if cfg["imputation"]["m"] < 1 or cfg["imputation"]["iterations"] < 1:
        raise ValueError("Positive imputation counts are required.")
    if not 0 < cfg["evaluation"]["clip"] < .5:
        raise ValueError("Invalid logit clipping epsilon.")
    if not cfg["models"]:
        raise ValueError("Select at least one model.")
    known = {"lr", "dt", "rf", "xgboost", "lightgbm", "svm", "ann"}
    if set(cfg["models"]) - known:
        raise ValueError("Unknown model name.")
    return cfg

def output_directory(value):
    path = Path(value).expanduser().resolve()
    if path == ROOT or ROOT in path.parents:
        raise ValueError("Keep analysis outputs outside this public repository.")
    path.mkdir(parents=True, exist_ok=True)
    return path

def write_json(path, value):
    def convert(x):
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value, default=convert, indent=2, allow_nan=False))
