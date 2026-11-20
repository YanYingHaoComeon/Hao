# SPDX-License-Identifier: MIT
"""Explicit binary outcome mapping and training-only categorical encoding."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def read_data(path, schema):
    df = pd.read_csv(path, na_values=schema.get("na_values", ["NA", ""]))
    cols = schema["continuous"] + schema["categorical"]
    missing = set(cols + [schema["outcome"]]) - set(df.columns)
    if missing: raise ValueError("Missing configured columns: " + str(sorted(missing)))
    target = df[schema["outcome"]]
    if target.isna().any() or target.nunique() != 2:
        raise ValueError("A fully observed binary outcome is required.")
    if schema["positive_label"] not in target.unique():
        raise ValueError("Configured positive_label is absent.")
    y = (target == schema["positive_label"]).astype(int).to_numpy()
    x = df[cols].copy()
    for name in schema["continuous"]:
        x[name] = pd.to_numeric(x[name], errors="raise")
        if np.isinf(x[name]).any(): raise ValueError("Infinite predictor: " + name)
    return x, y

def split_indices(y, settings):
    return train_test_split(np.arange(len(y)), test_size=settings["validation_fraction"],
        stratify=y, random_state=settings["seed"])

def encode_partition(training, assessment, categorical):
    a, b = training.copy(), assessment.copy()
    mappings = {}
    for name in categorical:
        levels = list(pd.unique(a[name].dropna()))
        if not levels: raise ValueError("Training predictor entirely missing: " + name)
        if len(levels) > 2:
            raise ValueError("Binary categorical inputs are supported; pre-encode multilevel predictors explicitly.")
        unseen = set(b[name].dropna().unique()) - set(levels)
        if unseen: raise ValueError("Unseen assessment category in " + name)
        mapping = {v: i for i, v in enumerate(levels)}
        a[name], b[name] = a[name].map(mapping), b[name].map(mapping)
        mappings[name] = {str(k): v for k, v in mapping.items()}
    return a.astype(float), b.astype(float), mappings
