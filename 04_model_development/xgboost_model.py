# SPDX-License-Identifier: MIT
"""Configured XGBClassifier factory."""
def create(parameters, seed):
    from xgboost import XGBClassifier
    options = dict(parameters)
    options.setdefault("random_state", seed)
    options.setdefault("eval_metric", "logloss")
    options.setdefault("n_jobs", 1)
    return XGBClassifier(**options)
