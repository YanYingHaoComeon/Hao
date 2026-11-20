# SPDX-License-Identifier: MIT
"""Configured RandomForestClassifier factory."""
def create(parameters, seed):
    from sklearn.ensemble import RandomForestClassifier
    options = dict(parameters)
    options.setdefault("random_state", seed)
    options.setdefault("n_jobs", 1)
    return RandomForestClassifier(**options)
