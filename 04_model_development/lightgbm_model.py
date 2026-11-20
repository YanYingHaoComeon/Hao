# SPDX-License-Identifier: MIT
"""Configured LGBMClassifier factory."""
def create(parameters, seed):
    from lightgbm import LGBMClassifier
    options = dict(parameters)
    options.setdefault("random_state", seed)
    options.setdefault("verbosity", -1)
    options.setdefault("n_jobs", 1)
    return LGBMClassifier(**options)
