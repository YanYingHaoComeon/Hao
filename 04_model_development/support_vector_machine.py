# SPDX-License-Identifier: MIT
"""Configured SVC factory."""
def create(parameters, seed):
    from sklearn.svm import SVC
    options = dict(parameters)
    options.setdefault("random_state", seed)
    options["probability"] = True
    return SVC(**options)
