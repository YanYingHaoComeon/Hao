# SPDX-License-Identifier: MIT
"""Configured MLPClassifier factory."""
def create(parameters, seed):
    from sklearn.neural_network import MLPClassifier
    options = dict(parameters)
    options.setdefault("random_state", seed)
    return MLPClassifier(**options)
