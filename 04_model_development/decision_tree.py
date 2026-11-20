# SPDX-License-Identifier: MIT
"""Configured DecisionTreeClassifier factory."""
def create(parameters, seed):
    from sklearn.tree import DecisionTreeClassifier
    options = dict(parameters)
    options.setdefault("random_state", seed)
    return DecisionTreeClassifier(**options)
