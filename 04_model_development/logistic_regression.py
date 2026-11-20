# SPDX-License-Identifier: MIT
"""Unpenalized maximum-likelihood logistic regression."""
import statsmodels.api as sm

def fit(x, y, parameters, seed):
    return sm.Logit(y, sm.add_constant(x, has_constant="add")).fit(disp=False, **parameters)

def predict(model, x):
    return model.predict(sm.add_constant(x, has_constant="add"))
