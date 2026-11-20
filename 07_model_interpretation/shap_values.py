# SPDX-License-Identifier: MIT
"""Identity-link positive-class empirical-background Shapley game."""
import math
import numpy as np
import pandas as pd


def explain(predictor, background, cases, options):
    if list(background.columns) != list(cases.columns): raise ValueError("Feature order differs")
    p = len(background.columns)
    base = float(np.mean(predictor.predict(background)))
    if options["method"] == "kernel":
        import shap
        np.random.seed(options["seed"])
        explainer = shap.KernelExplainer(lambda a: predictor.predict(pd.DataFrame(a, columns=background.columns)), background)
        values = np.asarray(explainer.shap_values(cases, nsamples=options["kernel_samples"], l1_reg=0))
    elif options["method"] == "exact":
        if p > options["max_exact_features"]: raise ValueError("Exact enumeration exceeds configured feature budget")
        values = np.zeros((len(cases), p))
        for row, case in enumerate(cases.to_numpy()):
            game = np.empty(1 << p)
            for mask in range(1 << p):
                combinations = background.copy()
                for j, name in enumerate(background.columns):
                    if mask & (1 << j): combinations[name] = case[j]
                game[mask] = np.mean(predictor.predict(combinations))
            for j in range(p):
                for mask in range(1 << p):
                    if not mask & (1 << j):
                        size = mask.bit_count()
                        weight = math.factorial(size)*math.factorial(p-size-1)/math.factorial(p)
                        values[row, j] += weight*(game[mask | (1 << j)]-game[mask])
    else: raise ValueError("Unknown SHAP method")
    error = np.max(np.abs(base+values.sum(axis=1)-predictor.predict(cases)))
    return values, base, float(error)
