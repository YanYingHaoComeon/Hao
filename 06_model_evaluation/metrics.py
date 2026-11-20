# SPDX-License-Identifier: MIT
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from hao.common import module

def evaluate(y, p, epsilon):
    p = np.asarray(p, dtype=float)
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any(): raise ValueError("Invalid probabilities")
    answer = {"roc_auc": float(roc_auc_score(y, p)),
        "pr_auc_average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p))}
    answer.update(module("06_model_evaluation", "calibration").calibration(y, p, epsilon))
    return answer

def frozen_bootstrap_ci(y, p, iterations, seed):
    rng = np.random.default_rng(seed); y, p = np.asarray(y), np.asarray(p)
    rows = []
    for _ in range(iterations):
        idx = rng.integers(0, len(y), len(y))
        if np.unique(y[idx]).size == 2:
            rows.append([roc_auc_score(y[idx], p[idx]), average_precision_score(y[idx], p[idx]), brier_score_loss(y[idx], p[idx])])
    if len(rows) < 2: raise ValueError("Insufficient nondegenerate resamples")
    return {name: list(map(float, np.quantile(np.asarray(rows)[:, j], [.025, .975])))
        for j, name in enumerate(["roc_auc", "pr_auc_average_precision", "brier"])}
