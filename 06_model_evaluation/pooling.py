# SPDX-License-Identifier: MIT
"""Conditional fixed-design LR coefficient pooling; descriptive performance ranges."""
import numpy as np
from scipy.stats import t

def rubin_scalar(estimates, variances, complete_df):
    q, u = np.asarray(estimates), np.asarray(variances)
    if len(q) < 2 or len(q) != len(u) or complete_df <= 0: raise ValueError("Invalid pooling inputs")
    m = len(q); within = u.mean(); between = q.var(ddof=1)
    total = within+(1+1/m)*between
    lam = (1+1/m)*between/total if total else 0.
    old_df = (m-1)/(lam*lam) if lam else np.inf
    observed_df = (complete_df+1)/(complete_df+3)*complete_df*(1-lam)
    df = 1/(1/old_df+1/observed_df)
    half = t.ppf(.975, df)*np.sqrt(total)
    return dict(estimate=float(q.mean()), standard_error=float(np.sqrt(total)),
        lower=float(q.mean()-half), upper=float(q.mean()+half), df=float(df))
