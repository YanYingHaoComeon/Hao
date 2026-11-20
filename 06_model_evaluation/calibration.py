# SPDX-License-Identifier: MIT
"""Offset calibration-in-the-large and separate logistic calibration slope."""
import warnings
import numpy as np
import statsmodels.api as sm
from scipy.optimize import brentq
from scipy.special import expit, logit
from scipy.stats import norm

def calibration(y, p, epsilon):
    y, p = np.asarray(y), np.asarray(p)
    z = logit(np.clip(p, epsilon, 1-epsilon))
    if np.unique(y).size != 2: raise ValueError("Calibration needs both outcome classes")
    score = lambda a: float(np.sum(y-expit(z+a)))
    intercept = brentq(score, -100, 100, xtol=1e-12)
    q = expit(z+intercept); se = 1/np.sqrt(np.sum(q*(1-q)))
    critical = norm.ppf(.975)
    result = {"calibration_intercept": float(intercept),
        "intercept_ci_lower": float(intercept-critical*se),
        "intercept_ci_upper": float(intercept+critical*se),
        "offset_score_residual": score(intercept), "probability_clip": epsilon}
    if np.ptp(z) < 1e-12:
        result.update(calibration_slope=None, slope_ci_lower=None, slope_ci_upper=None,
            slope_status="not identifiable: constant predictions", calibration_warnings=[])
        return result
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = sm.GLM(y, sm.add_constant(z), family=sm.families.Binomial()).fit()
    messages = [str(w.message) for w in caught]
    reliable = fit.converged and not any("separation" in m.lower() for m in messages)
    ci = np.asarray(fit.conf_int())
    result.update(calibration_slope=float(fit.params[1]) if reliable else None,
        slope_ci_lower=float(ci[1, 0]) if reliable else None,
        slope_ci_upper=float(ci[1, 1]) if reliable else None,
        slope_status="estimated" if reliable else "not reliably estimable",
        calibration_warnings=messages)
    return result
