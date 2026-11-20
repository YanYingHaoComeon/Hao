# SPDX-License-Identifier: MIT
import numpy as np
import pandas as pd

def decision_curve(y, probability, thresholds):
    y, p = np.asarray(y), np.asarray(probability)
    grid = np.asarray(thresholds, dtype=float)
    if not len(grid) or not np.isfinite(grid).all() or ((grid<=0)|(grid>=1)).any():
        raise ValueError("DCA thresholds must be strictly between zero and one")
    rows = []
    for t in grid:
        positive = p >= t
        benefit = np.mean(positive & (y==1))-np.mean(positive & (y==0))*t/(1-t)
        rows.append([t, benefit, y.mean()-(1-y.mean())*t/(1-t), 0.])
    return pd.DataFrame(rows, columns=["threshold","model","treat_all","treat_none"])
