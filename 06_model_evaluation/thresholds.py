# SPDX-License-Identifier: MIT
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from statsmodels.stats.proportion import proportion_confint

def classification(y, probability, thresholds):
    result = []
    for cutoff in thresholds:
        if not 0 <= cutoff <= 1: raise ValueError("Invalid classification threshold")
        tn, fp, fn, tp = map(int, confusion_matrix(y, np.asarray(probability)>=cutoff, labels=[0,1]).ravel())
        row = dict(threshold=cutoff, tn=tn, fp=fp, fn=fn, tp=tp)
        for name, a, b in [("sensitivity",tp,tp+fn),("specificity",tn,tn+fp),("ppv",tp,tp+fp),("npv",tn,tn+fn)]:
            row[name] = a/b if b else np.nan
            lo, hi = proportion_confint(a, b, method="wilson") if b else (np.nan, np.nan)
            row[name+"_lower"], row[name+"_upper"] = lo, hi
        row["accuracy"] = (tp+tn)/(tp+tn+fp+fn)
        row["f1"] = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan
        result.append(row)
    return pd.DataFrame(result)
