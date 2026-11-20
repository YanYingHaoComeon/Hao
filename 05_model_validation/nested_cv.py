# SPDX-License-Identifier: MIT
"""Repeated nested CV with preprocessing and selection inside each partition."""
import copy
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from hao.common import module, write_json
from hao.workflow import prepare, fit_model


def validate(x, y, cfg, output):
    # The fixed holdout is excluded from development-workflow resampling.
    training, holdout = module("01_data_preprocessing", "schema").split_indices(y, cfg["split"])
    settings = cfg["validation"]
    outer = RepeatedStratifiedKFold(n_splits=settings["outer_folds"], n_repeats=settings["outer_repeats"], random_state=settings["seed"])
    probabilities, parameters = [], []
    for fold, (a, b) in enumerate(outer.split(x.iloc[training], y[training]), 1):
        train, test = training[a], training[b]
        seed = settings["seed"] + 100*fold
        local = copy.deepcopy(cfg)
        local["imputation"]["m"] = settings.get("imputations", 1)
        inner = StratifiedKFold(n_splits=settings["inner_folds"], shuffle=True, random_state=seed)
        splits = []
        for j, (ia, ib) in enumerate(inner.split(x.iloc[train], y[train]), 1):
            local["imputation"]["seed"] = seed+j
            local["selection"]["seed"] = seed+j
            batches, _ = prepare(x.iloc[train[ia]], x.iloc[train[ib]], y[train[ia]], local,
                output / f"outer_{fold}_inner_{j}")
            splits.append((batches, y[train[ia]], y[train[ib]]))
        local["imputation"]["seed"] = seed+10
        local["selection"]["seed"] = seed+10
        final, _ = prepare(x.iloc[train], x.iloc[test], y[train], local, output / f"outer_{fold}_final")
        for name, model_cfg in cfg["models"].items():
            scores = []
            candidates = model_cfg.get("candidates", [model_cfg["params"]])
            if not candidates: raise ValueError("Empty candidate grid")
            for candidate in candidates:
                values = []
                for batches, iy, ivy in splits:
                    for a, b, selected in batches:
                        predictor = fit_model(name, a[selected], iy, candidate, cfg)
                        values.append(roc_auc_score(ivy, predictor.predict(b)))
                scores.append(float(np.mean(values)))
            best = int(np.argmax(scores))  # deterministic first-candidate tie rule
            predictions = []
            for a, b, selected in final:
                predictor = fit_model(name, a[selected], y[train], candidates[best], cfg)
                predictions.append(predictor.predict(b))
            p = np.mean(predictions, axis=0)
            repeat = (fold-1)//settings["outer_folds"]+1
            parameters.append(dict(global_fold=fold, repeat=repeat, model=name,
                selected_candidate=best, parameters=candidates[best], inner_scores=scores))
            probabilities.extend(dict(row=int(idx), repeat=repeat, global_fold=fold, model=name,
                outcome=int(yy), probability=float(pp)) for idx, yy, pp in zip(test, y[test], p))
    frame = pd.DataFrame(probabilities)
    frame.to_csv(output / "oof_predictions.csv", index=False)
    write_json(output / "selected_parameters.json", parameters)
    evaluation = module("06_model_evaluation", "metrics")
    rows = []
    for (repeat, name), group in frame.groupby(["repeat", "model"]):
        if len(group) != len(training) or group.row.nunique() != len(training):
            raise RuntimeError("OOF coverage failure")
        rows.append(dict(repeat=int(repeat), model=name, n=len(training),
            **evaluation.evaluate(group.outcome, group.probability, cfg["evaluation"]["clip"])))
    write_json(output / "repeat_performance.json", rows)
    return rows
