# SPDX-License-Identifier: MIT
"""Shared training orchestration; assessment outcomes never enter preprocessing."""
from pathlib import Path
import copy, pickle
import numpy as np
import pandas as pd
from hao.common import module, write_json


def prepare(training, assessment, y, cfg, path):
    return module("02_missing_data_imputation", "bridge").prepare(training, assessment, y, cfg, path)


def fit_model(name, x, y, settings, cfg):
    factory = module("04_model_development", "registry").Predictor
    return factory(name, settings, cfg["training"]["seed"],
        cfg["training"]["standardize_svm_ann"]).fit(x, y)


def train_holdout(x, y, cfg, output):
    train, valid = module("01_data_preprocessing", "schema").split_indices(y, cfg["split"])
    batches, mappings = prepare(x.iloc[train], x.iloc[valid], y[train], cfg, output / "imputation")
    metrics = module("06_model_evaluation", "metrics")
    threshold = module("06_model_evaluation", "thresholds")
    dca = module("06_model_evaluation", "dca")
    rows, frequency, coefficients = [], [], []
    for i, (a, b, selected) in enumerate(batches, 1):
        frequency.extend({"imputation": i, "predictor": column, "selected": column in selected} for column in x.columns)
        for name, settings in cfg["models"].items():
            predictor = fit_model(name, a[selected], y[train], settings["params"], cfg)
            predictor.positive_label = cfg["schema"]["positive_label"]
            probabilities = predictor.predict(b)
            row = dict(imputation=i, model=name, predictors=len(selected),
                **metrics.evaluate(y[valid], probabilities, cfg["evaluation"]["clip"]))
            rows.append(row)
            folder = output / f"imputation_{i}" / name
            folder.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"row": valid, "outcome": y[valid], "probability": probabilities}).to_csv(folder / "predictions.csv", index=False)
            a[selected].to_csv(folder / "background.csv", index=False)
            b[selected].to_csv(folder / "explanation_input.csv", index=False)
            with (folder / "model.pkl").open("wb") as handle:
                pickle.dump(predictor, handle)
            threshold.classification(y[valid], probabilities, cfg["evaluation"]["thresholds"]).to_csv(folder / "thresholds.csv", index=False)
            dca.decision_curve(y[valid], probabilities, cfg["evaluation"]["dca_thresholds"]).to_csv(folder / "dca.csv", index=False)
            if name == "lr" and selected:
                for term, beta, se in zip(["Intercept"] + selected, predictor.model.params, predictor.model.bse):
                    coefficients.append(dict(imputation=i, term=term, estimate=float(beta), variance=float(se**2), complete_df=len(train)-len(selected)-1))
    write_json(output / "performance.json", rows)
    frame = pd.DataFrame(rows)
    frame.groupby("model")[["roc_auc", "pr_auc_average_precision", "brier"]].agg(["mean", "min", "max"]).to_csv(output / "descriptive_imputation_ranges.csv")
    pd.DataFrame(frequency).groupby("predictor").selected.agg(["sum", "mean"]).to_csv(output / "selection_frequency.csv")
    write_json(output / "category_encoding.json", mappings)
    pd.DataFrame({"row": np.r_[train, valid], "split": ["training"]*len(train)+["validation"]*len(valid)}).to_csv(output / "split.csv", index=False)
    if coefficients:
        data = pd.DataFrame(coefficients); data.to_csv(output / "lr_coefficients.csv", index=False)
        names = [set(g.term) for _, g in data.groupby("imputation")]
        if len(names) == cfg["imputation"]["m"] and all(n == names[0] for n in names) and len(names) > 1:
            pooling = module("06_model_evaluation", "pooling")
            pooled = [{"term": term, **pooling.rubin_scalar(g.estimate, g.variance, int(g.complete_df.iloc[0]))}
                for term, g in data.groupby("term")]
            write_json(output / "lr_rubin_pooling.json", pooled)
        else:
            write_json(output / "lr_pooling_status.json", {"status": "not pooled: selected designs differ or fewer than two imputations"})
    return rows
