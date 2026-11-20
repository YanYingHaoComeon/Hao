# SPDX-License-Identifier: MIT
import numpy as np
from sklearn.preprocessing import StandardScaler
from hao.common import module

FILES = {"lr": "logistic_regression", "dt": "decision_tree", "rf": "random_forest",
    "xgboost": "xgboost_model", "lightgbm": "lightgbm_model",
    "svm": "support_vector_machine", "ann": "neural_network"}

class Predictor:
    def __init__(self, name, parameters, seed, standardize=True):
        self.name, self.parameters, self.seed = name, dict(parameters), seed
        self.standardize, self.scaler = standardize, None

    def fit(self, x, y):
        self.columns = list(x.columns); self.prior = float(np.mean(y))
        if not self.columns: return self
        values = x.to_numpy(dtype=float)
        if self.standardize and self.name in {"svm", "ann"}:
            self.scaler = StandardScaler().fit(values)
            values = self.scaler.transform(values)
        algorithm = module("04_model_development", FILES[self.name])
        if self.name == "lr": self.model = algorithm.fit(values, y, self.parameters, self.seed)
        else: self.model = algorithm.create(self.parameters, self.seed).fit(values, y)
        return self

    def predict(self, x):
        if not self.columns: return np.full(len(x), self.prior)
        values = x[self.columns].to_numpy(dtype=float)
        if self.scaler is not None: values = self.scaler.transform(values)
        if self.name == "lr":
            return np.asarray(module("04_model_development", "logistic_regression").predict(self.model, values))
        index = list(self.model.classes_).index(1)
        return self.model.predict_proba(values)[:, index]
