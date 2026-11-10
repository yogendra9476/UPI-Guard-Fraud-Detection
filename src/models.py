"""Model zoo: preprocessing pipelines, hyper-parameter grids and calibration."""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from . import config as C
from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None


def onehot_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])


def ordinal_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", "passthrough", NUMERIC_FEATURES),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
         CATEGORICAL_FEATURES),
    ])


def make_model(name: str, **params):
    s = C.SEED
    if name == "Prevalence baseline":
        return Pipeline([("pre", ordinal_preprocessor()),
                         ("clf", DummyClassifier(strategy="prior"))])
    if name == "Logistic Regression":
        return Pipeline([("pre", onehot_preprocessor()),
                         ("clf", LogisticRegression(C=params.get("C", 0.1), class_weight="balanced",
                                                    max_iter=3000, random_state=s))])
    if name == "Random Forest":
        return Pipeline([("pre", ordinal_preprocessor()),
                         ("clf", RandomForestClassifier(
                             n_estimators=params.get("n_estimators", 300),
                             min_samples_leaf=params.get("min_samples_leaf", 50),
                             max_features="sqrt", class_weight="balanced_subsample",
                             n_jobs=-1, random_state=s))])
    if name == "LightGBM":
        return Pipeline([("pre", ordinal_preprocessor()),
                         ("clf", lgb.LGBMClassifier(
                             n_estimators=params.get("n_estimators", 300),
                             learning_rate=0.03, num_leaves=params.get("num_leaves", 15),
                             min_child_samples=100, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, reg_lambda=1.0,
                             scale_pos_weight=params.get("scale_pos_weight", 1.0),
                             random_state=s, n_jobs=-1, verbose=-1))])
    if name == "MLP":
        return Pipeline([("pre", onehot_preprocessor()),
                         ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), alpha=params.get("alpha", 1e-3),
                                               batch_size=1024, learning_rate_init=1e-3, max_iter=60,
                                               early_stopping=True, n_iter_no_change=6,
                                               random_state=s))])
    raise ValueError(name)


# Small, pre-declared grids searched by 5-fold stratified CV on the training window only.
GRIDS = {
    "Logistic Regression": [{"C": 0.01}, {"C": 0.1}, {"C": 1.0}],
    "Random Forest": [{"min_samples_leaf": 25}, {"min_samples_leaf": 100}],
    "LightGBM": [{"num_leaves": 7, "scale_pos_weight": 1.0}, {"num_leaves": 31, "scale_pos_weight": 1.0},
                 {"num_leaves": 7, "scale_pos_weight": 25.0}, {"num_leaves": 31, "scale_pos_weight": 25.0}],
    "MLP": [{"alpha": 1e-4}, {"alpha": 1e-2}],
}
MODEL_ORDER = ["Prevalence baseline", "Logistic Regression", "Random Forest", "LightGBM", "MLP", "1D-CNN"]


class PlattCalibrator:
    """Monotone logistic recalibration of a score, fitted on the validation window."""

    def fit(self, score, y):
        z = self._logit(score).reshape(-1, 1)
        self.lr_ = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
        return self

    def transform(self, score):
        return self.lr_.predict_proba(self._logit(score).reshape(-1, 1))[:, 1]

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))
