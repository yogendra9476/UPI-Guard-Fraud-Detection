"""Learnability audit: is the fraud label associated with any recorded attribute?"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

from . import config as C

AUDIT_CATEGORICAL = [
    "transaction_type", "merchant_category", "transaction_status", "sender_age_group",
    "receiver_age_group", "sender_state", "sender_bank", "receiver_bank", "device_type",
    "network_type", "day_of_week", "hour_of_day", "is_weekend",
]


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adj, 1.0)
    return out


def _mutual_info_bits(x, y) -> float:
    ct = pd.crosstab(x, y).to_numpy().astype(float)
    pxy = ct / ct.sum()
    px, py = pxy.sum(1, keepdims=True), pxy.sum(0, keepdims=True)
    nz = pxy > 0
    return float((pxy[nz] * np.log2(pxy[nz] / (px @ py)[nz])).sum())


def univariate_tests(df: pd.DataFrame, engineered: pd.DataFrame | None = None) -> pd.DataFrame:
    y = df[C.TARGET].to_numpy()
    rows = []
    for c in AUDIT_CATEGORICAL:
        ct = pd.crosstab(df[c], y)
        chi, p, dof, _ = chi2_contingency(ct)
        v = np.sqrt(chi / (ct.to_numpy().sum() * (min(ct.shape) - 1)))
        rates = ct[1] / ct.sum(1)
        rows.append({"feature": c, "test": "chi-square", "levels": int(ct.shape[0]),
                     "statistic": chi, "dof": int(dof), "p_value": p, "effect": v,
                     "effect_name": "Cramer's V", "mi_bits": _mutual_info_bits(df[c], y),
                     "min_rate_pct": 100 * rates.min(), "max_rate_pct": 100 * rates.max()})
    num = {"amount_inr": df["amount_inr"]}
    if engineered is not None:
        for c in ["amount_to_cat_median", "seg_count_1h", "seg_count_24h", "global_count_10m"]:
            num[c] = engineered[c]
    for c, s in num.items():
        s = np.asarray(s, dtype=float)
        u, p = mannwhitneyu(s[y == 1], s[y == 0])
        auc = u / ((y == 1).sum() * (y == 0).sum())
        rows.append({"feature": c, "test": "Mann-Whitney", "levels": np.nan, "statistic": u,
                     "dof": np.nan, "p_value": p, "effect": auc, "effect_name": "univariate AUC",
                     "mi_bits": np.nan, "min_rate_pct": np.nan, "max_rate_pct": np.nan})
    out = pd.DataFrame(rows)
    out["p_bh"] = benjamini_hochberg(out["p_value"].to_numpy())
    return out


def _design(df):
    enc = OneHotEncoder(handle_unknown="ignore")
    cats = [c for c in AUDIT_CATEGORICAL if c != "transaction_status"]
    Xc = enc.fit_transform(df[cats].astype(str))
    amt = csr_matrix(((np.log1p(df["amount_inr"]) - 6.5) / 1.2).to_numpy().reshape(-1, 1))
    return hstack([Xc, amt]).tocsr()


def cv_auc(X, y, seed=C.SEED, folds=3) -> float:
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    pred = np.zeros(len(y))
    for tr, te in skf.split(np.zeros(len(y)), y):
        m = LogisticRegression(C=0.1, class_weight="balanced", max_iter=500, solver="liblinear")
        m.fit(X[tr], y[tr])
        pred[te] = m.decision_function(X[te])
    return roc_auc_score(y, pred)


def label_permutation_test(df: pd.DataFrame, n_perm=C.N_PERMUTATIONS, seed=C.SEED):
    """Compare CV AUC on true labels with CV AUC on randomly permuted labels
    (Ojala & Garriga, 2010). Returns observed AUC, null distribution and p-value."""
    X = _design(df)
    y = df[C.TARGET].to_numpy()
    observed = cv_auc(X, y)
    rng = np.random.default_rng(seed)
    null = np.array([cv_auc(X, rng.permutation(y)) for _ in range(n_perm)])
    p = (1 + (null >= observed).sum()) / (1 + n_perm)
    return {"observed_cv_auc": float(observed), "null": null.tolist(),
            "null_mean": float(null.mean()), "null_sd": float(null.std()),
            "null_q95": float(np.quantile(null, 0.95)), "p_value": float(p), "n_perm": int(n_perm)}
