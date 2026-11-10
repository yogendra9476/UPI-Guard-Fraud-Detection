"""Pre-debit feature engineering.

Rule: every feature must be computable at the moment the payer presses *Pay*,
before the debit is authorised. Concretely
  * no authorisation outcome (``transaction_status`` is dropped),
  * statistics that need a reference distribution (category medians, p95) are
    fitted on the training window only,
  * velocity counts look strictly backwards in time (window ends just before
    the current transaction) and never use labels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

AGE_ORDER = {"18-25": 0, "26-35": 1, "36-45": 2, "46-55": 3, "56+": 4}

NUMERIC_FEATURES = [
    "log_amount", "amount_to_cat_median", "amount_cat_pctile", "high_amount_cat",
    "hour_of_day", "hour_sin", "hour_cos", "is_night", "is_weekend",
    "same_bank", "age_gap", "seg_count_1h", "seg_count_24h", "global_count_10m",
]
CATEGORICAL_FEATURES = list(C.PRE_DEBIT_CATEGORICAL)
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

FEATURE_GROUPS = {
    "Transaction": ["log_amount", "transaction_type", "merchant_category",
                    "amount_to_cat_median", "amount_cat_pctile", "high_amount_cat"],
    "Temporal": ["hour_of_day", "hour_sin", "hour_cos", "is_night", "is_weekend", "day_of_week"],
    "Party/demographic": ["sender_age_group", "receiver_age_group", "age_gap",
                          "sender_state", "sender_bank", "receiver_bank", "same_bank"],
    "Device/network": ["device_type", "network_type"],
    "Velocity (past-only)": ["seg_count_1h", "seg_count_24h", "global_count_10m"],
}

SEGMENT_KEYS = ["sender_bank", "sender_state", "sender_age_group", "device_type"]


def _past_window_count(times_s: np.ndarray, window_s: float) -> np.ndarray:
    """Number of earlier events in (t - window, t) for a sorted time array."""
    left = np.searchsorted(times_s, times_s - window_s, side="right")
    here = np.searchsorted(times_s, times_s, side="left")  # excludes ties at t
    return (here - left).astype(np.int32)


def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-looking counts. Requires df sorted by timestamp."""
    out = df.copy()
    t = (out["timestamp"] - pd.Timestamp("2024-01-01")).dt.total_seconds().to_numpy()
    out["global_count_10m"] = _past_window_count(t, 600)
    seg = out[SEGMENT_KEYS].astype(str).agg("|".join, axis=1)
    c1 = np.zeros(len(out), dtype=np.int32)
    c24 = np.zeros(len(out), dtype=np.int32)
    for _, idx in seg.groupby(seg).indices.items():
        tt = t[idx]  # already sorted because df is sorted
        c1[idx] = _past_window_count(tt, 3600)
        c24[idx] = _past_window_count(tt, 86400)
    out["seg_count_1h"] = c1
    out["seg_count_24h"] = c24
    return out


class FeatureBuilder:
    """Fits reference statistics on the training window, then transforms any frame."""

    def fit(self, train_df: pd.DataFrame) -> "FeatureBuilder":
        g = train_df.groupby("merchant_category")["amount_inr"]
        self.cat_median_ = g.median().to_dict()
        self.cat_p95_ = g.quantile(0.95).to_dict()
        self.cat_sorted_ = {k: np.sort(v.to_numpy()) for k, v in g}
        self.global_median_ = float(train_df["amount_inr"].median())
        self.global_p95_ = float(train_df["amount_inr"].quantile(0.95))
        self.global_sorted_ = np.sort(train_df["amount_inr"].to_numpy())
        self.velocity_medians_ = {
            c: float(train_df[c].median()) if c in train_df else 0.0
            for c in ["seg_count_1h", "seg_count_24h", "global_count_10m"]
        }
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=df.index)
        amt = df["amount_inr"].astype(float)
        cat = df["merchant_category"].astype(str)
        med = cat.map(self.cat_median_).fillna(self.global_median_).astype(float)
        p95 = cat.map(self.cat_p95_).fillna(self.global_p95_).astype(float)
        X["log_amount"] = np.log1p(amt)
        X["amount_to_cat_median"] = amt / med
        pct = np.empty(len(df))
        a_np, c_np = amt.to_numpy(), cat.to_numpy()
        for c in np.unique(c_np):
            m = c_np == c
            ref = self.cat_sorted_.get(c, self.global_sorted_)
            pct[m] = np.searchsorted(ref, a_np[m], side="right") / len(ref)
        X["amount_cat_pctile"] = pct
        X["high_amount_cat"] = (amt > p95).astype(int)
        h = df["hour_of_day"].astype(int)
        X["hour_of_day"] = h
        X["hour_sin"] = np.sin(2 * np.pi * h / 24)
        X["hour_cos"] = np.cos(2 * np.pi * h / 24)
        X["is_night"] = h.between(0, 5).astype(int)
        X["is_weekend"] = df["is_weekend"].astype(int)
        X["same_bank"] = (df["sender_bank"] == df["receiver_bank"]).astype(int)
        X["age_gap"] = (df["sender_age_group"].map(AGE_ORDER)
                        - df["receiver_age_group"].map(AGE_ORDER)).abs().fillna(0).astype(int)
        for c in ["seg_count_1h", "seg_count_24h", "global_count_10m"]:
            X[c] = df[c].astype(float) if c in df else self.velocity_medians_[c]
        for c in CATEGORICAL_FEATURES:
            X[c] = df[c].astype(str)
        return X[ALL_FEATURES]
