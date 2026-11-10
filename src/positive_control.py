"""Positive control: semi-synthetic labels with a *known* pre-debit fraud signal.

All features are the real recorded attributes; only the label is replaced by
    y* ~ Bernoulli( sigmoid( b0 + s * z ) ),
where z is a fixed, documented risk score and b0 is solved so that the
prevalence equals the real 0.192 %. s = 0 reproduces the real situation
(label independent of features); larger s plants a stronger signal.
These labels are SIMULATED and are reported only as a validation of the
pipeline's ability to detect signal - never as evidence about real UPI fraud.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from . import config as C

RISK_SCORE_TERMS = [
    ("Night hour (00:00-05:59)", 1.0),
    ("Amount above category p95 (train-fitted)", 1.0),
    ("Web device", 0.7),
    ("Night x high amount (interaction)", 1.5),
    ("Age gap >= 3 brackets", 0.6),
    ("Segment velocity in top 5 % (past 1 h)", 0.8),
    ("P2P transfer above INR 5,000", 0.6),
]


def risk_score(X: pd.DataFrame, raw: pd.DataFrame, seg_p95: float) -> np.ndarray:
    night = X["is_night"].to_numpy()
    high = X["high_amount_cat"].to_numpy()
    web = (X["device_type"] == "Web").to_numpy().astype(float)
    gap = (X["age_gap"] >= 3).to_numpy().astype(float)
    vel = (X["seg_count_1h"] >= seg_p95).to_numpy().astype(float)
    p2p = ((raw["transaction_type"] == "P2P") & (raw["amount_inr"] > 5000)).to_numpy().astype(float)
    w = dict(RISK_SCORE_TERMS)
    z = (w["Night hour (00:00-05:59)"] * night
         + w["Amount above category p95 (train-fitted)"] * high
         + w["Web device"] * web
         + w["Night x high amount (interaction)"] * night * high
         + w["Age gap >= 3 brackets"] * gap
         + w["Segment velocity in top 5 % (past 1 h)"] * vel
         + w["P2P transfer above INR 5,000"] * p2p)
    return (z - z.mean()) / z.std()


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def simulate_labels(z: np.ndarray, strength: float, prevalence: float, seed: int):
    b0 = brentq(lambda b: _sigmoid(b + strength * z).mean() - prevalence, -30, 10)
    p_true = _sigmoid(b0 + strength * z)
    rng = np.random.default_rng(seed)
    y = (rng.random(len(z)) < p_true).astype(int)
    return y, p_true
