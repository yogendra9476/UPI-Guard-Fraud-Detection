"""Shared, cached loaders and helpers for the Streamlit app."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from . import config as C
from .data import load_raw
from .features import add_velocity_features

A = C.ARTIFACTS
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Random Forest": "random_forest.joblib",
    "LightGBM": "lightgbm.joblib",
    "MLP": "mlp.joblib",
}
PALETTE = {
    "Prevalence baseline": "#9e9e9e", "Logistic Regression": "#1f77b4",
    "Random Forest": "#2ca02c", "LightGBM": "#ff7f0e", "MLP": "#9467bd", "1D-CNN": "#d62728",
}


def artifacts_ready() -> bool:
    return (A / "manifest.json").exists()


def require_artifacts():
    if not artifacts_ready():
        st.error("No trained artifacts found. Run `python run_pipeline.py` in the project "
                 "folder first (about 45 minutes), then reload this page.")
        st.stop()


@st.cache_data(show_spinner=False)
def load_json(name):
    return json.loads((A / name).read_text())


@st.cache_data(show_spinner=False)
def load_csv(name):
    try:
        return pd.read_csv(A / name)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


@st.cache_data(show_spinner="Loading 250,000 transactions ...")
def load_dataset():
    return load_raw()


@st.cache_resource(show_spinner="Loading models ...")
def load_models():
    fb = joblib.load(A / "models" / "feature_builder.joblib")
    cals = joblib.load(A / "models" / "calibrators.joblib")
    models = {n: joblib.load(A / "models" / f) for n, f in MODEL_FILES.items()
              if (A / "models" / f).exists()}
    return fb, models, cals


def score_frame(raw: pd.DataFrame, compute_velocity: bool) -> pd.DataFrame:
    """Score raw transactions (dataset schema, renamed columns) with every model."""
    fb, models, cals = load_models()
    df = raw.copy()
    if compute_velocity:
        df = df.sort_values("timestamp", kind="mergesort")
        df = add_velocity_features(df).loc[raw.index]
    X = fb.transform(df)
    out = pd.DataFrame(index=raw.index)
    for name, m in models.items():
        s = m.predict_proba(X)[:, 1]
        out[name] = cals[name].transform(s) if name in cals else s
    return out


def wilson(k, n, z=1.96):
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return c - h, c + h


def inr(x: float) -> str:
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e7:
        return f"{sign}₹{x / 1e7:,.2f} Cr"
    if x >= 1e5:
        return f"{sign}₹{x / 1e5:,.2f} L"
    return f"{sign}₹{x:,.0f}"


def model_names(metrics: dict, include_baseline=True):
    order = ["Prevalence baseline", "Logistic Regression", "Random Forest", "LightGBM", "MLP", "1D-CNN"]
    return [m for m in order if m in metrics and (include_baseline or m != "Prevalence baseline")]
