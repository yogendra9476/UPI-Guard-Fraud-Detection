"""Data loading, schema validation, provenance fingerprinting and splitting."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C

EXPECTED_COLUMNS = [
    "transaction id", "timestamp", "transaction type", "merchant_category",
    "amount (INR)", "transaction_status", "sender_age_group",
    "receiver_age_group", "sender_state", "sender_bank", "receiver_bank",
    "device_type", "network_type", "fraud_flag", "hour_of_day", "day_of_week",
    "is_weekend",
]


def sha256_of_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(path=C.DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")
    df = df.rename(columns=C.RAW_RENAME)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["timestamp", "transaction_id"], kind="mergesort").reset_index(drop=True)
    return df


def validate(df: pd.DataFrame) -> dict:
    """Integrity checks reported in the data card and the paper."""
    checks = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_ids": int(df["transaction_id"].duplicated().sum()),
        "fraud_count": int(df[C.TARGET].sum()),
        "fraud_rate": float(df[C.TARGET].mean()),
        "date_min": str(df["timestamp"].min()),
        "date_max": str(df["timestamp"].max()),
        "amount_min": int(df["amount_inr"].min()),
        "amount_max": int(df["amount_inr"].max()),
        "amount_median": float(df["amount_inr"].median()),
        "hour_consistent_with_timestamp": bool((df["timestamp"].dt.hour == df["hour_of_day"]).all()),
        "weekday_consistent_with_timestamp": bool(
            (df["timestamp"].dt.day_name() == df["day_of_week"]).all()),
        "weekend_flag_consistent": bool(
            ((df["timestamp"].dt.dayofweek >= 5).astype(int) == df["is_weekend"]).all()),
        "has_customer_identifier": False,
    }
    return checks


@dataclass
class Split:
    train: np.ndarray
    valid: np.ndarray
    test: np.ndarray


def time_split(df: pd.DataFrame) -> Split:
    ts = df["timestamp"]
    tr = (ts < C.TRAIN_END).to_numpy()
    va = ((ts >= C.TRAIN_END) & (ts < C.VALID_END)).to_numpy()
    te = (ts >= C.VALID_END).to_numpy()
    assert tr.sum() + va.sum() + te.sum() == len(df)
    return Split(np.where(tr)[0], np.where(va)[0], np.where(te)[0])


def split_table(df: pd.DataFrame, split: Split) -> pd.DataFrame:
    rows = []
    for name, idx in [("Train", split.train), ("Validation", split.valid), ("Test", split.test)]:
        part = df.iloc[idx]
        rows.append({
            "split": name,
            "start": part["timestamp"].min().strftime("%Y-%m-%d"),
            "end": part["timestamp"].max().strftime("%Y-%m-%d"),
            "rows": int(len(part)),
            "frauds": int(part[C.TARGET].sum()),
            "legitimate": int(len(part) - part[C.TARGET].sum()),
            "fraud_rate_pct": round(100 * part[C.TARGET].mean(), 4),
        })
    total = {"split": "Total", "start": rows[0]["start"], "end": rows[-1]["end"],
             "rows": sum(r["rows"] for r in rows), "frauds": sum(r["frauds"] for r in rows),
             "legitimate": sum(r["legitimate"] for r in rows)}
    total["fraud_rate_pct"] = round(100 * total["frauds"] / total["rows"], 4)
    rows.append(total)
    return pd.DataFrame(rows)
