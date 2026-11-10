"""Metrics, uncertainty, calibration and INR cost analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, norm
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             precision_recall_curve, roc_auc_score, roc_curve)

from . import config as C


# --------------------------------------------------------------------------- ranking
def auc_pvalue(y, s) -> float:
    """One-sided Mann-Whitney test of H0: AUC = 0.5 vs H1: AUC > 0.5."""
    y = np.asarray(y)
    return float(mannwhitneyu(s[y == 1], s[y == 0], alternative="greater").pvalue)


def bootstrap_ci(y, s, fn, n=C.N_BOOTSTRAP, seed=C.SEED, alpha=0.05):
    """Stratified percentile bootstrap (resample frauds and legitimates separately)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = np.empty(n)
    for b in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        vals[b] = fn(y[idx], s[idx])
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def auc_null_se(n_pos: int, n_neg: int) -> float:
    """Standard error of AUC under H0 (Hanley & McNeil / Mann-Whitney variance)."""
    return float(np.sqrt((n_pos + n_neg + 1) / (12.0 * n_pos * n_neg)))


def minimum_detectable_auc(n_pos, n_neg, alpha=0.05, power=0.80) -> float:
    return 0.5 + (norm.ppf(1 - alpha) + norm.ppf(power)) * auc_null_se(n_pos, n_neg)


# --------------------------------------------------------------------------- calibration
def ece_quantile(y, p, bins=10) -> float:
    """Expected calibration error with equal-mass bins (robust at 0.2 % prevalence)."""
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(p, kind="mergesort")
    err = 0.0
    for chunk in np.array_split(order, bins):
        if len(chunk):
            err += len(chunk) / len(p) * abs(p[chunk].mean() - y[chunk].mean())
    return float(err)


def reliability_table(y, p, bins=10) -> pd.DataFrame:
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(p, kind="mergesort")
    rows = []
    for i, chunk in enumerate(np.array_split(order, bins)):
        rows.append({"bin": i + 1, "mean_predicted": p[chunk].mean(),
                     "observed_rate": y[chunk].mean(), "n": len(chunk)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- thresholds
def confusion(y, p, thr) -> dict:
    y = np.asarray(y)
    pred = np.asarray(p) >= thr
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    prev = (tp + fn) / len(y)
    # Bayes identity used to cross-check precision (reviewer item, V1 error)
    denom = rec * prev + (1 - spec) * (1 - prev)
    prec_identity = rec * prev / denom if denom else 0.0
    return {"threshold": float(thr), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": prec, "recall": rec, "specificity": spec, "f1": f1,
            "alert_rate": float(pred.mean()), "precision_identity": prec_identity,
            "lift": prec / prev if prev else 0.0}


def threshold_for_alert_rate(p_valid, rate) -> float:
    return float(np.quantile(p_valid, 1 - rate))


# --------------------------------------------------------------------------- INR cost model
def net_benefit(y, p, amount, thr, cost_review=C.COST_REVIEW_INR,
                effectiveness=C.STEP_UP_EFFECTIVENESS) -> dict:
    """Net INR benefit of flagging p >= thr for step-up, relative to 'flag nothing'.

    benefit = effectiveness * sum(amount of flagged frauds) - cost_review * (#flagged)
    """
    y, p, amount = np.asarray(y), np.asarray(p), np.asarray(amount, dtype=float)
    flag = p >= thr
    saved = effectiveness * amount[flag & (y == 1)].sum()
    cost = cost_review * flag.sum()
    return {"threshold": float(thr), "flagged": int(flag.sum()),
            "frauds_caught": int((flag & (y == 1)).sum()),
            "fraud_loss_total": float(amount[y == 1].sum()),
            "loss_prevented": float(saved), "review_cost": float(cost),
            "net_benefit": float(saved - cost)}


def best_cost_threshold(y, p, amount, cost_review=C.COST_REVIEW_INR):
    """Pick the threshold maximising net benefit; 'never flag' (+inf) is a valid option."""
    cands = np.unique(np.quantile(p, np.linspace(0.5, 1.0, 201)))
    best = {"threshold": float("inf"), "net_benefit": 0.0}
    for t in cands:
        nb = net_benefit(y, p, amount, t, cost_review)
        if nb["net_benefit"] > best["net_benefit"]:
            best = nb
    return best["threshold"]


def break_even_precision(cost_review, mean_fraud_amount, effectiveness=C.STEP_UP_EFFECTIVENESS):
    """A flag pays for itself only if precision * effectiveness * E[amount|fraud] > cost."""
    return cost_review / (effectiveness * mean_fraud_amount)


# --------------------------------------------------------------------------- summary
def summarize(y, p, n_boot=C.N_BOOTSTRAP) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    prev = y.mean()
    auc = roc_auc_score(y, p)
    ap = average_precision_score(y, p)
    brier = brier_score_loss(y, p)
    brier_ref = prev * (1 - prev)
    k1 = max(1, int(round(0.01 * len(p))))
    k10 = max(1, int(round(0.10 * len(p))))
    order = np.argsort(-p, kind="mergesort")
    return {
        "roc_auc": auc, "roc_auc_ci": bootstrap_ci(y, p, roc_auc_score, n_boot),
        "auc_p_value": auc_pvalue(y, p),
        "pr_auc": ap, "pr_auc_ci": bootstrap_ci(y, p, average_precision_score, n_boot),
        "pr_auc_baseline": prev,
        "brier": brier, "brier_skill": 1 - brier / brier_ref,
        "log_loss": log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)),
        "ece": ece_quantile(y, p),
        "lift_top1": y[order[:k1]].mean() / prev, "lift_top10": y[order[:k10]].mean() / prev,
        "recall_top1": y[order[:k1]].sum() / y.sum(), "recall_top10": y[order[:k10]].sum() / y.sum(),
    }


def curve_points(y, p, kind="roc", n=40):
    """Down-sampled curve coordinates (for plots and pgfplots)."""
    if kind == "roc":
        fpr, tpr, _ = roc_curve(y, p)
        grid = np.linspace(0, 1, n)
        return grid, np.interp(grid, fpr, tpr)
    prec, rec, _ = precision_recall_curve(y, p)
    rec, prec = rec[::-1], prec[::-1]
    grid = np.linspace(0, 1, n)
    return grid, np.interp(grid, rec, prec)
