"""Render report figures (PNG) from ./artifacts.  python make_figures.py"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as C

A, F = C.ARTIFACTS, C.FIGURES
F.mkdir(parents=True, exist_ok=True)
COL = {"Prevalence baseline": "#9e9e9e", "Logistic Regression": "#1f77b4", "Random Forest": "#2ca02c",
       "LightGBM": "#ff7f0e", "MLP": "#9467bd", "1D-CNN": "#d62728"}
plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "savefig.bbox": "tight"})


def load(n):
    return json.loads((A / n).read_text())


m, curves = load("metrics_test.json"), load("curves_test.json")
names = [n for n in COL if n in m]

# ROC
fig, ax = plt.subplots(figsize=(3.4, 3.0))
for n in names:
    ax.plot(*curves[n]["roc"], label=f"{n} ({m[n]['roc_auc']:.3f})", color=COL[n], lw=1.2)
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Test ROC (Nov-Dec 2024)")
ax.legend(fontsize=6.5, loc="lower right")
fig.savefig(F / "roc_test.png")

# Permutation null
perm = load("audit_permutation.json")
fig, ax = plt.subplots(figsize=(3.4, 2.4))
ax.hist(perm["null"], bins=25, color="#9ecae1", edgecolor="white")
ax.axvline(perm["observed_cv_auc"], color="#d62728", lw=2, label=f"true labels ({perm['observed_cv_auc']:.3f})")
ax.set(xlabel="3-fold CV ROC-AUC", ylabel="Permutations", title=f"Label-permutation test (p = {perm['p_value']:.2f})")
ax.legend(fontsize=7)
fig.savefig(F / "permutation_null.png")

# Positive control
pc = pd.read_csv(A / "positive_control.csv")
agg = pc.groupby(["strength", "model"])["test_auc"].agg(["mean", "std"]).reset_index()
fig, ax = plt.subplots(figsize=(3.4, 2.6))
orc = pc.groupby("strength")["oracle_auc"].mean()
ax.plot(orc.index, orc.values, "k--", label="Oracle")
for n, g in agg.groupby("model"):
    ax.errorbar(g["strength"], g["mean"], yerr=g["std"].fillna(0), label=n, color=COL[n], marker="o", ms=3, capsize=2)
ax.set(xlabel="Planted signal strength $s$", ylabel="Test ROC-AUC", title="Positive control (simulated labels)")
ax.legend(fontsize=7)
fig.savefig(F / "positive_control.png")

# Net benefit curve
pred = pd.read_csv(A / "predictions_test.csv")
y, amt = pred["y"].to_numpy(), pred["amount_inr"].to_numpy(float)
fig, ax = plt.subplots(figsize=(3.4, 2.6))
rates = np.linspace(0.001, 0.2, 80)
for n in [n for n in names if n != "Prevalence baseline"]:
    order = np.argsort(-pred[n].to_numpy(), kind="mergesort")
    nb = [amt[order[:int(r * len(y))]][y[order[:int(r * len(y))]] == 1].sum() - C.COST_REVIEW_INR * int(r * len(y))
          for r in rates]
    ax.plot(100 * rates, np.array(nb) / 1e3, color=COL[n], lw=1.1, label=n)
rand = (amt[y == 1].sum() / len(y) - C.COST_REVIEW_INR) * rates * len(y)
ax.plot(100 * rates, rand / 1e3, "k--", lw=0.9, label="Random flagging")
ax.axhline(0, color="black", lw=0.6)
ax.set(xlabel="Transactions flagged (%)", ylabel="Net benefit (INR '000)", title="Net benefit, review cost INR 25")
ax.legend(fontsize=6.5)
fig.savefig(F / "net_benefit.png")
print("Figures written to", F)
