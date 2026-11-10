"""Fill report/template.tex with numbers from ./artifacts -> report/main.tex.

Every number in the paper comes from this script, so the report cannot drift from the run.
    python build_report.py
"""
import json
import re

import numpy as np
import pandas as pd

from src import config as C

A = C.ARTIFACTS
R = C.ROOT / "report"


def J(n):
    return json.loads((A / n).read_text())


def f(x, d=3):
    return f"{x:.{d}f}"


def pval(p):
    return r"$<$0.001" if p < 0.001 else f"{p:.3f}"


def tex_escape(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#").replace(">=", r"$\geq$")


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0
    for k, i in enumerate(order):
        run = max(run, (m - k) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def coords(xs, ys, d=4):
    return " ".join(f"({x:.{d}f},{y:.{d}f})" for x, y in zip(xs, ys))


V = {}
man, met, perm, power = J("manifest.json"), J("metrics_test.json"), J("audit_permutation.json"), J("power.json")
lat, curves, cv = J("latency.json"), J("curves_test.json"), J("cv_results.json")
split = pd.read_csv(A / "split_table.csv")
uni = pd.read_csv(A / "audit_univariate.csv")
conf = pd.read_csv(A / "confusion_test.csv")
cost = pd.read_csv(A / "cost_test.csv")
sens = pd.read_csv(A / "cost_sensitivity.csv")
pc = pd.read_csv(A / "positive_control.csv")
abl = pd.read_csv(A / "ablation.csv")
imp = pd.read_csv(A / "permutation_importance.csv")
cnn = pd.read_csv(A / "cnn_order.csv") if (A / "cnn_order.csv").stat().st_size > 5 else pd.DataFrame()
pred = pd.read_csv(A / "predictions_test.csv")

ck = man["checks"]
V["NROWS"] = f"{ck['rows']:,}"
V["NFRAUD"] = f"{ck['fraud_count']}"
V["PREV"] = f"{100 * ck['fraud_rate']:.3f}"
V["SHA"] = man["dataset_sha256"]
V["SHASHORT"] = man["dataset_sha256"][:16]
V["RUNTIME"] = f"{man['runtime_seconds'] / 60:.1f}"
V["RUNDATE"] = man["run_utc"][:10]
for k, v in man["versions"].items():
    V["VER_" + k.upper().replace("-", "")] = str(v)
V["NFEAT"] = str(len(man["features"]))
V["AMTMED"] = f"{ck['amount_median']:,.0f}"

for _, r in split.iterrows():
    k = {"Train": "TR", "Validation": "VA", "Test": "TE", "Total": "TOT"}[r["split"]]
    V[k + "N"] = f"{r['rows']:,}"
    V[k + "F"] = f"{r['frauds']}"
    V[k + "L"] = f"{r['legitimate']:,}"
    V[k + "RATE"] = f"{r['fraud_rate_pct']:.3f}"
    V[k + "START"], V[k + "END"] = r["start"], r["end"]

# ---- audit
V["NTESTS"] = str(len(uni))
V["NRAWSIG"] = str(int((uni.p_value < 0.05).sum()))
V["NBHSIG"] = str(int((uni.p_bh < 0.05).sum()))
V["MINPBH"] = f(uni.p_bh.min(), 3)
V["MAXCRAMER"] = f(uni.loc[uni.test == "chi-square", "effect"].max(), 4)
rawsig = uni[uni.p_value < 0.05]
V["RAWSIGNAMES"] = ", ".join(r"\texttt{" + tex_escape(x) + "}" for x in rawsig.feature) or "none"
rows = []
for _, r in uni.iterrows():
    stat = f"{r.statistic:.2f} ({int(r.dof)})" if r.test == "chi-square" else f"AUC$_1$={r.effect:.3f}"
    eff = f"{r.effect:.4f}" if r.test == "chi-square" else "--"
    rng = f"{r.min_rate_pct:.3f}--{r.max_rate_pct:.3f}" if r.test == "chi-square" else "--"
    rows.append(rf"\texttt{{{tex_escape(r.feature)}}} & {stat} & {rng} & {eff} & {pval(r.p_value)} & {pval(r.p_bh)} \\")
V["AUDITROWS"] = "\n".join(rows)
V["PERMOBS"] = f(perm["observed_cv_auc"], 4)
V["PERMMEAN"] = f(perm["null_mean"], 4)
V["PERMSD"] = f(perm["null_sd"], 4)
V["PERMQ95"] = f(perm["null_q95"], 4)
V["PERMP"] = f(perm["p_value"], 3)
V["PERMN"] = str(perm["n_perm"])
hist, edges = np.histogram(perm["null"], bins=20)
V["PERMHIST"] = coords((edges[:-1] + edges[1:]) / 2, hist, 4)
V["PERMHISTMAX"] = str(int(hist.max() + 3))
V["SE0"] = f(power["auc_null_se"], 4)
V["AUCCRIT"] = f(power["auc_critical_5pct"], 3)
V["MDA"] = f(power["min_detectable_auc_80power"], 3)

# ---- models
order = [m for m in ["Prevalence baseline", "Logistic Regression", "Random Forest", "LightGBM", "MLP", "1D-CNN"]
         if m in met]
learned = [m for m in order if m != "Prevalence baseline"]
ph = dict(zip(learned, holm([met[m]["auc_p_value"] for m in learned])))
rows = []
for m in order:
    r = met[m]
    cvs = f"{r['cv_auc_mean']:.3f}$\\pm${r['cv_auc_sd']:.3f}" if "cv_auc_mean" in r else "--"
    ci = f"[{r['roc_auc_ci'][0]:.3f}, {r['roc_auc_ci'][1]:.3f}]"
    pr = f"{pval(r['auc_p_value'])} / {pval(ph[m])}" if m in ph else "--"
    lift_txt = "1.00" if m == "Prevalence baseline" else f"{r['lift_top10']:.2f}"
    rows.append(f"{m} & {cvs} & {r['roc_auc']:.3f} {ci} & {pr} & {1000 * r['pr_auc']:.2f} & "
                f"{r['brier_skill'] * 1e3:+.2f} & {1000 * r['ece']:.2f} & {lift_txt} \\\\")
V["MODELROWS"] = "\n".join(rows)
best = max(learned, key=lambda m: met[m]["roc_auc"])
V["BEST"] = best
V["BESTAUC"] = f(met[best]["roc_auc"])
V["BESTLO"], V["BESTHI"] = f(met[best]["roc_auc_ci"][0]), f(met[best]["roc_auc_ci"][1])
V["BESTP"] = pval(met[best]["auc_p_value"])
V["BESTPHOLM"] = pval(ph[best])
V["BESTCV"] = f(met[best]["cv_auc_mean"])
V["BESTVAL"] = f(met[best]["valid_auc"])
V["AUCMIN"] = f(min(met[m]["roc_auc"] for m in learned))
V["AUCMAX"] = f(max(met[m]["roc_auc"] for m in learned))
V["NHOLMSIG"] = str(int(sum(v < 0.05 for v in ph.values())))
V["NRAWPSIG"] = str(int(sum(met[m]["auc_p_value"] < 0.05 for m in learned)))
V["PRBASE"] = f(1000 * met[order[0]]["pr_auc_baseline"], 2)
V["CICONTAINS"] = str(sum(met[m]["roc_auc_ci"][0] <= 0.5 <= met[m]["roc_auc_ci"][1] for m in learned))
V["NLEARNED"] = str(len(learned))
V["LIFTMIN"] = f(min(met[m]["lift_top10"] for m in learned), 2)
V["LIFTMAX"] = f(max(met[m]["lift_top10"] for m in learned), 2)
V["ECERAWBEST"] = f(1000 * met[best]["ece_raw"], 2)
V["ECECALBEST"] = f(1000 * met[best]["ece"], 2)
V["ECERAWLR"] = f(1000 * met["Logistic Regression"]["ece_raw"], 1)
V["ECECALLR"] = f(1000 * met["Logistic Regression"]["ece"], 2)
V["CVSELECT"] = "; ".join(f"{m}: " + ", ".join(f"{k}={v}" for k, v in cv["selected"][m]["params"].items())
                          for m in cv["selected"])
V["CVSELECT"] = tex_escape(V["CVSELECT"])

pal = {"Logistic Regression": "blue", "Random Forest": "green!60!black", "LightGBM": "orange",
       "MLP": "violet", "1D-CNN": "red"}
plots = []
for m in learned:
    fpr, tpr = curves[m]["roc"]
    plots.append(rf"\addplot[{pal[m]}, thick] coordinates {{{coords(fpr, tpr, 3)}}};"
                 + "\n" + rf"\addlegendentry{{{m} ({met[m]['roc_auc']:.3f})}}")
V["ROCPLOTS"] = "\n".join(plots)

# ---- confusion
ab = conf[conf.operating_point == "Alert budget 1%"]
rows = []
for _, r in ab.iterrows():
    if r.model == "Prevalence baseline":
        continue
    rows.append(f"{r.model} & {r.TP} & {r.FP} & {r.FN} & {r.TN:,} & {100 * r.precision:.2f} & "
                f"{100 * r.recall:.2f} & {100 * r.specificity:.2f} & {r.lift:.2f} \\\\")
V["CONFROWS"] = "\n".join(rows)
rb = ab[ab.model == best].iloc[0]
V["CB_TP"], V["CB_FP"], V["CB_FN"], V["CB_TN"] = str(rb.TP), str(rb.FP), str(rb.FN), f"{rb.TN:,}"
V["CB_PREC"], V["CB_REC"], V["CB_SPEC"] = f(100 * rb.precision, 2), f(100 * rb.recall, 2), f(100 * rb.specificity, 2)
learned_ab = ab[ab.model != "Prevalence baseline"]
V["MAXTP"] = str(int(learned_ab.TP.max()))
V["MAXPREC"] = f(100 * learned_ab.precision.max(), 2)
V["CB_PRECID"] = f(100 * rb.precision_identity, 2)
V["CB_FLAGS"] = str(rb.TP + rb.FP)
co = conf[(conf.operating_point != "Alert budget 1%") & (conf.model != "Prevalence baseline")]
V["NCOSTNOFLAG"] = str(int(np.isinf(co.threshold).sum()))
cst = cost[(cost.operating_point != "Alert budget 1%") & (cost.model != "Prevalence baseline")]
rows = []
for _, r in cst.iterrows():
    thr = r"$\infty$" if np.isinf(r.threshold) else f"{r.threshold:.5f}"
    rows.append(f"{r.model} & {thr} & {r.flagged} & {r.frauds_caught} & {r.loss_prevented:,.0f} & "
                f"{r.review_cost:,.0f} & {r.net_benefit:,.0f} \\\\")
V["COSTROWS"] = "\n".join(rows)
cab = cost[(cost.operating_point == "Alert budget 1%") & (cost.model == best)].iloc[0]
V["AB_NET"] = f"{cab.net_benefit:,.0f}"
V["AB_SAVED"] = f"{cab.loss_prevented:,.0f}"
V["AB_COST"] = f"{cab.review_cost:,.0f}"

# ---- business
y, amt = pred["y"].to_numpy(), pred["amount_inr"].to_numpy(float)
V["TESTLOSS"] = f"{amt[y == 1].sum():,.0f}"
V["TESTMEANF"] = f"{amt[y == 1].mean():,.0f}"
V["MEANF"] = f"{man['mean_fraud_amount_train_inr']:,.0f}"
V["BEPREC"] = f(100 * man["break_even_precision"], 2)
V["BELIFT"] = f(man["break_even_precision"] / ck["fraud_rate"], 1)
V["COSTREV"] = f"{C.COST_REVIEW_INR:.0f}"
rows = []
for c_, g in sens.groupby("cost_review"):
    bep = g.break_even_precision.iloc[0]
    bestrow = g.loc[g.net_benefit.idxmax()]
    nflag0 = int(np.isinf(g.threshold).sum())
    rows.append(f"{c_:.0f} & {100 * bep:.2f} & {bep / ck['fraud_rate']:.1f} & {nflag0}/{len(g)} & "
                f"{bestrow.model} & {bestrow.net_benefit:,.0f} \\\\")
V["SENSROWS"] = "\n".join(rows)
V["SENSMAX"] = f"{sens.net_benefit.max():,.0f}"
rates = np.linspace(0.002, 0.2, 34)
nbplots = []
for m in learned:
    o = np.argsort(-pred[m].to_numpy(), kind="mergesort")
    ys = [(amt[o[:int(r * len(y))]][y[o[:int(r * len(y))]] == 1].sum() - C.COST_REVIEW_INR * int(r * len(y))) / 1e3
          for r in rates]
    nbplots.append(rf"\addplot[{pal[m]}, thick] coordinates {{{coords(100 * rates, ys, 2)}}};"
                   + "\n" + rf"\addlegendentry{{{m}}}")
rand = (amt[y == 1].sum() / len(y) - C.COST_REVIEW_INR) * rates * len(y) / 1e3
nbplots.append(rf"\addplot[black, dashed] coordinates {{{coords(100 * rates, rand, 2)}}};"
               + "\n" + r"\addlegendentry{Random flagging}")
V["NBPLOTS"] = "\n".join(nbplots)

# ---- positive control
g = pc.groupby(["strength", "model"]).agg(auc=("test_auc", "mean"), sd=("test_auc", "std"),
                                          nb=("net_benefit_inr", "mean"), orc=("oracle_auc", "mean"),
                                          nf=("test_frauds", "mean")).reset_index()
rows, pcplots = [], []
for s, gg in g.groupby("strength"):
    d = gg.set_index("model")
    cells = " & ".join(f"{d.loc[m, 'auc']:.3f}$\\pm${d.loc[m, 'sd']:.3f}" for m in
                       ["Logistic Regression", "Random Forest", "LightGBM"])
    rows.append(f"{s:.2f} & {d['orc'].iloc[0]:.3f} & {cells} & {d.loc['LightGBM', 'nb']:,.0f} \\\\")
V["PCROWS"] = "\n".join(rows)
orc = g.groupby("strength")["orc"].mean()
pcplots.append(rf"\addplot[black, dashed, thick] coordinates {{{coords(orc.index, orc.values, 3)}}};"
               + "\n" + r"\addlegendentry{Oracle}")
for m in ["Logistic Regression", "Random Forest", "LightGBM"]:
    gg = g[g.model == m]
    pcplots.append(rf"\addplot[{pal[m]}, thick, mark=*, mark size=1.2pt] coordinates {{{coords(gg.strength, gg.auc, 3)}}};"
                   + "\n" + rf"\addlegendentry{{{m}}}")
V["PCPLOTS"] = "\n".join(pcplots)
s0 = g[g.strength == 0]
V["PC0MIN"], V["PC0MAX"] = f(s0.auc.min()), f(s0.auc.max())
s1 = g[g.strength == C.PC_CNN_STRENGTH]
V["PCREF"] = f"{C.PC_CNN_STRENGTH}"
V["PCREFORC"] = f(s1.orc.mean())
V["PCREFLGB"] = f(s1[s1.model == "LightGBM"].auc.iloc[0])
V["PCREFLR"] = f(s1[s1.model == "Logistic Regression"].auc.iloc[0])
V["PCREFRF"] = f(s1[s1.model == "Random Forest"].auc.iloc[0])
V["PCREFNB"] = f"{s1[s1.model == 'LightGBM'].nb.iloc[0]:,.0f}"
s05 = g[g.strength == 0.5]
V["PC05ORC"] = f(s05.orc.mean())
V["PC05LGB"] = f(s05[s05.model == "LightGBM"].auc.iloc[0])
V["PC05NB"] = f"{s05[s05.model == 'LightGBM'].nb.iloc[0]:,.0f}"
V["PCTERMS"] = "\n".join(f"{tex_escape(t)} & {w:.1f} \\\\" for t, w in man["risk_score_terms"])

# ---- ablation / importance
rows = []
for d_ in abl.dropped.unique():
    a = abl[abl.dropped == d_].set_index("setting")["test_auc"]
    rows.append(f"{tex_escape(d_)} & {a.iloc[0]:.3f} & {a.iloc[1]:.3f} \\\\")
V["ABLROWS"] = "\n".join(rows)
pci = imp[imp.setting != "real"].sort_values("importance_mean", ascending=False)
V["IMPTOP"] = ", ".join(r"\texttt{" + tex_escape(x) + "}" for x in pci.feature.head(5))
rei = imp[imp.setting == "real"]
V["IMPREALMAX"] = f(rei.importance_mean.max(), 4)

# ---- CNN order
if len(cnn):
    for setting, key in [("real", "REAL"), (f"positive control s={C.PC_CNN_STRENGTH}", "PC")]:
        c = cnn[(cnn.setting == setting) & cnn.ordering.str.contains("canonical|random")]
        V[f"CNN{key}MEAN"], V[f"CNN{key}SD"] = f(c.test_auc.mean()), f(c.test_auc.std())
        V[f"CNN{key}MIN"], V[f"CNN{key}MAX"] = f(c.test_auc.min()), f(c.test_auc.max())
        V[f"CNN{key}CANON"] = f(c[c.ordering == "canonical"].test_auc.iloc[0])
        for ref in ["MLP", "LightGBM"]:
            V[f"{ref.upper()}{key}"] = f(cnn[(cnn.setting == setting) & cnn.ordering.str.startswith(ref)].test_auc.iloc[0])
    rows = []
    for _, r in cnn.iterrows():
        rows.append(f"{'Real' if r.setting == 'real' else 'Planted, $s$=' + str(C.PC_CNN_STRENGTH)} & "
                    f"{tex_escape(r.ordering)} & {r.test_auc:.3f} & {1000 * r.test_ap:.2f} \\\\")
    V["CNNROWS"] = "\n".join(rows)
    V["CNNINPUTS"] = str(man["cnn"]["n_inputs"])

# ---- latency
V["LATLGB"] = f(lat["LightGBM"]["median_ms"], 1)
V["LATLGB95"] = f(lat["LightGBM"]["p95_ms"], 1)
V["LATLR"] = f(lat["Logistic Regression"]["median_ms"], 1)
V["LATRF"] = f(lat["Random Forest"]["median_ms"], 1)

if __name__ == "__main__":
    tpl = (R / "template.tex").read_text()
    missing = sorted(set(re.findall(r"<<([A-Z0-9_]+)>>", tpl)) - set(V))
    if missing:
        raise SystemExit(f"Template tokens without values: {missing}")
    out = re.sub(r"<<([A-Z0-9_]+)>>", lambda mm: V[mm.group(1)], tpl)
    (R / "main.tex").write_text(out)
    (A / "report_values.json").write_text(json.dumps(V, indent=1))
    print(f"report/main.tex written ({len(V)} values)")
