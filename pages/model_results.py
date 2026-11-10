import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.app_utils import PALETTE, load_csv, load_json, model_names, require_artifacts

st.title("📈 Model Results: untouched test window (Nov-Dec 2024)")
require_artifacts()
m = load_json("metrics_test.json")
names = model_names(m)

rows = []
for n in names:
    r = m[n]
    rows.append({"Model": n,
                 "CV AUC (train)": f"{r['cv_auc_mean']:.3f} ± {r['cv_auc_sd']:.3f}" if "cv_auc_mean" in r else "-",
                 "Test ROC-AUC": r["roc_auc"], "AUC 95% CI": f"[{r['roc_auc_ci'][0]:.3f}, {r['roc_auc_ci'][1]:.3f}]",
                 "p (AUC>0.5)": r["auc_p_value"], "PR-AUC": r["pr_auc"],
                 "Brier skill": r["brier_skill"], "ECE": r["ece"], "Lift@top10%": r["lift_top10"]})
tab = pd.DataFrame(rows)
st.dataframe(tab.style.format({"Test ROC-AUC": "{:.4f}", "p (AUC>0.5)": "{:.3f}", "PR-AUC": "{:.5f}",
                               "Brier skill": "{:.5f}", "ECE": "{:.5f}", "Lift@top10%": "{:.2f}"}),
             hide_index=True, width="stretch")
st.caption(f"PR-AUC of a random ranking equals the test prevalence ({m[names[0]]['pr_auc_baseline']:.5f}). "
           "Probabilities are Platt-calibrated on the validation window. AUC p-values are one-sided "
           "Mann-Whitney tests; with five models compared, apply a multiplicity correction (Holm).")

curves = load_json("curves_test.json")
c1, c2 = st.columns(2)
with c1:
    fig = go.Figure()
    for n in names:
        fpr, tpr = curves[n]["roc"]
        fig.add_scatter(x=fpr, y=tpr, name=f"{n} ({m[n]['roc_auc']:.3f})", line_color=PALETTE.get(n))
    fig.add_scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="black"), name="Chance")
    fig.update_layout(title="ROC curves", xaxis_title="False positive rate", yaxis_title="True positive rate",
                      height=420, legend=dict(font_size=10))
    st.plotly_chart(fig, width="stretch")
with c2:
    fig = go.Figure()
    for n in names:
        rec, prec = curves[n]["pr"]
        fig.add_scatter(x=rec, y=prec, name=n, line_color=PALETTE.get(n))
    fig.add_hline(y=m[names[0]]["pr_auc_baseline"], line_dash="dash", annotation_text="prevalence")
    fig.update_layout(title="Precision-recall curves", xaxis_title="Recall", yaxis_title="Precision",
                      height=420, yaxis_type="log", legend=dict(font_size=10))
    st.plotly_chart(fig, width="stretch")

st.subheader("Calibration (reliability, 10 equal-mass bins)")
rel = load_json("reliability_test.json")
sel = st.selectbox("Model", [n for n in names if n != "Prevalence baseline"])
r = rel[sel]
fig = go.Figure()
fig.add_scatter(x=r["mean_predicted"], y=r["observed_rate"], mode="lines+markers", name=sel)
mx = max(max(r["mean_predicted"]), max(r["observed_rate"])) * 1.1
fig.add_scatter(x=[0, mx], y=[0, mx], line=dict(dash="dash", color="black"), name="Perfect")
fig.update_layout(xaxis_title="Mean predicted probability", yaxis_title="Observed fraud rate", height=380)
st.plotly_chart(fig, width="stretch")
st.write(f"ECE before calibration {m[sel]['ece_raw']:.5f} -> after Platt scaling {m[sel]['ece']:.5f}.")

st.subheader("Confusion matrices at validation-chosen operating points")
cm = load_csv("confusion_test.csv")
op = st.radio("Operating point", cm["operating_point"].unique(), horizontal=True)
view = cm[cm["operating_point"] == op][["model", "threshold", "TP", "FP", "FN", "TN", "precision",
                                        "precision_identity", "recall", "specificity", "f1", "alert_rate", "lift"]]
st.dataframe(view.style.format({"threshold": "{:.5f}", "precision": "{:.4f}", "precision_identity": "{:.4f}",
                                "recall": "{:.4f}", "specificity": "{:.4f}", "f1": "{:.4f}",
                                "alert_rate": "{:.4f}", "lift": "{:.2f}"}),
             hide_index=True, width="stretch")
st.caption("`precision_identity` recomputes precision from recall, specificity and prevalence "
           "(Bayes' rule). It matches the count-based precision, which is the internal-consistency "
           "check the first review asked for. A threshold of `inf` means the cost-optimal policy "
           "on validation was to flag nothing.")
