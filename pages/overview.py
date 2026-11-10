import streamlit as st

from src.app_utils import artifacts_ready, load_csv, load_json

st.title("🛡️ UPI Guard: Pre-Debit Fraud Risk Scoring for UPI Payments")
st.caption("Goa Institute of Management - Big Data Analytics | Yogendra Pandey, Sahil Mangla, Kuldeep Khengar")

st.markdown("""
**Decision supported.** When a payer presses *Pay*, before the debit is authorised, the
bank or PSP chooses between **approve** and **step-up verification** (extra OTP, cooling period,
or manual review). UPI Guard estimates a calibrated fraud probability from information that
exists at that moment, and converts it into an INR-denominated decision.

**What this study does.**
1. **Provenance and learnability audit.** Before any model is trusted, we test whether the
   fraud label is statistically associated with *any* recorded attribute.
2. **Leakage-free, out-of-time benchmark.** Six model families, trained on Jan-Aug 2024,
   calibrated and thresholded on Sep-Oct, and evaluated once on Nov-Dec, with exact
   confusion counts and bootstrap confidence intervals.
3. **Positive control.** A known signal is planted in simulated labels to confirm the
   pipeline *can* detect fraud when it is present, and to test the 1D-CNN feature-order claim.
4. **INR business case.** The net benefit of each policy is computed from the real transaction
   amounts, with explicit and varied review-cost assumptions.
""")

if not artifacts_ready():
    st.warning("Artifacts not found. Run `python run_pipeline.py` first.")
    st.stop()

m = load_json("metrics_test.json")
man = load_json("manifest.json")
perm = load_json("audit_permutation.json")
power = load_json("power.json")
split = load_csv("split_table.csv")

st.subheader("Headline results (single identified run)")
best = max((k for k in m if k != "Prevalence baseline"), key=lambda k: m[k]["roc_auc"])
c1, c2, c3, c4 = st.columns(4)
c1.metric("Transactions / frauds", f"{man['checks']['rows']:,} / {man['checks']['fraud_count']}",
          f"{100 * man['checks']['fraud_rate']:.3f}% prevalence", delta_color="off")
c2.metric("Label-permutation test p", f"{perm['p_value']:.2f}",
          f"CV AUC {perm['observed_cv_auc']:.3f} vs null {perm['null_mean']:.3f}", delta_color="off")
lo, hi = m[best]["roc_auc_ci"]
c3.metric(f"Best test AUC ({best})", f"{m[best]['roc_auc']:.3f}", f"95% CI [{lo:.3f}, {hi:.3f}]",
          delta_color="off")
c4.metric("Min. detectable AUC (80% power)", f"{power['min_detectable_auc_80power']:.3f}",
          f"{power['test_frauds']} test frauds", delta_color="off")

st.info(f"""
**Main finding.** In this dataset the `fraud_flag` is statistically indistinguishable from a label
assigned independently of every recorded attribute. No model separates fraud from legitimate
traffic better than chance on the untouched Nov-Dec test window, and every intervention policy has
non-positive expected INR benefit. On semi-synthetic labels with a planted signal, the *same*
pipeline recovers near-oracle discrimination. This shows the null result is a property of the data,
not a pipeline failure. Earlier claims of AUC 0.999 are therefore **not reproducible** on this file.
""")

st.subheader("Pipeline")
st.graphviz_chart("""
digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef3f8", fontsize=11];
 A [label="Raw CSV\\n250,000 rows\\nSHA-256 pinned"]; B [label="Schema &\\nintegrity checks"];
 C [label="Drop post-auth\\ntransaction_status"]; D [label="Pre-debit features\\n(train-fitted stats,\\npast-only velocity)"];
 E [label="Out-of-time split\\nJan-Aug | Sep-Oct | Nov-Dec"]; F [label="Learnability audit\\nchi-square + BH,\\npermutation, power"];
 G [label="5-fold CV model\\nselection (train)"]; H [label="Platt calibration +\\nthreshold (validation)"];
 I [label="Single test\\nevaluation"]; J [label="INR cost model\\n& sensitivity"]; K [label="Positive control\\n(planted signal)"];
 A->B->C->D->E; E->F; E->G->H->I->J; D->K; }
""")

st.subheader("Data split (by timestamp)")
st.dataframe(split, hide_index=True, width="stretch")
