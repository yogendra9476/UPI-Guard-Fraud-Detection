import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.app_utils import load_csv, load_json, require_artifacts

st.title("🔬 Learnability Audit")
require_artifacts()
st.markdown("""
Before any classifier is reported, we ask a simpler question: **is the fraud label associated with
any recorded attribute at all?** Three complementary checks are run on all 250,000 rows:
1. Univariate tests for each attribute (chi-square for categorical, Mann-Whitney for numeric), with
   Benjamini-Hochberg correction for testing 18 hypotheses.
2. A label-permutation test of cross-validated AUC (Ojala & Garriga, 2010): the model is refit on
   randomly shuffled labels to obtain the AUC that chance alone produces.
3. A power analysis: the smallest test AUC that the 77 test frauds can distinguish from 0.5.
""")

uni = load_csv("audit_univariate.csv")
show = uni[["feature", "test", "levels", "statistic", "dof", "p_value", "p_bh", "effect", "effect_name",
            "min_rate_pct", "max_rate_pct"]].copy()
st.subheader("1. Univariate association tests")
st.dataframe(show.style.format({"statistic": "{:.2f}", "p_value": "{:.4f}", "p_bh": "{:.4f}",
                                "effect": "{:.4f}", "min_rate_pct": "{:.3f}", "max_rate_pct": "{:.3f}",
                                "levels": "{:.0f}", "dof": "{:.0f}"}, na_rep="-")
             .background_gradient(subset=["p_bh"], cmap="RdYlGn"),
             hide_index=True, width="stretch")
n_sig = int((uni["p_bh"] < 0.05).sum())
n_raw = int((uni["p_value"] < 0.05).sum())
st.write(f"**{n_raw}** of {len(uni)} tests have unadjusted p < 0.05; **{n_sig}** survive "
         f"Benjamini-Hochberg correction. With 18 tests, about 0.9 false positives are expected by chance.")

st.subheader("2. Label-permutation test")
perm = load_json("audit_permutation.json")
fig = go.Figure(go.Histogram(x=perm["null"], nbinsx=30, marker_color="#9ecae1", name="Permuted labels"))
fig.add_vline(x=perm["observed_cv_auc"], line_color="#d62728", line_width=3,
              annotation_text=f"True labels: {perm['observed_cv_auc']:.3f}")
fig.update_layout(xaxis_title="3-fold CV ROC-AUC (logistic regression)", yaxis_title="Count",
                  height=360, margin=dict(t=30))
st.plotly_chart(fig, width="stretch")
st.write(f"Null mean {perm['null_mean']:.4f} (sd {perm['null_sd']:.4f}); permutation p-value "
         f"**{perm['p_value']:.3f}** from {perm['n_perm']} permutations.")

st.subheader("3. Statistical power of the test window")
pw = load_json("power.json")
c1, c2, c3 = st.columns(3)
c1.metric("Test frauds / legitimate", f"{pw['test_frauds']} / {pw['test_legit']:,}")
c2.metric("SE of AUC under H0", f"{pw['auc_null_se']:.4f}")
c3.metric("Min. detectable AUC (α=0.05, 80% power)", f"{pw['min_detectable_auc_80power']:.3f}")
st.caption("A model with true AUC below this value would usually be indistinguishable from chance on "
           "this test window. Larger evaluation windows or pooled multi-period testing would be needed.")
