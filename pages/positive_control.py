import plotly.graph_objects as go
import streamlit as st

from src.app_utils import PALETTE, load_csv, load_json, require_artifacts

st.title("🧪 Positive Control: can the pipeline find fraud when it exists?")
require_artifacts()
man = load_json("manifest.json")
st.warning("Labels on this page are **simulated**. Real features are kept, but the fraud label is "
           "redrawn from a documented risk score at the real 0.192% prevalence. These results validate "
           "the method. They are **not** evidence about real UPI fraud.")
st.latex(r"y^*_i \sim \mathrm{Bernoulli}\big(\sigma(\beta_0 + s\, z_i)\big),\quad "
         r"\beta_0 \text{ s.t. } \bar{\sigma} = 0.00192")
st.markdown("Risk score $z$ (standardised) is a weighted sum of:")
st.table({"Term": [t for t, _ in man["risk_score_terms"]], "Weight": [w for _, w in man["risk_score_terms"]]})

pc = load_csv("positive_control.csv")
agg = pc.groupby(["strength", "model"]).agg(auc=("test_auc", "mean"), sd=("test_auc", "std"),
                                            oracle=("oracle_auc", "mean"),
                                            nb=("net_benefit_inr", "mean")).reset_index()
fig = go.Figure()
orc = agg.groupby("strength")["oracle"].mean()
fig.add_scatter(x=orc.index, y=orc.values, name="Oracle (true probability)",
                line=dict(color="black", dash="dash"))
for mname, g in agg.groupby("model"):
    fig.add_scatter(x=g["strength"], y=g["auc"], error_y=dict(array=g["sd"].fillna(0)),
                    name=mname, mode="lines+markers", line_color=PALETTE.get(mname))
fig.update_layout(xaxis_title="Planted signal strength s", yaxis_title="Test ROC-AUC (mean ± sd, 3 draws)",
                  height=420)
st.plotly_chart(fig, width="stretch")
st.caption("At s = 0 (the real data's situation) every model sits at about 0.5. As soon as a signal "
           "exists, the pipeline recovers most of the oracle AUC.")

st.subheader("Net INR benefit on the test window (review cost ₹25)")
st.dataframe(agg.pivot(index="strength", columns="model", values="nb").style.format("₹{:,.0f}"),
             width="stretch")

c1, c2 = st.columns(2)
with c1:
    st.subheader("1D-CNN feature-order sensitivity")
    co = load_csv("cnn_order.csv")
    if len(co):
        st.dataframe(co.style.format({"test_auc": "{:.4f}", "test_ap": "{:.5f}"}), hide_index=True,
                     width="stretch")
        st.caption("A convolution over tabular columns assumes neighbouring columns are related. Randomly "
                   "permuting the column order measures how much the result depends on that arbitrary choice.")
    else:
        st.info("PyTorch was not available during the run, so the CNN experiment was skipped.")
with c2:
    st.subheader("Feature-group ablation (LightGBM)")
    ab = load_csv("ablation.csv")
    st.dataframe(ab.style.format({"test_auc": "{:.4f}"}), hide_index=True, width="stretch")

st.subheader("Permutation importance: does explanation recover the planted drivers?")
imp = load_csv("permutation_importance.csv")
setting = st.radio("Setting", imp["setting"].unique(), horizontal=True, index=1)
d = imp[imp["setting"] == setting].sort_values("importance_mean")
fig = go.Figure(go.Bar(x=d["importance_mean"], y=d["feature"], orientation="h",
                       error_x=dict(array=d["importance_sd"]), marker_color="#1f4e79"))
fig.update_layout(xaxis_title="Drop in test ROC-AUC when feature is shuffled", height=520, margin=dict(t=10))
st.plotly_chart(fig, width="stretch")
