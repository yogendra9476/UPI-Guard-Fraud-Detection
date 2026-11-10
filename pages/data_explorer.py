import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.app_utils import load_dataset, wilson

st.title("📊 Data Explorer")
df = load_dataset()

with st.sidebar:
    st.markdown("**Filters**")
    months = st.slider("Month range (2024)", 1, 12, (1, 12))
    types = st.multiselect("Transaction type", sorted(df["transaction_type"].unique()),
                           default=sorted(df["transaction_type"].unique()))
d = df[(df["timestamp"].dt.month.between(*months)) & (df["transaction_type"].isin(types))]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Transactions", f"{len(d):,}")
c2.metric("Frauds", f"{int(d['fraud_flag'].sum()):,}")
c3.metric("Fraud rate", f"{100 * d['fraud_flag'].mean():.3f}%")
c4.metric("Median amount", f"₹{d['amount_inr'].median():,.0f}")

st.subheader("Fraud rate by attribute, with 95% Wilson intervals")
st.caption("The dashed line is the overall rate. Overlapping intervals mean the attribute does "
           "not discriminate. Compare with the formal tests on the *Learnability Audit* page.")
attr = st.selectbox("Attribute", ["merchant_category", "transaction_type", "sender_age_group",
                                  "receiver_age_group", "sender_state", "sender_bank", "receiver_bank",
                                  "device_type", "network_type", "day_of_week", "hour_of_day",
                                  "transaction_status"])
g = d.groupby(attr)["fraud_flag"].agg(["sum", "count"]).reset_index()
g["rate"] = 100 * g["sum"] / g["count"]
lo, hi = wilson(g["sum"], g["count"])
g["lo"], g["hi"] = 100 * lo, 100 * hi
fig = go.Figure(go.Bar(x=g[attr].astype(str), y=g["rate"], marker_color="#1f4e79",
                       error_y=dict(type="data", symmetric=False, array=g["hi"] - g["rate"],
                                    arrayminus=g["rate"] - g["lo"]),
                       hovertemplate="%{x}: %{y:.3f}%<extra></extra>"))
fig.add_hline(y=100 * d["fraud_flag"].mean(), line_dash="dash", line_color="gray")
fig.update_layout(yaxis_title="Fraud rate (%)", height=420, margin=dict(t=20))
st.plotly_chart(fig, width="stretch")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Amount distribution by class")
    fig = px.histogram(d.assign(cls=d["fraud_flag"].map({0: "Legitimate", 1: "Fraud"})),
                       x="amount_inr", color="cls", log_x=False, histnorm="probability density",
                       nbins=120, barmode="overlay", opacity=0.6, range_x=[0, 8000],
                       color_discrete_map={"Legitimate": "#1f77b4", "Fraud": "#d62728"})
    fig.update_layout(height=380, xaxis_title="Amount (INR)", legend_title=None, margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")
with col2:
    st.subheader("Monthly volume and fraud count")
    mth = d.groupby(d["timestamp"].dt.month)["fraud_flag"].agg(["count", "sum"]).reset_index()
    fig = go.Figure()
    fig.add_bar(x=mth["timestamp"], y=mth["count"], name="Transactions", marker_color="#9ecae1")
    fig.add_scatter(x=mth["timestamp"], y=mth["sum"], name="Frauds", yaxis="y2",
                    mode="lines+markers", line_color="#d62728")
    fig.update_layout(height=380, yaxis2=dict(overlaying="y", side="right", title="Frauds"),
                      xaxis_title="Month", yaxis_title="Transactions", margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")

st.subheader("Sample rows")
st.dataframe(d.head(200), hide_index=True, width="stretch")
