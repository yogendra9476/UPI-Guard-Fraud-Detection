import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.app_utils import inr, load_csv, load_json, model_names, require_artifacts

st.title("💰 Business Impact in INR")
require_artifacts()
st.markdown("""
**Policy.** Flag a transaction for step-up verification when its calibrated fraud probability is at
or above a threshold. **Benefit** = (effectiveness × amount of each flagged fraud) - (review cost ×
number of flags), measured against *flag nothing*. Fraud amounts are the real transaction values in
the test window. Review cost and effectiveness are **assumptions**, which you can vary below.
""")
pred = load_csv("predictions_test.csv")
m = load_json("metrics_test.json")
man = load_json("manifest.json")

c1, c2, c3 = st.columns(3)
cost = c1.slider("Review / step-up cost per flag (₹)", 1, 200, 25)
eff = c2.slider("Share of flagged fraud prevented", 0.1, 1.0, 1.0, 0.05)
model = c3.selectbox("Model", model_names(m, include_baseline=False))

y = pred["y"].to_numpy()
amt = pred["amount_inr"].to_numpy(float)
p = pred[model].to_numpy()
order = np.argsort(-p, kind="mergesort")
rates = np.unique(np.concatenate([np.linspace(0.0005, 0.02, 40), np.linspace(0.02, 0.3, 40)]))
rows = []
for r in rates:
    k = int(round(r * len(p)))
    sel = order[:k]
    saved = eff * amt[sel][y[sel] == 1].sum()
    rows.append({"alert_rate": r, "flags": k, "frauds_caught": int(y[sel].sum()),
                 "loss_prevented": saved, "review_cost": cost * k, "net": saved - cost * k})
curve = pd.DataFrame(rows)
# random-ranking expectation for reference
curve["net_random"] = eff * curve["flags"] * (amt[y == 1].sum() / len(y)) - cost * curve["flags"]

fig = go.Figure()
fig.add_scatter(x=100 * curve["alert_rate"], y=curve["net"], name=model, line_color="#1f4e79")
fig.add_scatter(x=100 * curve["alert_rate"], y=curve["net_random"], name="Random flagging (expected)",
                line=dict(dash="dash", color="gray"))
fig.add_hline(y=0, line_color="black")
fig.update_layout(xaxis_title="Share of transactions flagged (%)", yaxis_title="Net benefit vs. no model (₹)",
                  height=420)
st.plotly_chart(fig, width="stretch")

total_loss = amt[y == 1].sum()
mean_f = man["mean_fraud_amount_train_inr"]
be = cost / (eff * mean_f)
prev = y.mean()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Fraud loss in test window", inr(total_loss), f"{int(y.sum())} frauds", delta_color="off")
k2.metric("Best net benefit on this curve", inr(curve["net"].max()))
k3.metric("Break-even precision", f"{100 * be:.2f}%", f"needs lift ≥ {be / prev:.1f}×", delta_color="off")
k4.metric("Achieved lift @ top 10%", f"{m[model]['lift_top10']:.2f}×")

st.markdown(f"""
**Interpretation.** A flag pays for itself only if precision × effectiveness × mean fraud amount
(₹{mean_f:,.0f}, training window) exceeds the review cost, i.e. precision ≥ **{100 * be:.2f}%**. That is
**{be / prev:.1f}×** the base rate of {100 * prev:.3f}%. A model whose lift is near 1 cannot clear
this bar at any threshold, so the cost-optimal policy is not to intervene. The table below gives the
policy chosen on the validation window for each review-cost assumption.
""")
sens = load_csv("cost_sensitivity.csv")
st.dataframe(sens[["cost_review", "model", "threshold", "flagged", "frauds_caught", "loss_prevented",
                   "review_cost", "net_benefit", "break_even_precision"]]
             .style.format({"threshold": "{:.5f}", "loss_prevented": "₹{:,.0f}", "review_cost": "₹{:,.0f}",
                            "net_benefit": "₹{:,.0f}", "break_even_precision": "{:.2%}"}),
             hide_index=True, width="stretch")
