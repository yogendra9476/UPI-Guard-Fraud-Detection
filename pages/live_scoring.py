from datetime import datetime

import pandas as pd
import streamlit as st

from src.app_utils import inr, load_dataset, load_json, require_artifacts, score_frame
from src.data import EXPECTED_COLUMNS
from src import config as C

st.title("⚡ Live Pre-Debit Scoring")
require_artifacts()
man = load_json("manifest.json")
metrics = load_json("metrics_test.json")
best = max((k for k in metrics if k != "Prevalence baseline"), key=lambda k: metrics[k]["roc_auc"])
st.warning(f"These models were trained on the supplied dataset, where the label carries no detectable "
           f"signal (best test AUC {metrics[best]['roc_auc']:.3f}). The scores below show the **mechanics** "
           f"of a pre-debit decision service. Scores near the base rate are the expected outcome.")

ref = load_dataset()
opts = {c: sorted(ref[c].unique()) for c in C.PRE_DEBIT_CATEGORICAL if c != "day_of_week"}
fb_med = None

tab1, tab2 = st.tabs(["Single transaction", "Batch CSV"])
with tab1:
    with st.form("txn"):
        c1, c2, c3 = st.columns(3)
        amount = c1.number_input("Amount (₹)", 1, 500000, 2500)
        ttype = c1.selectbox("Transaction type", opts["transaction_type"])
        cat = c1.selectbox("Merchant category", opts["merchant_category"])
        sage = c2.selectbox("Sender age group", opts["sender_age_group"])
        rage = c2.selectbox("Receiver age group", opts["receiver_age_group"])
        state = c2.selectbox("Sender state", opts["sender_state"])
        sbank = c3.selectbox("Sender bank", opts["sender_bank"])
        rbank = c3.selectbox("Receiver bank", opts["receiver_bank"])
        dev = c3.selectbox("Device", opts["device_type"])
        net = c3.selectbox("Network", opts["network_type"])
        d1, d2, d3, d4 = st.columns(4)
        date = d1.date_input("Date", datetime(2024, 11, 15))
        hour = d2.slider("Hour of day", 0, 23, 14)
        seg1 = d3.number_input("Segment txns, past 1 h", 0, 1000, 0,
                               help="Earlier payments from the same bank/state/age/device segment in the last hour")
        seg24 = d4.number_input("Segment txns, past 24 h", 0, 5000, 7)
        glob10 = st.number_input("All txns in the previous 10 min", 0, 1000, 5)
        go_ = st.form_submit_button("Score transaction", type="primary")
    if go_:
        ts = pd.Timestamp(datetime(date.year, date.month, date.day, hour))
        row = pd.DataFrame([{
            "transaction_id": "LIVE", "timestamp": ts, "transaction_type": ttype, "merchant_category": cat,
            "amount_inr": amount, "sender_age_group": sage, "receiver_age_group": rage,
            "sender_state": state, "sender_bank": sbank, "receiver_bank": rbank, "device_type": dev,
            "network_type": net, "hour_of_day": hour, "day_of_week": ts.day_name(),
            "is_weekend": int(ts.dayofweek >= 5), "seg_count_1h": seg1, "seg_count_24h": seg24,
            "global_count_10m": glob10}])
        s = score_frame(row, compute_velocity=False).iloc[0]
        base = man["checks"]["fraud_rate"]
        be = man["break_even_precision"]
        cols = st.columns(len(s))
        for col, (name, p) in zip(cols, s.items()):
            col.metric(name, f"{100 * p:.3f}%", f"{p / base:.2f}× base rate", delta_color="off")
        p_best = s[best]
        decision = "STEP-UP VERIFICATION" if p_best >= be else "APPROVE"
        exp_loss = p_best * amount
        st.markdown(f"### Decision ({best}): **{decision}**")
        st.write(f"Expected fraud loss if approved: p × amount = {inr(exp_loss)}; review cost "
                 f"₹{C.COST_REVIEW_INR:.0f}. Step-up is worthwhile only if p × amount > review cost, "
                 f"i.e. p ≥ {C.COST_REVIEW_INR / amount:.4%} for this amount.")
        st.caption("transaction_status is deliberately not an input: it is only known after authorisation.")

with tab2:
    st.markdown("Upload a CSV with the original dataset schema (the `fraud_flag` and "
                "`transaction_status` columns are optional and ignored for scoring). Velocity "
                "features are computed backwards in time *within the uploaded batch*.")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        raw = pd.read_csv(up)
        needed = [c for c in EXPECTED_COLUMNS if c not in ("fraud_flag", "transaction_status")]
        miss = [c for c in needed if c not in raw.columns]
        if miss:
            st.error(f"Missing columns: {miss}")
        else:
            raw = raw.rename(columns=C.RAW_RENAME)
            raw["timestamp"] = pd.to_datetime(raw["timestamp"])
            scores = score_frame(raw, compute_velocity=True)
            out = pd.concat([raw[["transaction_id", "timestamp", "amount_inr"]], scores.add_prefix("p_")], axis=1)
            out["decision"] = (out[f"p_{best}"] * out["amount_inr"] > C.COST_REVIEW_INR).map(
                {True: "STEP-UP", False: "APPROVE"})
            st.dataframe(out.head(1000), hide_index=True, width="stretch")
            st.download_button("Download scores", out.to_csv(index=False), "upi_guard_scores.csv", "text/csv")
