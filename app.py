"""UPI Guard - Streamlit application.

    streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="UPI Guard", page_icon="🛡️", layout="wide")

pages = [
    st.Page("pages/overview.py", title="Overview", icon="🏠", default=True),
    st.Page("pages/data_explorer.py", title="Data Explorer", icon="📊"),
    st.Page("pages/signal_audit.py", title="Learnability Audit", icon="🔬"),
    st.Page("pages/model_results.py", title="Model Results", icon="📈"),
    st.Page("pages/positive_control.py", title="Positive Control", icon="🧪"),
    st.Page("pages/business_impact.py", title="Business Impact (INR)", icon="💰"),
    st.Page("pages/live_scoring.py", title="Live Scoring", icon="⚡"),
    st.Page("pages/reproducibility.py", title="Reproducibility", icon="📋"),
]
nav = st.navigation(pages)
with st.sidebar:
    st.markdown("### 🛡️ UPI Guard")
    st.caption("Pre-debit fraud risk scoring for UPI payments - an audited, single-run study.")
nav.run()
