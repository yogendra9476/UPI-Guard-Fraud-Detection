import json

import pandas as pd
import streamlit as st

from src.app_utils import load_csv, load_json, require_artifacts

st.title("📋 Reproducibility Record")
require_artifacts()
man = load_json("manifest.json")

c1, c2, c3 = st.columns(3)
c1.metric("Run (UTC)", man["run_utc"].replace("T", " ")[:16])
c2.metric("Random seed", man["seed"])
c3.metric("Runtime", f"{man['runtime_seconds'] / 60:.1f} min")
st.markdown(f"**Dataset:** `{man['dataset_file']}`  \n**SHA-256:** `{man['dataset_sha256']}`")

st.subheader("Integrity checks")
st.dataframe(pd.DataFrame(list(man["checks"].items()), columns=["check", "value"]).astype(str),
             hide_index=True, width="stretch")
st.subheader("Split")
st.dataframe(load_csv("split_table.csv"), hide_index=True, width="stretch")
st.subheader("Excluded columns")
st.table(pd.DataFrame(list(man["excluded_columns"].items()), columns=["column", "reason"]))
st.subheader("Model inputs (all available before the debit)")
st.write(", ".join(f"`{f}`" for f in man["features"]))
st.subheader("Selected hyper-parameters (5-fold CV on train, by PR-AUC)")
cv = load_json("cv_results.json")
st.dataframe(pd.DataFrame([{"model": k, "params": json.dumps(v["params"]),
                            "cv_auc": f"{v['cv_auc_mean']:.4f} ± {v['cv_auc_sd']:.4f}",
                            "cv_pr_auc": f"{v['cv_ap_mean']:.5f} ± {v['cv_ap_sd']:.5f}"}
                           for k, v in cv["selected"].items()]), hide_index=True, width="stretch")
st.subheader("Software versions")
st.json(man["versions"])
st.subheader("Reproduce")
st.code("pip install -r requirements.txt\npython run_pipeline.py      # writes ./artifacts\n"
        "python make_figures.py      # writes ./report/figures\npytest -q\nstreamlit run app.py",
        language="bash")
