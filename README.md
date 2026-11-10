# 🛡️ UPI Guard: Audited Pre-Debit Fraud-Risk Scoring for UPI Payments

Goa Institute of Management, Big Data Analytics.
Yogendra Pandey (B2026109), Sahil Mangla (B2026097), Kuldeep Khengar (B2026082)

A complete, single-run, reproducible study and Streamlit application:

* **Learnability audit**: chi-square/Mann-Whitney tests with Benjamini-Hochberg correction, a
  label-permutation test of CV AUC, and a power analysis.
* **Leakage-free, out-of-time benchmark**: train Jan to Aug, validate (calibrate and choose
  thresholds) Sep to Oct, test once on Nov to Dec 2024. Models: prevalence baseline, logistic
  regression, random forest, LightGBM, MLP and 1D-CNN.
* **INR cost model** using real transaction amounts, with break-even precision and sensitivity analysis.
* **Positive control**: simulated labels with a planted signal show that the pipeline can detect
  fraud when it exists. A **1D-CNN feature-order test** checks whether column order matters.

## Quick start
```bash
pip install -r requirements.txt          # add `pip install torch` to rerun the CNN parts
python run_pipeline.py                   # ~45 min on 4 CPU cores (CNN parts ~20 min); writes ./artifacts
python make_figures.py                   # PNG figures -> ./report/figures
python build_report.py                   # fills report/template.tex -> report/main.tex
pytest -q                                # 12 correctness/leakage tests
streamlit run app.py                     # open http://localhost:8501
```
Pre-computed artifacts from the reported run are included, so `streamlit run app.py` works
immediately.

## Layout
```
app.py                  Streamlit entry point (8 pages in ./pages)
run_pipeline.py         the one experiment run behind every number in the report
make_figures.py         matplotlib figures
build_report.py         writes the numbers from ./artifacts into the LaTeX report
src/config.py           seed, split dates, cost assumptions (single source of truth)
src/data.py             loading, schema checks, SHA-256, time split
src/features.py         pre-debit features (train-fitted stats, past-only velocity)
src/audit.py            learnability audit
src/models.py           model zoo, CV grids, Platt calibration
src/cnn.py              1D-CNN (PyTorch, optional)
src/evaluate.py         metrics, bootstrap CIs, ECE, confusion, INR cost model
src/positive_control.py simulated-label generator
tests/                  pytest suite
artifacts/              outputs of the reported run (manifest.json pins the data hash)
report/                 LaTeX source and figures
DATA_CARD.md, MODEL_CARD.md
```

## Key result
`fraud_flag` in the supplied file is statistically independent of every recorded attribute.
All models are at chance on the untouched test window, and the INR-optimal policy is to flag
nothing. The same code recovers near-oracle AUC when a signal is planted. See the report and
the app for the full numbers.
