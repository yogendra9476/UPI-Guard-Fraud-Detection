# Model Card: UPI Guard pre-debit risk scorer

**Task.** Estimate P(fraud | information available before the debit) for a UPI payment, and
recommend *approve* or *step-up verification*.

**Models.** Logistic regression, random forest, LightGBM, MLP and 1D-CNN, plus a prevalence
baseline. Hyper-parameters are chosen by 5-fold stratified CV (PR-AUC) on Jan to Aug 2024.
Platt calibration and threshold selection use Sep to Oct 2024. A single evaluation is run on
Nov to Dec 2024. Numbers: `artifacts/metrics_test.json`.

**Inputs.** 24 pre-debit features (`artifacts/manifest.json` → `features`).
`transaction_status` is excluded because it is an authorisation outcome.

**Performance (this dataset).** No model's test ROC-AUC is distinguishable from 0.5 after
multiplicity correction. PR-AUC is at the prevalence level. The INR-cost-optimal policy is to flag
nothing. **The scorer must not be deployed.** It demonstrates a decision service and an
evaluation protocol.

**Positive control.** On simulated labels with a planted signal, the same pipeline reaches
test AUC close to the oracle (`artifacts/positive_control.csv`), so the pipeline is able to
detect signal when signal exists.

**Ethical considerations.** Age-group features can produce disparate step-up rates across
demographics. Any real deployment needs group-wise false-positive monitoring and a review of the
legal basis for using age.

**Monitoring if trained on real data.** PSI on inputs, monthly out-of-time AUC/PR-AUC with
confidence intervals, alert-rate and precision by segment, and label-delay-aware back-testing.
