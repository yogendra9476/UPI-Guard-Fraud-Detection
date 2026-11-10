"""End-to-end, single-run experiment for UPI Guard.

    python run_pipeline.py            # full run (~15-25 min on 4 CPU cores)
    python run_pipeline.py --fast     # smaller grids / fewer replicates for a smoke test

Writes everything the report and the Streamlit app need into ./artifacts.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src import config as C
from src.audit import label_permutation_test, univariate_tests
from src.cnn import TORCH_OK, TabularCNN
from src.data import load_raw, sha256_of_file, split_table, time_split, validate
from src.evaluate import (auc_pvalue, best_cost_threshold, break_even_precision, confusion, curve_points,
                          ece_quantile, minimum_detectable_auc, auc_null_se, net_benefit,
                          reliability_table, summarize, threshold_for_alert_rate)
from src.features import (ALL_FEATURES, FEATURE_GROUPS, FeatureBuilder, add_velocity_features)
from src.models import GRIDS, PlattCalibrator, make_model, onehot_preprocessor
from src.positive_control import RISK_SCORE_TERMS, risk_score, simulate_labels

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def jdump(obj, path):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    path.write_text(json.dumps(obj, indent=2, default=conv))


def cv_select(name, X, y, grid, folds=5):
    skf = StratifiedKFold(folds, shuffle=True, random_state=C.SEED)
    results = []
    for params in grid:
        aucs, aps = [], []
        for tr, te in skf.split(X, y):
            m = make_model(name, **params).fit(X.iloc[tr], y[tr])
            s = m.predict_proba(X.iloc[te])[:, 1]
            aucs.append(roc_auc_score(y[te], s))
            aps.append(average_precision_score(y[te], s))
        results.append({"params": params, "cv_auc_mean": float(np.mean(aucs)),
                        "cv_auc_sd": float(np.std(aucs)), "cv_ap_mean": float(np.mean(aps)),
                        "cv_ap_sd": float(np.std(aps)), "fold_aucs": aucs})
        log(f"   CV {name} {params}: AUC {np.mean(aucs):.4f}+-{np.std(aucs):.4f} AP {np.mean(aps):.5f}")
    best = max(results, key=lambda r: r["cv_ap_mean"])
    return best, results


def main(fast=False):
    A = C.ARTIFACTS
    (A / "models").mkdir(parents=True, exist_ok=True)
    grids = {k: v[:1] for k, v in GRIDS.items()} if fast else GRIDS
    n_perm = 20 if fast else C.N_PERMUTATIONS
    reps = 1 if fast else C.PC_REPLICATES
    n_boot = 200 if fast else C.N_BOOTSTRAP

    # ------------------------------------------------------------------ 1. data
    log("Loading data")
    df = load_raw()
    checks = validate(df)
    split = time_split(df)
    st = split_table(df, split)
    st.to_csv(A / "split_table.csv", index=False)
    log(f"   {checks['rows']} rows, {checks['fraud_count']} frauds; split:\n{st}")

    # ------------------------------------------------------------------ 2. features
    log("Building pre-debit features")
    df = add_velocity_features(df)
    fb = FeatureBuilder().fit(df.iloc[split.train])
    X = fb.transform(df)
    y = df[C.TARGET].to_numpy()
    amount = df["amount_inr"].to_numpy(dtype=float)
    joblib.dump(fb, A / "models" / "feature_builder.joblib")
    Xtr, Xva, Xte = X.iloc[split.train], X.iloc[split.valid], X.iloc[split.test]
    ytr, yva, yte = y[split.train], y[split.valid], y[split.test]

    # ------------------------------------------------------------------ 3. audit
    log("Learnability audit")
    uni = univariate_tests(df, X)
    uni.to_csv(A / "audit_univariate.csv", index=False)
    perm = label_permutation_test(df, n_perm=n_perm)
    jdump(perm, A / "audit_permutation.json")
    n_pos_te, n_neg_te = int(yte.sum()), int(len(yte) - yte.sum())
    power = {"test_frauds": n_pos_te, "test_legit": n_neg_te,
             "auc_null_se": auc_null_se(n_pos_te, n_neg_te),
             "auc_critical_5pct": 0.5 + 1.6449 * auc_null_se(n_pos_te, n_neg_te),
             "min_detectable_auc_80power": minimum_detectable_auc(n_pos_te, n_neg_te)}
    jdump(power, A / "power.json")
    log(f"   permutation: observed {perm['observed_cv_auc']:.4f}, null mean {perm['null_mean']:.4f}, p={perm['p_value']:.3f}")

    # ------------------------------------------------------------------ 4. models
    log("Model selection on the training window (5-fold stratified CV)")
    cv_records, selected = {}, {}
    for name in ["Logistic Regression", "Random Forest", "LightGBM", "MLP"]:
        best, allres = cv_select(name, Xtr, ytr, grids[name])
        cv_records[name] = allres
        selected[name] = best
    jdump({"selected": selected, "all": cv_records}, A / "cv_results.json")

    log("Fitting final models on train, calibrating on validation, scoring test")
    preds_va = pd.DataFrame({"transaction_id": df["transaction_id"].iloc[split.valid].values,
                             "y": yva, "amount_inr": amount[split.valid]})
    preds_te = pd.DataFrame({"transaction_id": df["transaction_id"].iloc[split.test].values,
                             "timestamp": df["timestamp"].iloc[split.test].astype(str).values,
                             "y": yte, "amount_inr": amount[split.test]})
    raw_scores = {}
    for name in ["Prevalence baseline", "Logistic Regression", "Random Forest", "LightGBM", "MLP"]:
        params = selected[name]["params"] if name in selected else {}
        m = make_model(name, **params).fit(Xtr, ytr)
        joblib.dump(m, A / "models" / f"{name.replace(' ', '_').lower()}.joblib")
        raw_scores[name] = (m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1])
        log(f"   {name} fitted")

    cnn_info = {"available": TORCH_OK}
    if TORCH_OK:
        pre = onehot_preprocessor().fit(Xtr)
        Ztr, Zva, Zte = pre.transform(Xtr), pre.transform(Xva), pre.transform(Xte)
        cnn = TabularCNN(seed=C.SEED).fit(Ztr, ytr, Zva, yva)
        raw_scores["1D-CNN"] = (cnn.predict_proba(Zva)[:, 1], cnn.predict_proba(Zte)[:, 1])
        cnn_info.update({"n_inputs": int(Ztr.shape[1]), "epochs_run": len(cnn.history_),
                         "val_ap_history": cnn.history_})
        import torch
        torch.save(cnn.net_.state_dict(), A / "models" / "cnn_state.pt")
        joblib.dump(pre, A / "models" / "cnn_preprocessor.joblib")
        log("   1D-CNN fitted")

    calibrators, metrics, conf_rows, cost_rows, curves, rel = {}, {}, [], [], {}, {}
    mean_fraud_amt_train = float(amount[split.train][ytr == 1].mean())
    for name, (sva, ste) in raw_scores.items():
        if name == "Prevalence baseline":
            pva, pte = sva, ste
        else:
            cal = PlattCalibrator().fit(sva, yva)
            calibrators[name] = cal
            pva, pte = cal.transform(sva), cal.transform(ste)
        preds_va[name] = pva
        preds_te[name] = pte
        met = summarize(yte, pte, n_boot=n_boot)
        met["ece_raw"] = ece_quantile(yte, ste) if name != "Prevalence baseline" else met["ece"]
        met["valid_auc"] = roc_auc_score(yva, pva)
        if name in selected:
            met["cv_auc_mean"] = selected[name]["cv_auc_mean"]
            met["cv_auc_sd"] = selected[name]["cv_auc_sd"]
        metrics[name] = met
        # operating points chosen on validation only
        for label, thr in [("Alert budget 1%", threshold_for_alert_rate(pva, 0.01)),
                           ("Cost-optimal (INR 25)", best_cost_threshold(yva, pva, amount[split.valid]))]:
            cm = confusion(yte, pte, thr)
            cm.update({"model": name, "operating_point": label})
            conf_rows.append(cm)
            nb = net_benefit(yte, pte, amount[split.test], thr)
            nb.update({"model": name, "operating_point": label})
            cost_rows.append(nb)
        curves[name] = {"roc": [a.tolist() for a in curve_points(yte, pte, "roc")],
                        "pr": [a.tolist() for a in curve_points(yte, pte, "pr")]}
        rel[name] = reliability_table(yte, pte).to_dict(orient="list")
        log(f"   {name}: test AUC {met['roc_auc']:.4f} {met['roc_auc_ci']} p={met['auc_p_value']:.3f} AP {met['pr_auc']:.5f}")
    joblib.dump(calibrators, A / "models" / "calibrators.joblib")

    # Inference latency: feature transform + model + calibration for one transaction
    log("Measuring single-transaction scoring latency")
    latency = {}
    raw_rows = df.iloc[split.test[:200]]
    for name in ["Logistic Regression", "Random Forest", "LightGBM", "MLP"]:
        mdl = joblib.load(A / "models" / f"{name.replace(' ', '_').lower()}.joblib")
        times = []
        for i in range(200):
            t1 = time.perf_counter()
            xi = fb.transform(raw_rows.iloc[[i]])
            calibrators[name].transform(mdl.predict_proba(xi)[:, 1])
            times.append(1000 * (time.perf_counter() - t1))
        latency[name] = {"median_ms": float(np.median(times)), "p95_ms": float(np.quantile(times, 0.95))}
    jdump(latency, A / "latency.json")
    preds_va.to_csv(A / "predictions_valid.csv", index=False)
    preds_te.to_csv(A / "predictions_test.csv", index=False)
    pd.DataFrame(conf_rows).to_csv(A / "confusion_test.csv", index=False)
    pd.DataFrame(cost_rows).to_csv(A / "cost_test.csv", index=False)
    jdump(metrics, A / "metrics_test.json")
    jdump(curves, A / "curves_test.json")
    jdump(rel, A / "reliability_test.json")

    # INR sensitivity: for each review cost, best achievable test benefit by any threshold (oracle-
    # optimistic) versus the validation-chosen threshold, for the best-AUC model.
    sens = []
    for cr in C.REVIEW_GRID_INR:
        for name in [n for n in raw_scores if n != "Prevalence baseline"]:
            thr = best_cost_threshold(yva, preds_va[name].to_numpy(), amount[split.valid], cr)
            nb = net_benefit(yte, preds_te[name].to_numpy(), amount[split.test], thr, cr)
            sens.append({"cost_review": cr, "model": name, **nb,
                         "break_even_precision": break_even_precision(cr, mean_fraud_amt_train)})
    pd.DataFrame(sens).to_csv(A / "cost_sensitivity.csv", index=False)

    # Feature-group ablation (LightGBM, real labels)
    log("Feature-group ablation (real labels)")
    abl = []
    lgb_params = selected["LightGBM"]["params"]
    full = make_model("LightGBM", **lgb_params).fit(Xtr, ytr)
    abl.append({"setting": "real", "dropped": "None (all features)",
                "test_auc": roc_auc_score(yte, full.predict_proba(Xte)[:, 1])})
    ablation_models = {}
    for g, cols in FEATURE_GROUPS.items():
        keep = [c for c in ALL_FEATURES if c not in cols]
        m = _fit_subset("LightGBM", lgb_params, Xtr, ytr, keep)
        abl.append({"setting": "real", "dropped": g,
                    "test_auc": roc_auc_score(yte, m.predict_proba(Xte[keep])[:, 1])})
        ablation_models[g] = keep

    # ------------------------------------------------------------------ 5. positive control
    log("Positive-control experiment (semi-synthetic labels)")
    seg_p95 = float(np.quantile(Xtr["seg_count_1h"], 0.95))
    z = risk_score(X, df, seg_p95)
    prevalence = float(y.mean())
    pc_rows = []
    pc_models = {"Logistic Regression": {"C": 0.1}, "Random Forest": {"min_samples_leaf": 50, "n_estimators": 200},
                 "LightGBM": {"num_leaves": 15}}
    strengths = C.SIGNAL_STRENGTHS if not fast else [0.0, 1.5]
    for s in strengths:
        for r in range(reps):
            ys, ptrue = simulate_labels(z, s, prevalence, seed=1000 + r)
            ys_tr, ys_va, ys_te = ys[split.train], ys[split.valid], ys[split.test]
            oracle_auc = roc_auc_score(ys_te, ptrue[split.test])
            oracle_ap = average_precision_score(ys_te, ptrue[split.test])
            for name, params in pc_models.items():
                m = make_model(name, **params).fit(Xtr, ys_tr)
                sva, ste = m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]
                cal = PlattCalibrator().fit(sva, ys_va)
                pva, pte = cal.transform(sva), cal.transform(ste)
                thr = best_cost_threshold(ys_va, pva, amount[split.valid])
                nb = net_benefit(ys_te, pte, amount[split.test], thr)
                pc_rows.append({"strength": s, "replicate": r, "model": name,
                                "test_frauds": int(ys_te.sum()),
                                "test_auc": roc_auc_score(ys_te, pte),
                                "test_ap": average_precision_score(ys_te, pte),
                                "auc_p_value": auc_pvalue(ys_te, pte),
                                "oracle_auc": oracle_auc, "oracle_ap": oracle_ap,
                                "net_benefit_inr": nb["net_benefit"], "flagged": nb["flagged"],
                                "frauds_caught": nb["frauds_caught"]})
            log(f"   s={s} rep={r}: oracle AUC {oracle_auc:.3f}; " + ", ".join(
                f"{row['model']} {row['test_auc']:.3f}" for row in pc_rows[-3:]))
    pd.DataFrame(pc_rows).to_csv(A / "positive_control.csv", index=False)

    # Ablation + permutation importance on the planted-signal setting
    s_ref = C.PC_CNN_STRENGTH
    ys, ptrue = simulate_labels(z, s_ref, prevalence, seed=1000)
    ys_tr, ys_va, ys_te = ys[split.train], ys[split.valid], ys[split.test]
    full_pc = make_model("LightGBM", num_leaves=15).fit(Xtr, ys_tr)
    abl.append({"setting": f"positive control s={s_ref}", "dropped": "None (all features)",
                "test_auc": roc_auc_score(ys_te, full_pc.predict_proba(Xte)[:, 1])})
    for g, keep in ablation_models.items():
        m = _fit_subset("LightGBM", {"num_leaves": 15}, Xtr, ys_tr, keep)
        abl.append({"setting": f"positive control s={s_ref}", "dropped": g,
                    "test_auc": roc_auc_score(ys_te, m.predict_proba(Xte[keep])[:, 1])})
    pd.DataFrame(abl).to_csv(A / "ablation.csv", index=False)

    log("Permutation importance (LightGBM)")
    imp_rows = []
    for setting, model, yy in [("real", full, yte), (f"positive control s={s_ref}", full_pc, ys_te)]:
        pi = permutation_importance(model, Xte, yy, scoring="roc_auc", n_repeats=5 if not fast else 2,
                                    random_state=C.SEED, n_jobs=1)
        for f, mu, sd in zip(ALL_FEATURES, pi.importances_mean, pi.importances_std):
            imp_rows.append({"setting": setting, "feature": f, "importance_mean": mu, "importance_sd": sd})
    pd.DataFrame(imp_rows).to_csv(A / "permutation_importance.csv", index=False)

    # CNN feature-order sensitivity
    order_rows = []
    if TORCH_OK:
        log("1D-CNN feature-order sensitivity")
        rng = np.random.default_rng(C.SEED)
        nfeat = Ztr.shape[1]
        orders = [None] + [rng.permutation(nfeat) for _ in range(C.CNN_ORDERINGS - 1 if not fast else 1)]
        for setting, (a, b, c) in [("real", (ytr, yva, yte)),
                                   (f"positive control s={s_ref}", (ys_tr, ys_va, ys_te))]:
            for k, o in enumerate(orders):
                m = TabularCNN(seed=C.SEED, order=o).fit(Ztr, a, Zva, b)
                order_rows.append({"setting": setting, "ordering": "canonical" if o is None else f"random #{k}",
                                   "test_auc": roc_auc_score(c, m.predict_proba(Zte)[:, 1]),
                                   "test_ap": average_precision_score(c, m.predict_proba(Zte)[:, 1])})
                log(f"   {setting} order {k}: AUC {order_rows[-1]['test_auc']:.4f}")
            for name, params in [("MLP", {"alpha": 1e-3}), ("LightGBM", {"num_leaves": 15})]:
                m = make_model(name, **params).fit(Xtr, a)
                order_rows.append({"setting": setting, "ordering": f"{name} (order-invariant reference)",
                                   "test_auc": roc_auc_score(c, m.predict_proba(Xte)[:, 1]),
                                   "test_ap": average_precision_score(c, m.predict_proba(Xte)[:, 1])})
    pd.DataFrame(order_rows).to_csv(A / "cnn_order.csv", index=False)

    # ------------------------------------------------------------------ 6. manifest
    import lightgbm
    manifest = {
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fast_mode": fast,
        "dataset_file": C.DATA_PATH.name,
        "dataset_sha256": sha256_of_file(C.DATA_PATH),
        "checks": checks,
        "split": st.to_dict(orient="records"),
        "seed": C.SEED,
        "features": ALL_FEATURES,
        "excluded_columns": C.EXCLUDED,
        "cost_review_inr": C.COST_REVIEW_INR,
        "mean_fraud_amount_train_inr": mean_fraud_amt_train,
        "break_even_precision": break_even_precision(C.COST_REVIEW_INR, mean_fraud_amt_train),
        "risk_score_terms": RISK_SCORE_TERMS,
        "seg_count_1h_p95_train": seg_p95,
        "cnn": cnn_info,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
                     "lightgbm": lightgbm.__version__,
                     "torch": __import__("torch").__version__ if TORCH_OK else None},
        "runtime_seconds": round(time.time() - T0, 1),
    }
    jdump(manifest, A / "manifest.json")
    log("Done.")


def _fit_subset(name, params, Xtr, ytr, keep):
    """Fit a model on a column subset by building a pipeline restricted to those columns."""
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder
    from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
    base = make_model(name, **params)
    nums = [c for c in NUMERIC_FEATURES if c in keep]
    cats = [c for c in CATEGORICAL_FEATURES if c in keep]
    pre = ColumnTransformer([("num", "passthrough", nums),
                             ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cats)])
    pipe = Pipeline([("pre", pre), ("clf", clone(base.named_steps["clf"]))])
    return pipe.fit(Xtr[keep], ytr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    main(**vars(ap.parse_args()))
