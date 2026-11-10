"""Fast correctness tests:  pytest -q"""
import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.audit import benjamini_hochberg
from src.data import load_raw, time_split, validate
from src.evaluate import break_even_precision, confusion, ece_quantile, net_benefit
from src.features import ALL_FEATURES, FeatureBuilder, _past_window_count, add_velocity_features
from src.positive_control import simulate_labels


@pytest.fixture(scope="module")
def df():
    return load_raw()


def test_schema_and_counts(df):
    chk = validate(df)
    assert chk["rows"] == 250_000
    assert chk["fraud_count"] == 480
    assert chk["missing_values"] == 0 and chk["duplicate_ids"] == 0
    assert chk["hour_consistent_with_timestamp"]


def test_split_is_disjoint_ordered_and_complete(df):
    s = time_split(df)
    assert len(set(s.train) | set(s.valid) | set(s.test)) == len(df)
    assert df["timestamp"].iloc[s.train].max() < df["timestamp"].iloc[s.valid].min()
    assert df["timestamp"].iloc[s.valid].max() < df["timestamp"].iloc[s.test].min()


def test_no_post_authorisation_feature():
    assert "transaction_status" not in ALL_FEATURES
    assert C.TARGET not in ALL_FEATURES


def test_velocity_is_strictly_past():
    t = np.array([0, 10, 10, 20, 4000], dtype=float)
    # the event itself and simultaneous events are excluded; window is (t-w, t)
    assert _past_window_count(t, 3600).tolist() == [0, 1, 1, 3, 0]


def test_velocity_ignores_future_rows(df):
    small = df.iloc[:5000].copy()
    full = add_velocity_features(small)
    trunc = add_velocity_features(small.iloc[:2500])
    pd.testing.assert_series_equal(full["seg_count_1h"].iloc[:2500], trunc["seg_count_1h"], check_names=False)


def test_feature_builder_uses_train_stats_only(df):
    s = time_split(df)
    fb = FeatureBuilder().fit(df.iloc[s.train])
    expected = df.iloc[s.train].groupby("merchant_category")["amount_inr"].median().to_dict()
    assert fb.cat_median_ == expected


def test_confusion_and_precision_identity():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 1])
    p = np.array([.9, .2, .8, .1, .1, .1, .1, .1, .1, .7])
    cm = confusion(y, p, 0.5)
    assert (cm["TP"], cm["FP"], cm["FN"], cm["TN"]) == (2, 1, 1, 6)
    assert cm["precision"] == pytest.approx(cm["precision_identity"])


def test_v1_precision_claim_is_inconsistent():
    """Reviewer's check: prevalence 0.192 %, recall 97.69 %, specificity 99.39 % => precision ~23.6 %."""
    pi, r, sp = 0.00192, 0.9769, 0.9939
    prec = r * pi / (r * pi + (1 - sp) * (1 - pi))
    assert prec == pytest.approx(0.236, abs=0.001)


def test_bh_monotone():
    adj = benjamini_hochberg(np.array([0.01, 0.04, 0.03, 0.5]))
    assert np.all(adj >= np.array([0.01, 0.04, 0.03, 0.5]))
    assert adj.max() <= 1


def test_cost_model():
    y = np.array([1, 0, 0]); p = np.array([.9, .8, .1]); a = np.array([1000., 50., 50.])
    nb = net_benefit(y, p, a, 0.5, cost_review=25)
    assert nb["net_benefit"] == pytest.approx(1000 - 50)
    assert break_even_precision(25, 1000) == pytest.approx(0.025)


def test_ece_perfect():
    p = np.repeat([0.1, 0.5], 1000)
    y = np.concatenate([np.r_[np.ones(100), np.zeros(900)], np.r_[np.ones(500), np.zeros(500)]])
    assert ece_quantile(y, p, bins=2) == pytest.approx(0.0)


def test_simulated_prevalence():
    z = np.random.default_rng(0).standard_normal(200_000)
    _, ptrue = simulate_labels(z, 1.5, 0.00192, seed=1)
    assert ptrue.mean() == pytest.approx(0.00192, rel=1e-6)
