import math

import pandas as pd
import pytest

from insurance_fraud_detection.features import WOEEncoder


def _toy():
    # x: 2 frauds, 1 non-fraud. y: 1 fraud, 4 non-frauds.
    # Totals: 3 frauds, 5 non-frauds.
    X = pd.DataFrame({"a": ["x"] * 3 + ["y"] * 5})
    y = pd.Series([1, 1, 0, 1, 0, 0, 0, 0])
    return X, y


def test_woe_matches_hand_calculation():
    X, y = _toy()
    enc = WOEEncoder(cols=["a"], alpha=0).fit(X, y)
    # WOE(x) = ln((2/3) / (1/5)),  WOE(y) = ln((1/3) / (4/5))
    assert enc.woe_maps_["a"]["x"] == pytest.approx(math.log(10 / 3))
    assert enc.woe_maps_["a"]["y"] == pytest.approx(math.log(5 / 12))


def test_unseen_category_gets_zero():
    X, y = _toy()
    enc = WOEEncoder(cols=["a"], alpha=0).fit(X, y)
    out = enc.transform(pd.DataFrame({"a": ["z"]}))
    assert out["a"].iloc[0] == 0.0


def test_min_iv_drops_weak_columns():
    X, y = _toy()
    enc = WOEEncoder(cols=["a"], alpha=0, min_iv=5.0).fit(X, y)
    assert enc.selected_cols_ == []