import pandas as pd
import pytest

from insurance_fraud_detection.preprocessing import (
    fit_cleaner, apply_cleaner, apply_pooling, map_ordinal,
)

def test_fit_cleaner_ignores_zero_ages():
    df = pd.DataFrame({"age": [0, 0, 30, 40, 50]})
    params = fit_cleaner(df)
    assert params["age"] == 40


def test_apply_cleaner_replaces_only_zero_ages():
    df = pd.DataFrame({"age": [0, 35, 60]})
    out = apply_cleaner(df, {"age": 40})
    assert out["age"].tolist() == [40, 35, 60]
    assert df["age"].tolist() == [0, 35, 60]

def test_apply_pooling_merges_sparse_bin_and_keeps_others():
    df = pd.DataFrame({"number_of_cars": ["5 to 8", "1 vehicle"]})
    pool_map = {"number_of_cars": {"5 to 8": "3 or more vehicles"}}
    out = apply_pooling(df, pool_map)
    assert out["number_of_cars"].tolist() == ["3 or more vehicles", "1 vehicle"]


def test_map_ordinal_maps_in_order():
    df = pd.DataFrame({"number_of_cars": ["1 vehicle", "3 or more vehicles"]})
    orders = {"number_of_cars": ["1 vehicle", "2 vehicles", "3 or more vehicles"]}
    assert map_ordinal(df, orders)["number_of_cars"].tolist() == [0, 2]


def test_map_ordinal_raises_on_unseen_label():
    df = pd.DataFrame({"number_of_cars": ["1 vehicle", "7 vehicles"]})
    orders = {"number_of_cars": ["1 vehicle", "2 vehicles", "3 or more vehicles"]}
    with pytest.raises(ValueError):
        map_ordinal(df, orders)