import numpy as np
import pytest

from insurance_fraud_detection.config import ARTIFACTS_DIR, RAW_CSV, SELECTED_COLS, TARGET
from insurance_fraud_detection.data_prep import drop_bad_rows, load_data, split_data
from insurance_fraud_detection.predict import load_artifacts, predict_one
from insurance_fraud_detection.preprocessing import prepare

pytestmark = pytest.mark.skipif(
    not (ARTIFACTS_DIR / "pipeline.pkl").exists() or not RAW_CSV.exists(),
    reason="needs data/ and trained artifacts (not available in CI)",
)


def test_single_claim_matches_batch_score() -> None:
    _, test_raw = split_data(load_data())
    test_raw = drop_bad_rows(test_raw).head(5)

    pipe, _, params = load_artifacts()
    X = prepare(test_raw.drop(columns=[TARGET]), params)[SELECTED_COLS]
    expected = pipe.predict_proba(X)[:, 1]

    got = [predict_one(row.to_dict())["score"] for _, row in test_raw.iterrows()]
    assert np.allclose(got, expected)