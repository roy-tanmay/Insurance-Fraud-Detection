import numpy as np

from insurance_fraud_detection.evaluate import score_report, top_frac_capture


def test_top_frac_capture_counts_correctly() -> None:
    y = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    s = np.array([0.9, 0.1, 0.2, 0.8, 0.3, 0.4, 0.05, 0.06, 0.07, 0.08])
    # top 20% = 2 claims = scores 0.9 and 0.8, both fraud
    assert top_frac_capture(y, s, frac=0.2) == 2
    # top 10% = 1 claim = score 0.9, which is fraud
    assert top_frac_capture(y, s, frac=0.1) == 1


def test_score_report_perfect_ranking() -> None:
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    r = score_report(y, s, frac=0.2)
    assert r["roc_auc"] == 1.0
    assert r["top_caught"] == 2
    assert r["total_fraud"] == 2