import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def top_frac_capture(y_true, scores, frac: float = 0.10) -> int:
    """Number of frauds caught in the top `frac` of claims by score."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    k = int(np.ceil(len(y_true) * frac))
    top_idx = np.argsort(scores)[::-1][:k]
    return int(y_true[top_idx].sum())


def score_report(y_true, scores, frac: float = 0.10) -> dict:
    """PR-AUC, ROC-AUC, top-frac capture and lift."""
    y_true = np.asarray(y_true)
    caught = top_frac_capture(y_true, scores, frac)
    total = int(y_true.sum())
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "top_caught": caught,
        "total_fraud": total,
        "lift": (caught / total) / frac,
    }