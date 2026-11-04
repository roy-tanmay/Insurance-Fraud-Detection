import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def top_frac_capture(y_true, scores, frac: float = 0.10) -> int:
    """Number of frauds caught in the top `frac` of claims by score."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    k = int(np.ceil(len(y_true) * frac))
    top_idx = np.argsort(scores)[::-1][:k]
    return int(y_true[top_idx].sum())


def score_report(y_true, scores, frac: float = 0.10) -> dict:
    """PR-AUC, ROC-AUC, Gini, KS, top-frac capture and lift."""
    y_true = np.asarray(y_true)
    caught = top_frac_capture(y_true, scores, frac)
    total = int(y_true.sum())
    roc = float(roc_auc_score(y_true, scores))
    fpr, tpr, _ = roc_curve(y_true, scores)
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": roc,
        "gini": 2 * roc - 1,
        "ks": float(np.max(tpr - fpr)),
        "top_caught": caught,
        "total_fraud": total,
        "lift": (caught / total) / frac,
    }