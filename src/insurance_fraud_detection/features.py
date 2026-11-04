import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class WOEEncoder(BaseEstimator, TransformerMixin):
    """
    Weight of evidence encoder with IV-based column selection.

    WOE = ln(share of all frauds in the bin / share of all non-frauds in the bin).
    A positive value means the bin is riskier than average.

    Everything is learned in fit() from the data it is given, so inside a
    Pipeline, cross-validation refits it per fold with no leakage.

    Parameters
    ----------
    cols : columns to encode (treated as categories).
    alpha : smoothing added to each bin's counts, so a bin with zero
            frauds does not give an infinite WOE. Use 0 to switch it off.
    min_iv : columns whose IV is below this are dropped in transform().

    """

    def __init__(self, cols: list[str], alpha: float = 0.5, min_iv: float = 0.0):
        self.cols = cols
        self.alpha = alpha
        self.min_iv = min_iv

    def fit(self, X: pd.DataFrame, y) -> "WOEEncoder":
        y = pd.Series(np.asarray(y), index=X.index)
        self.woe_maps_: dict[str, dict] = {}
        self.iv_: dict[str, float] = {}

        for col in self.cols:
            stats = y.groupby(X[col]).agg(bad="sum", n="count")
            stats["good"] = stats["n"] - stats["bad"]
            k = len(stats)

            dist_bad = (stats["bad"] + self.alpha) / (stats["bad"].sum() + self.alpha * k)
            dist_good = (stats["good"] + self.alpha) / (stats["good"].sum() + self.alpha * k)
            woe = np.log(dist_bad / dist_good)

            self.woe_maps_[col] = woe.to_dict()
            self.iv_[col] = float(((dist_bad - dist_good) * woe).sum())

        self.selected_cols_ = [c for c in self.cols if self.iv_[c] >= self.min_iv]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "woe_maps_")
        out = pd.DataFrame(index=X.index)
        for col in self.selected_cols_:
            # unseen category -> 0.0, meaning "no evidence either way"
            out[col] = X[col].map(self.woe_maps_[col]).fillna(0.0)
        return out