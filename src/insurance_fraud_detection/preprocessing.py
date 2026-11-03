
import pandas as pd
import json
from pathlib import Path
from insurance_fraud_detection.config import BIN_ORDERS, RAW_CSV, TARGET, SEED, TEST_SIZE, DROP_COLUMNS,  POOL_MAP


def fit_cleaner(train_df: pd.DataFrame) -> dict: 
    """
    Learn cleaning parameters from the training set only.
    """
    age_median = float(train_df.loc[train_df["age"] > 0, "age"].median())
    return {"age": age_median}



def save_params(params: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(params, indent=2))

def load_params(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def apply_cleaner(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Apply cleaning parameters to a dataframe.
    """
    df = df.copy()
    df.loc[df["age"] == 0, "age"] = params["age"]
    return df

def apply_pooling(df: pd.DataFrame) -> pd.DataFrame:
    """Fixed rule: merge sparse bins using POOL_MAP. Learns nothing."""
    df = df.copy()
    for col, mapping in POOL_MAP.items():
        if col not in df.columns:
            raise KeyError(f"Pooling column missing: {col}")
        df[col] = df[col].map(lambda v: mapping.get(v, v)) # type: ignore
    return df

def map_ordinal( df: pd.DataFrame,
                bin_orders: dict[str, list[str]] = BIN_ORDERS,
                ) -> pd.DataFrame:
    """
    Fixed rule: map ordered bin labels to integer codes. Raises on unseen labels.
    """

    df = df.copy()
    for col, order in bin_orders.items():
        codes = {label: i for i, label in enumerate(order)}
        mapped = df[col].map(codes)
        unseen = df.loc[mapped.isna(), col].unique()
        if len(unseen) > 0:
            raise ValueError(f"Unseen labels in '{col}': {list(unseen)}")
        df[col] = mapped.astype(int)
    return df


def prepare(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = apply_cleaner(df, params)
    df = apply_pooling(df)
    return map_ordinal(df)