from pathlib import Path
import re
import pandas as pd
from sklearn.model_selection import train_test_split

from insurance_fraud_detection.config import BIN_ORDERS, RAW_CSV, TARGET, SEED, TEST_SIZE, DROP_COLUMNS,  POOL_MAP

def to_snake_case(name: str) -> str:
    """
    Convert a string to snake_case, e.g. 'FraudFound_P' -> 'fraud_found_p'.
    """
    name = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    return re.sub(r"_+", "_", name)


def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that are not useful for model training.
    Raises KeyError if any name in DROP_COLUMNS is missing.
    """
    return df.drop(columns=DROP_COLUMNS)



def load_data(path: Path = RAW_CSV) -> pd.DataFrame:
    """
    Load the raw CSV, convert column names to snake_case, drop unused columns.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Data file not found at {path}. "
            "Download fraud_oracle.csv into the data/ folder."
        ) from None

    df.columns = [to_snake_case(col) for col in df.columns]

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found. Columns: {list(df.columns)}"
        )

    return drop_unnecessary_columns(df)


def split_data(df: pd.DataFrame):
    """
    Stratified train/test split. Returns two dataframes (target kept as a column)
    """
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found.")
    
    return train_test_split(
        df, test_size=TEST_SIZE, random_state=SEED, stratify=df[TARGET]
    )


def drop_bad_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with the invalid '0' string in month_claimed
    (the same row also has '0' in day_of_week_claimed).
    """
    return df[df["month_claimed"] != "0"].copy()


