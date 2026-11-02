import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from insurance_fraud_detection.config import RAW_CSV, TARGET, SEED, TEST_SIZE

def to_snake_case(name: str) -> str:
    """
    Convert a string to snake_case, e.g. 'FraudFound_P' -> 'fraud_found_p'.
    """
    name = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    return re.sub(r"_+", "_", name)

def load_data(path= RAW_CSV) -> pd.DataFrame:
    """
    Load the raw CSV data and convert column names to snake_case.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Data File not found at {path}."
            "Download fraud_oracle.csv into the data/ folder.")
    df.columns = [to_snake_case(col) for col in df.columns]
    return df

def split_data(df: pd.DataFrame):
    """
    Split the data into training and testing sets.
    """
    return train_test_split(
        df, test_size=TEST_SIZE, random_state=SEED, stratify=df[TARGET]
    )

 