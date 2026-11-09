import json
from functools import lru_cache

import joblib
import pandas as pd

from insurance_fraud_detection.config import (
    ARTIFACTS_DIR,
    CLEANER_PARAMS_PATH,
    SELECTED_COLS,
    TARGET,
)
from insurance_fraud_detection.preprocessing import load_params, prepare

DEFAULTS = {
    "week_of_month": 1,
    "day_of_week": "Thursday",
    "accident_area": "Urban",
    "day_of_week_claimed": "Friday",
    "week_of_month_claimed": 3,
    "sex": "Male",
    "marital_status": "Married",
    "age": 41,
    "policy_type": "Sedan - Liability",
    "vehicle_category": "Sport",
    "driver_rating": 4,
    "days_policy_accident": "more than 30",
    "days_policy_claim": "more than 30",
    "police_report_filed": "No",
    "witness_present": "No",
    "number_of_cars": "1 vehicle",
    "year": 1995,
}


@lru_cache(maxsize=1)
def load_artifacts():
    pipe = joblib.load(ARTIFACTS_DIR / "pipeline.pkl")
    cutoff = json.loads((ARTIFACTS_DIR / "threshold.json").read_text())["cutoff"]
    params = load_params(CLEANER_PARAMS_PATH)
    return pipe, cutoff, params


def predict_one(claim: dict) -> dict:
    """Score one claim. Needs the 13 selected columns; the rest are filled from DEFAULTS."""
    pipe, cutoff, params = load_artifacts()
    raw = pd.DataFrame([{**DEFAULTS, **claim}]).drop(columns=[TARGET], errors="ignore")
    X = prepare(raw, params)[SELECTED_COLS]
    score = float(pipe.predict_proba(X)[:, 1][0])
    return {"score": score, "flag": score >= cutoff, "cutoff": float(cutoff)}