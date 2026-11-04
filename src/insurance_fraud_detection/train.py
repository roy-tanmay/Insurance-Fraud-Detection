import json

import joblib
import mlflow
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from insurance_fraud_detection.config import (
    ARTIFACTS_DIR,
    CLEANER_PARAMS_PATH,
    SELECTED_COLS,
    SEED,
    TARGET,
)
from insurance_fraud_detection.data_prep import drop_bad_rows, load_data, split_data
from insurance_fraud_detection.evaluate import score_report
from insurance_fraud_detection.features import WOEEncoder
from insurance_fraud_detection.preprocessing import load_params, prepare

XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED,
)


def build_datasets():
    """Same steps as the notebooks: load, split, drop bad row, apply cleaner."""
    df = load_data()
    train_raw, test_raw = split_data(df)
    train_raw = drop_bad_rows(train_raw)
    test_raw = drop_bad_rows(test_raw)
    params = load_params(CLEANER_PARAMS_PATH)
    return prepare(train_raw, params), prepare(test_raw, params)


def main() -> None:
    train, test = build_datasets()
    print(train.shape, test.shape)  # expect (12335, 31) and (3084, 31)

    X_train, y_train = train[SELECTED_COLS], train[TARGET]
    X_test, y_test = test[SELECTED_COLS], test[TARGET]

    pipe = Pipeline([
        ("woe", WOEEncoder(cols=SELECTED_COLS)),
        ("clf", XGBClassifier(**XGB_PARAMS)),
    ])

    # cutoff for "top 10%", taken from train only, so test is not used
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    cutoff = float(np.quantile(oof, 0.90))

    pipe.fit(X_train, y_train)
    report = score_report(y_test, pipe.predict_proba(X_test)[:, 1])
    for k, v in report.items():
        print(f"{k:<12} {v:.3f}" if isinstance(v, float) else f"{k:<12} {v}")

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    model_path = ARTIFACTS_DIR / "pipeline.pkl"
    joblib.dump(pipe, model_path)
    (ARTIFACTS_DIR / "threshold.json").write_text(json.dumps({"cutoff": cutoff}))

    mlflow.set_experiment("insurance-fraud")
    with mlflow.start_run():
        mlflow.log_params(XGB_PARAMS)
        mlflow.log_metrics({**report, "cutoff": cutoff})
        mlflow.log_artifact(str(model_path))


if __name__ == "__main__":
    main()