"""Train and evaluate the station-aware complaint trend classifier."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = Path(__file__).resolve().parent
TRAINING_FILE = DATA_DIR / "complaint_trend_training.csv"
RAW_FILE = DATA_DIR / "synthetic_complaints_raw.csv"
MODEL_FILE = OUTPUT_DIR / "complaint_trend_logistic_regression.joblib"
METRICS_FILE = OUTPUT_DIR / "model_evaluation.json"
PREDICTIONS_FILE = OUTPUT_DIR / "next_month_predictions.csv"

FEATURE_COLUMNS = [
    "stationID",
    "complaintCategory",
    "currentMonthCount",
    "previousMonthCount",
    "twoMonthsAgoCount",
    "threeMonthAverage",
    "sixMonthAverage",
    "recentTrend",
]
CATEGORICAL_COLUMNS = ["stationID", "complaintCategory"]
NUMERIC_COLUMNS = [column for column in FEATURE_COLUMNS if column not in CATEGORICAL_COLUMNS]


def build_model() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_COLUMNS,
            ),
            ("numeric", StandardScaler(), NUMERIC_COLUMNS),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def load_training_data() -> pd.DataFrame:
    data = pd.read_csv(TRAINING_FILE)
    missing = set(FEATURE_COLUMNS + ["month", "targetIncrease"]) - set(data.columns)
    if missing:
        raise ValueError(f"Training data is missing columns: {sorted(missing)}")
    data["month"] = pd.to_datetime(data["month"], format="%Y-%m")
    data["targetIncrease"] = data["targetIncrease"].astype(int)
    return data.sort_values(["month", "stationID", "complaintCategory"]).reset_index(drop=True)


def make_features_from_raw_data() -> pd.DataFrame:
    raw = pd.read_csv(RAW_FILE)
    raw["month"] = pd.to_datetime(raw["month"], format="%Y-%m")
    counts = (
        raw.groupby(["stationID", "stationName", "month", "complaintCategory"])
        .size()
        .rename("currentMonthCount")
        .reset_index()
    )
    stations = raw[["stationID", "stationName"]].drop_duplicates()
    categories = sorted(raw["complaintCategory"].unique())
    months = pd.date_range(raw["month"].min(), raw["month"].max(), freq="MS")
    complete = pd.MultiIndex.from_product(
        [
            stations["stationID"].sort_values(),
            months,
            categories,
        ],
        names=["stationID", "month", "complaintCategory"],
    ).to_frame(index=False)
    complete = complete.merge(stations, on="stationID", how="left")
    complete = complete.merge(
        counts,
        on=["stationID", "stationName", "month", "complaintCategory"],
        how="left",
    )
    complete["currentMonthCount"] = complete["currentMonthCount"].fillna(0).astype(int)
    complete = complete.sort_values(["stationID", "complaintCategory", "month"])
    grouped = complete.groupby(["stationID", "complaintCategory"], group_keys=False)
    complete["previousMonthCount"] = grouped["currentMonthCount"].shift(1).fillna(0)
    complete["twoMonthsAgoCount"] = grouped["currentMonthCount"].shift(2).fillna(0)
    complete["threeMonthAverage"] = (
        grouped["currentMonthCount"]
        .transform(lambda values: values.shift(1).rolling(3, min_periods=3).mean())
        .fillna(0)
    )
    complete["sixMonthAverage"] = (
        grouped["currentMonthCount"]
        .transform(lambda values: values.shift(1).rolling(6, min_periods=6).mean())
        .fillna(0)
    )
    complete["recentTrend"] = complete["currentMonthCount"] - complete["threeMonthAverage"]
    return complete.reset_index(drop=True)


def evaluate(model: Pipeline, test: pd.DataFrame) -> dict[str, float | int | str]:
    probabilities = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    labels = (probabilities >= 0.5).astype(int)
    actual = test["targetIncrease"].to_numpy()
    return {
        "training_period": "2026-04 to 2026-06",
        "testing_period": "2026-07 to 2026-08",
        "training_rows": 72,
        "testing_rows": int(len(test)),
        "accuracy": round(float(accuracy_score(actual, labels)), 4),
        "precision": round(float(precision_score(actual, labels, zero_division=0)), 4),
        "recall": round(float(recall_score(actual, labels, zero_division=0)), 4),
        "f1": round(float(f1_score(actual, labels, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(actual, probabilities)), 4),
        "brier_score": round(float(brier_score_loss(actual, probabilities)), 4),
        "warning": "Evaluation is experimental because only five usable training months are available.",
    }


def main() -> None:
    training = load_training_data()
    train = training[training["month"] <= "2026-06-01"]
    test = training[training["month"] >= "2026-07-01"]
    if train["targetIncrease"].nunique() < 2:
        raise ValueError("Training split must contain both target classes.")
    if test["targetIncrease"].nunique() < 2:
        raise ValueError("Testing split must contain both target classes.")

    model = build_model()
    model.fit(train[FEATURE_COLUMNS], train["targetIncrease"])
    metrics = evaluate(model, test)

    inference = make_features_from_raw_data()
    latest_month = inference["month"].max()
    current_month = pd.Timestamp(datetime.now().strftime("%Y-%m-01"))
    next_month = current_month + pd.offsets.MonthBegin(1)
    latest = inference[inference["month"] == latest_month].copy()
    latest["probabilityOfIncrease"] = model.predict_proba(latest[FEATURE_COLUMNS])[:, 1]
    latest["predictedIncrease"] = latest["probabilityOfIncrease"] >= 0.5
    latest["riskLevel"] = pd.cut(
        latest["probabilityOfIncrease"],
        bins=[-0.01, 0.4, 0.7, 1.0],
        labels=["Low", "Medium", "High"],
    ).astype(str)
    predictions = latest[
        ["stationID", "stationName", "complaintCategory", "probabilityOfIncrease", "predictedIncrease", "riskLevel"]
    ].copy()
    predictions.insert(2, "forecastMonth", next_month.strftime("%Y-%m"))
    predictions["probabilityOfIncrease"] = predictions["probabilityOfIncrease"].round(6)
    predictions.to_csv(PREDICTIONS_FILE, index=False)

    joblib.dump(
        {
            "model": model,
            "featureColumns": FEATURE_COLUMNS,
            "threshold": 0.5,
            "modelVersion": "complaint-trend-logistic-v1",
            "trainedThrough": "2026-06",
        },
        MODEL_FILE,
    )
    METRICS_FILE.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved model: {MODEL_FILE}")
    print(f"Saved predictions: {PREDICTIONS_FILE}")


if __name__ == "__main__":
    main()

