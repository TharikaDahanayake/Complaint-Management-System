"""Train and evaluate the station-aware complaint trend classifier."""

from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
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

RAW_COLUMNS = [
    "complaintId",
    "stationID",
    "stationName",
    "stationDistrict",
    "stationDivision",
    "complaintCategory",
    "incidentDate",
    "createdAt",
    "latitude",
    "longitude",
    "status",
    "dataSource",
    "month",
]
FEATURE_COLUMNS = [
    "stationID",
    "complaintCategory",
    "currentMonthCount",
    "previousMonthCount",
    "twoMonthsAgoCount",
    "threeMonthAverage",
    "sixMonthAverage",
    "recentTrend",
    "percentageChange",
    "rollingStd3",
    "increaseCount3",
    "monthSin",
    "monthCos",
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


def load_raw_data() -> pd.DataFrame:
    data = pd.read_csv(RAW_FILE)
    missing = set(RAW_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"Raw complaint data is missing columns: {sorted(missing)}")
    if data["complaintId"].duplicated().any():
        raise ValueError("Raw complaint data contains duplicate complaintId values.")

    data["incidentDate"] = pd.to_datetime(data["incidentDate"], errors="coerce")
    if data["incidentDate"].isna().any():
        raise ValueError("Raw complaint data contains invalid incidentDate values.")
    data["month"] = data["incidentDate"].dt.to_period("M").dt.to_timestamp()
    if data[["stationID", "complaintCategory"]].isna().any().any():
        raise ValueError("Raw complaint data contains missing station or category values.")
    return data


def make_features(raw: pd.DataFrame) -> pd.DataFrame:
    counts = (
        raw.groupby(["stationID", "stationName", "month", "complaintCategory"])
        .size()
        .rename("currentMonthCount")
        .reset_index()
    )
    stations = raw[["stationID", "stationName"]].drop_duplicates("stationID")
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
    previous = grouped["currentMonthCount"].shift(1)
    complete["previousMonthCount"] = previous.fillna(0)
    complete["twoMonthsAgoCount"] = grouped["currentMonthCount"].shift(2).fillna(0)
    complete["threeMonthAverage"] = (
        grouped["currentMonthCount"]
        .transform(lambda values: values.shift(1).rolling(3, min_periods=1).mean())
        .fillna(0)
    )
    complete["sixMonthAverage"] = (
        grouped["currentMonthCount"]
        .transform(lambda values: values.shift(1).rolling(6, min_periods=1).mean())
        .fillna(0)
    )
    complete["recentTrend"] = complete["currentMonthCount"] - complete["threeMonthAverage"]
    complete["percentageChange"] = (
        complete["currentMonthCount"] - complete["previousMonthCount"]
    ) / complete["previousMonthCount"].replace(0, 1)
    complete["rollingStd3"] = (
        grouped["currentMonthCount"]
        .transform(lambda values: values.shift(1).rolling(3, min_periods=2).std())
        .fillna(0)
    )
    changes = complete["currentMonthCount"] - complete["previousMonthCount"]
    complete["increaseCount3"] = (
        changes.groupby([complete["stationID"], complete["complaintCategory"]])
        .transform(lambda values: values.shift(1).gt(0).rolling(3, min_periods=1).sum())
        .fillna(0)
    )
    month_number = complete["month"].dt.month
    complete["monthSin"] = (month_number * 2 * math.pi / 12).map(math.sin)
    complete["monthCos"] = (month_number * 2 * math.pi / 12).map(math.cos)
    complete["nextMonthCount"] = grouped["currentMonthCount"].shift(-1)
    return complete.reset_index(drop=True)


def make_training_data(raw: pd.DataFrame) -> pd.DataFrame:
    features = make_features(raw)
    features["targetIncrease"] = (
        features["nextMonthCount"].notna()
        & (features["nextMonthCount"] > features["currentMonthCount"])
    ).astype(int)
    return features[features["nextMonthCount"].notna()].drop(columns="nextMonthCount").reset_index(drop=True)


def select_threshold(model: Pipeline, validation: pd.DataFrame) -> float:
    probabilities = model.predict_proba(validation[FEATURE_COLUMNS])[:, 1]
    thresholds = [round(0.25 + index * 0.01, 2) for index in range(51)]
    scores = [
        f1_score(validation["targetIncrease"], probabilities >= threshold, zero_division=0)
        for threshold in thresholds
    ]
    return float(thresholds[scores.index(max(scores))])


def evaluate(
    model: Pipeline,
    test: pd.DataFrame,
    threshold: float,
    training: pd.DataFrame,
    validation: pd.DataFrame,
) -> dict[str, object]:
    probabilities = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    labels = (probabilities >= threshold).astype(int)
    actual = test["targetIncrease"].to_numpy()
    baseline_labels = pd.Series(0, index=test.index)
    return {
        "training_period": training["month"].min().strftime("%Y-%m")
        + " to "
        + training["month"].max().strftime("%Y-%m"),
        "validation_period": validation["month"].min().strftime("%Y-%m")
        + " to "
        + validation["month"].max().strftime("%Y-%m"),
        "testing_period": test["month"].min().strftime("%Y-%m")
        + " to "
        + test["month"].max().strftime("%Y-%m"),
        "training_rows": int(len(training)),
        "validation_rows": int(len(validation)),
        "testing_rows": int(len(test)),
        "positive_rate": round(float(training["targetIncrease"].mean()), 4),
        "threshold": threshold,
        "accuracy": round(float(accuracy_score(actual, labels)), 4),
        "precision": round(float(precision_score(actual, labels, zero_division=0)), 4),
        "recall": round(float(recall_score(actual, labels, zero_division=0)), 4),
        "f1": round(float(f1_score(actual, labels, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(actual, probabilities)), 4),
        "brier_score": round(float(brier_score_loss(actual, probabilities)), 4),
        "confusion_matrix": confusion_matrix(actual, labels).tolist(),
        "no_increase_baseline_accuracy": round(float(accuracy_score(actual, baseline_labels)), 4),
        "warning": "Evaluation uses synthetic data; validate with real complaints before operational use.",
    }


def main() -> None:
    raw = load_raw_data()
    training = make_training_data(raw)
    training.to_csv(TRAINING_FILE, index=False)
    usable_months = sorted(training["month"].unique())
    if len(usable_months) < 12:
        raise ValueError("At least 12 usable months are required for time-based evaluation.")

    test_start = usable_months[-5]
    validation_start = usable_months[-8]
    train = training[training["month"] < validation_start]
    validation = training[
        (training["month"] >= validation_start) & (training["month"] < test_start)
    ]
    test = training[training["month"] >= test_start]
    if any(frame["targetIncrease"].nunique() < 2 for frame in [train, validation, test]):
        raise ValueError("Training, validation, and testing splits must contain both target classes.")

    threshold_model = build_model()
    threshold_model.fit(train[FEATURE_COLUMNS], train["targetIncrease"])
    threshold = select_threshold(threshold_model, validation)

    evaluation_model = build_model()
    evaluation_model.fit(pd.concat([train, validation])[FEATURE_COLUMNS], pd.concat([train, validation])["targetIncrease"])
    metrics = evaluate(evaluation_model, test, threshold, train, validation)

    inference = make_features(raw)
    latest_month = raw["month"].max()
    current_month = pd.Timestamp.now().to_period("M").to_timestamp()
    next_month = current_month + pd.offsets.MonthBegin(1)
    latest = inference[inference["month"] == latest_month].copy()
    production_model = build_model()
    production_model.fit(training[FEATURE_COLUMNS], training["targetIncrease"])
    latest["probabilityOfIncrease"] = production_model.predict_proba(latest[FEATURE_COLUMNS])[:, 1]
    latest["predictedIncrease"] = latest["probabilityOfIncrease"] >= threshold
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
            "model": production_model,
            "featureColumns": FEATURE_COLUMNS,
            "threshold": threshold,
            "modelVersion": "complaint-trend-logistic-v2",
            "trainedThrough": latest_month.strftime("%Y-%m"),
        },
        MODEL_FILE,
    )
    METRICS_FILE.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved model: {MODEL_FILE}")
    print(f"Saved predictions: {PREDICTIONS_FILE}")


if __name__ == "__main__":
    main()
