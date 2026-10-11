"""Validated anomaly detection helpers for the InsiderWatch AI learning project."""
from __future__ import annotations

from pathlib import Path
from typing import IO

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

REQUIRED_COLUMNS = [
    "user_id", "hour", "files_accessed", "data_downloaded_mb",
    "failed_logins", "login_attempts", "synthetic_label",
]
FEATURES = [
    "hour", "files_accessed", "data_downloaded_mb",
    "failed_logins", "login_attempts",
]
BASELINE_FEATURES = [
    "files_accessed", "data_downloaded_mb", "failed_logins", "login_attempts",
]
NUMERIC_COLUMNS = FEATURES + ["synthetic_label"]
NONNEGATIVE_COLUMNS = [
    "files_accessed", "data_downloaded_mb", "failed_logins", "login_attempts",
]


class DataValidationError(ValueError):
    """Raised when activity data cannot be used safely by the detector."""


def load_activity_data(source: str | Path | IO[str]) -> pd.DataFrame:
    """Read and validate CSV data; drop incomplete rows and report their count."""
    try:
        frame = pd.read_csv(source)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DataValidationError(f"Could not read the CSV: {exc}") from exc

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")
    if frame.empty:
        raise DataValidationError("The CSV contains no activity rows.")

    frame = frame.copy()
    frame["user_id"] = frame["user_id"].astype("string").str.strip()
    frame["user_id"] = frame["user_id"].replace("", pd.NA)
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    valid = frame[REQUIRED_COLUMNS].notna().all(axis=1)
    finite = np.isfinite(frame[NUMERIC_COLUMNS].to_numpy(dtype=float)).all(axis=1)
    valid &= pd.Series(finite, index=frame.index)
    dropped_rows = int((~valid).sum())
    frame = frame.loc[valid].copy()
    if frame.empty:
        raise DataValidationError(
            "No valid rows remain after removing missing or non-numeric values."
        )

    if not frame["hour"].between(0, 23).all():
        raise DataValidationError("Values in 'hour' must be between 0 and 23.")
    if not frame["synthetic_label"].isin([0, 1]).all():
        raise DataValidationError(
            "'synthetic_label' must contain only 0 (normal) or 1 (synthetic anomaly)."
        )
    for column in NONNEGATIVE_COLUMNS:
        if (frame[column] < 0).any():
            raise DataValidationError(f"Values in '{column}' cannot be negative.")

    frame["synthetic_label"] = frame["synthetic_label"].astype(int)
    frame = frame.reset_index(drop=True)
    frame.attrs["rows_dropped"] = dropped_rows
    return frame


def run_detection(
    frame: pd.DataFrame,
    contamination: float = 0.03,
    random_state: int = 42,
    n_estimators: int = 200,
) -> pd.DataFrame:
    """Run Isolation Forest and per-user 95th-percentile behavioral baselines."""
    if not 0.01 <= contamination <= 0.20:
        raise ValueError("contamination must be between 0.01 and 0.20.")
    if n_estimators < 1:
        raise ValueError("n_estimators must be positive.")
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")
    if len(frame) < 2:
        raise DataValidationError("At least two valid events are required.")

    result = frame.copy()
    # Synthetic labels are excluded to prevent evaluation-label leakage.
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    )
    result["model_prediction"] = model.fit_predict(result[FEATURES].astype(float))
    result["is_anomaly"] = result["model_prediction"] == -1
    result["anomaly_score"] = -model.score_samples(result[FEATURES].astype(float))

    # Keep stable, human-readable column names used by the dashboard.
    signal_names = {
        "files_accessed": "unusual_user_file_access",
        "data_downloaded_mb": "unusual_user_download",
        "failed_logins": "unusual_user_failed_logins",
        "login_attempts": "unusual_user_login_attempts",
    }
    signal_columns = []
    for feature in BASELINE_FEATURES:
        threshold = f"{feature}_user_threshold"
        signal = signal_names[feature]
        result[threshold] = result.groupby("user_id")[feature].transform(
            lambda values: values.quantile(0.95)
        )
        result[signal] = result[feature] > result[threshold]
        signal_columns.append(signal)

    result["baseline_signal_count"] = result[signal_columns].astype(int).sum(axis=1)
    result["baseline_flag"] = result["baseline_signal_count"] >= 2
    result["investigation_flag"] = result["is_anomaly"] | result["baseline_flag"]

    # Explain alert rows so investigators can understand why they were surfaced.
    def explain_event(row: pd.Series) -> str:
        reasons: list[str] = []
        if bool(row["is_anomaly"]):
            reasons.append(
                "Isolation Forest identified an unusual combination of activity features."
            )

        details = [
            ("unusual_user_download", "Data download", "data_downloaded_mb",
             "data_downloaded_mb_user_threshold", "MB", 1),
            ("unusual_user_file_access", "File access", "files_accessed",
             "files_accessed_user_threshold", "files", 0),
            ("unusual_user_failed_logins", "Failed logins", "failed_logins",
             "failed_logins_user_threshold", "attempts", 0),
            ("unusual_user_login_attempts", "Login attempts", "login_attempts",
             "login_attempts_user_threshold", "attempts", 0),
        ]
        for signal, label, feature, threshold, unit, decimals in details:
            if bool(row[signal]):
                value = float(row[feature])
                limit = float(row[threshold])
                reasons.append(
                    f"{label}: {value:.{decimals}f} {unit} exceeded this user's "
                    f"95th-percentile baseline ({limit:.{decimals}f} {unit})."
                )

        if bool(row["baseline_flag"]):
            reasons.append(
                f"{int(row['baseline_signal_count'])} behavioral signals exceeded "
                "user-specific thresholds."
            )
        if not reasons:
            return "No model or behavioral-baseline alert for this event."
        return " ".join(reasons)

    result["alert_reason"] = result.apply(explain_event, axis=1)
    return result


def evaluate_predictions(actual: object, predicted: object) -> dict[str, float | int]:
    """Calculate binary classification metrics safely."""
    y_true = np.asarray(actual, dtype=int).reshape(-1)
    y_pred = np.asarray(predicted, dtype=int).reshape(-1)
    if len(y_true) != len(y_pred):
        raise ValueError("actual and predicted must have the same number of items.")
    if len(y_true) == 0:
        raise ValueError("Cannot evaluate empty predictions.")
    if not np.isin(y_true, [0, 1]).all() or not np.isin(y_pred, [0, 1]).all():
        raise ValueError("actual and predicted must contain only binary values 0 or 1.")

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": tp, "false_positives": fp,
        "false_negatives": fn, "true_negatives": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "accuracy": (tp + tn) / len(y_true),
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
    }
