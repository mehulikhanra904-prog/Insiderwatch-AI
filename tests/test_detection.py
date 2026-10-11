import unittest
from io import StringIO

import numpy as np
import pandas as pd

from detection import DataValidationError, evaluate_predictions, load_activity_data, run_detection


def sample_frame(rows=80):
    rng = np.random.default_rng(17)
    frame = pd.DataFrame({
        "user_id": [f"user_{i % 4:03d}" for i in range(rows)],
        "hour": rng.integers(8, 19, rows),
        "files_accessed": rng.integers(1, 15, rows),
        "data_downloaded_mb": rng.uniform(5, 150, rows),
        "failed_logins": rng.integers(0, 3, rows),
        "login_attempts": rng.integers(1, 4, rows),
        "synthetic_label": np.zeros(rows, dtype=int),
    })
    frame.loc[0, ["hour", "files_accessed", "data_downloaded_mb", "failed_logins", "login_attempts"]] = [2, 250, 12000, 8, 20]
    frame.loc[0, "synthetic_label"] = 1
    return frame


class DataLoadingTests(unittest.TestCase):
    def test_loads_valid_data_and_records_dropped_rows(self):
        frame = sample_frame(20)
        frame["hour"] = frame["hour"].astype(object)
        frame.loc[1, "hour"] = "bad-number"
        loaded = load_activity_data(StringIO(frame.to_csv(index=False)))
        self.assertEqual(len(loaded), 19)
        self.assertEqual(loaded.attrs["rows_dropped"], 1)

    def test_missing_columns_are_rejected(self):
        frame = sample_frame(10).drop(columns=["user_id"])
        with self.assertRaises(DataValidationError):
            load_activity_data(StringIO(frame.to_csv(index=False)))

    def test_invalid_hour_is_rejected(self):
        frame = sample_frame(10)
        frame.loc[1, "hour"] = 25
        with self.assertRaises(DataValidationError):
            load_activity_data(StringIO(frame.to_csv(index=False)))

    def test_negative_downloads_are_rejected(self):
        frame = sample_frame(10)
        frame.loc[1, "data_downloaded_mb"] = -1
        with self.assertRaises(DataValidationError):
            load_activity_data(StringIO(frame.to_csv(index=False)))


class DetectionTests(unittest.TestCase):
    def test_outputs_model_baseline_and_investigation_columns(self):
        result = run_detection(sample_frame(), contamination=0.03)
        for column in [
            "model_prediction", "is_anomaly", "anomaly_score",
            "baseline_signal_count", "baseline_flag", "investigation_flag",
        ]:
            self.assertIn(column, result.columns)
        self.assertTrue(np.isfinite(result["anomaly_score"]).all())
        self.assertTrue(result["investigation_flag"].equals(result["is_anomaly"] | result["baseline_flag"]))

    def test_synthetic_label_does_not_change_model_predictions(self):
        first = sample_frame()
        second = first.copy()
        second["synthetic_label"] = 1 - second["synthetic_label"]
        result_a = run_detection(first)
        result_b = run_detection(second)
        np.testing.assert_array_equal(result_a["model_prediction"], result_b["model_prediction"])
        np.testing.assert_allclose(result_a["anomaly_score"], result_b["anomaly_score"])

    def test_rejects_invalid_contamination(self):
        with self.assertRaises(ValueError):
            run_detection(sample_frame(), contamination=0.8)

    def test_evaluation_metrics(self):
        metrics = evaluate_predictions([1, 1, 0, 0], [1, 0, 1, 0])
        self.assertEqual(metrics["true_positives"], 1)
        self.assertEqual(metrics["false_positives"], 1)
        self.assertEqual(metrics["false_negatives"], 1)
        self.assertEqual(metrics["true_negatives"], 1)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 0.5)


if __name__ == "__main__":
    unittest.main()
