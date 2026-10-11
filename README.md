# InsiderWatch AI 🛡️

**An educational project for synthetic insider-threat activity analysis using machine learning.**

InsiderWatch AI combines an unsupervised Isolation Forest model with per-user behavioral baselines to surface activity for human review. The Streamlit dashboard includes interactive charts, filters, synthetic-label evaluation, and CSV export.

> **Important:** This is an educational demo, not a production security product. Its included activity data is synthetic. An anomaly is not proof of malicious behavior; every flag requires human review. Do not upload real employee, customer, or sensitive company logs.

## Features

- Validates required columns, numeric values, hours, non-negative activity counts, and empty or malformed datasets.
- Uses Isolation Forest to flag unusual activity.
- Calculates per-user 95th-percentile thresholds for file access, download volume, failed logins, and login attempts.
- Combines model anomalies with events exceeding at least two behavioral-baseline thresholds, and explains alerts with per-user threshold comparisons.
- Displays activity-by-hour and download-by-user charts, an anomaly scatter plot, filters, and a CSV export.
- Reports precision, recall, F1, accuracy, specificity, and confusion counts against synthetic labels.
- Keeps synthetic labels out of the model features to avoid evaluation-label leakage.
- Includes automated unit tests and a GitHub Actions workflow.

## Technology

Python 3.12, Streamlit, pandas, NumPy, scikit-learn, Plotly, and Python unittest.

## Quick start — Windows PowerShell

Open a terminal in the project folder and run these commands:

    # Create a fresh environment without moving or deleting an existing .venv
    py -3.12 -m venv .venv-insiderwatch

    # Install dependencies
    .\.venv-insiderwatch\Scripts\python.exe -m pip install --upgrade pip
    .\.venv-insiderwatch\Scripts\python.exe -m pip install -r requirements.txt

    # Run automated tests
    .\.venv-insiderwatch\Scripts\python.exe -m unittest discover -s tests -v

    # Start the dashboard
    .\.venv-insiderwatch\Scripts\python.exe -m streamlit run app.py

If Python 3.12 is not installed, run py -0p to list installed versions and substitute an installed supported Python version in the environment creation command.

Streamlit usually opens at **http://localhost:8501**. Keep the terminal open while using the dashboard; press Ctrl+C to stop it.

## Data format

The default dataset is activity_data.csv, stored beside app.py. A compatible CSV can also be uploaded through the dashboard sidebar. Required columns:

| Column | Meaning | Validation |
|---|---|---|
| user_id | Synthetic account identifier | Required, non-empty |
| hour | Hour of activity | 0 through 23 |
| files_accessed | Number of files accessed in an event | Non-negative |
| data_downloaded_mb | Download volume in megabytes | Non-negative |
| failed_logins | Failed login count | Non-negative |
| login_attempts | Login attempt count | Non-negative |
| synthetic_label | Ground truth: 0 normal, 1 injected anomaly | 0 or 1; evaluation only |

Incomplete or non-numeric rows are skipped with a visible warning. Missing required columns, invalid hours, negative activity counts, invalid labels, and empty datasets stop loading with an explanatory message.

## How detection works

1. Validate the input dataset.
2. Fit Isolation Forest using hour, file accesses, download volume, failed logins, and login attempts.
3. Calculate per-user 95th-percentile thresholds for four behavior features.
4. Mark baseline events with at least two threshold exceedances.
5. Add an investigation flag when the model or behavioral baseline flags an event.
6. Compare predictions with synthetic labels only in the evaluation panel.

The default random seed makes model results repeatable. The contamination slider controls the approximate proportion of model outliers; it is not an estimate of the real-world incident rate. Small datasets and synthetic data can produce misleading metrics.

## Project structure

    Insiderwatch-AI/
    ├── app.py
    ├── detection.py
    ├── activity_data.csv
    ├── requirements.txt
    ├── tests/
    │   └── test_detection.py
    └── .github/
        └── workflows/
            └── tests.yml

## Test the project

    .\.venv-insiderwatch\Scripts\python.exe -m unittest discover -s tests -v

## Roadmap

- [x] Dashboard with synthetic activity data
- [x] Isolation Forest anomaly detection
- [x] Per-user behavioral baselines
- [x] Input validation and CSV upload
- [x] Automated tests and continuous integration
- [x] Add explainable alert reasons with per-user threshold comparisons
- [ ] Add time-window-aware baselines
- [ ] Add an event-review workflow
- [ ] Evaluate against representative, documented datasets before making real-world claims

## Responsible use

Use only synthetic or explicitly authorized data. Do not use this demo to make employment, disciplinary, or other high-impact decisions. Anomaly detection produces leads for investigation, not proof of intent.
