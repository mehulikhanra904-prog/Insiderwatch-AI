import numpy as np
import pandas as pd
from pathlib import Path

# Save the dataset in the project folder
OUTPUT_FILE = Path(__file__).parent / "activity_data.csv"

# Reproducible synthetic data
rng = np.random.default_rng(42)

# Generate ordinary activity
number_of_events = 2000

users = [
    "user_001",
    "user_002",
    "user_003",
    "user_004",
    "user_005",
    "user_006",
    "user_007",
    "user_008",
]

data = pd.DataFrame({
    "user_id": rng.choice(users, number_of_events),
    "hour": rng.integers(8, 19, number_of_events),
    "files_accessed": rng.poisson(12, number_of_events),
    "data_downloaded_mb": rng.gamma(
        shape=2, scale=20, size=number_of_events
    ),
    "failed_logins": rng.poisson(0.2, number_of_events),
    "login_attempts": rng.poisson(1, number_of_events) + 1,
})

# Generate unusual synthetic activity
number_of_anomalies = 60

anomalies = pd.DataFrame({
    "user_id": rng.choice(users, number_of_anomalies),
    "hour": rng.choice([0, 1, 2, 3, 4, 23], number_of_anomalies),
    "files_accessed": rng.integers(150, 500, number_of_anomalies),
    "data_downloaded_mb": rng.uniform(1000, 5000, number_of_anomalies),
    "failed_logins": rng.integers(5, 20, number_of_anomalies),
    "login_attempts": rng.integers(10, 30, number_of_anomalies),
})

# Labels are for demonstration and later evaluation
data["synthetic_label"] = 0
anomalies["synthetic_label"] = 1

# Combine and shuffle the records
data = pd.concat([data, anomalies], ignore_index=True)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Save as CSV
data.to_csv(OUTPUT_FILE, index=False)

print("Synthetic activity data generated successfully!")
print(f"Dataset location: {OUTPUT_FILE}")
print(f"Total events: {len(data)}")
print(
    "Ordinary synthetic events:",
    (data["synthetic_label"] == 0).sum()
)
print(
    "Injected unusual events:",
    (data["synthetic_label"] == 1).sum()
)

print("\nFirst five records:")
print(data.head())