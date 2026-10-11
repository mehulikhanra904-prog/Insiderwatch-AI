import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.ensemble import IsolationForest
from pathlib import Path

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="InsiderWatch AI",
    page_icon="🛡️",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
.stApp {
    background-color: #0b1120;
    color: #f1f5f9;
}
[data-testid="stMetric"] {
    background: #151f32;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #263550;
}
h1, h2, h3 {
    color: #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------

st.title("🛡️ InsiderWatch AI")
st.caption("AI-Powered Insider Threat & Anomaly Detection")

DATA_FILE = Path(__file__).parent / "activity_data.csv"

if not DATA_FILE.exists():
    st.error("activity_data.csv was not found in the project directory.")
    st.info("Place activity_data.csv in the same folder as app.py.")
    st.stop()

df = pd.read_csv(DATA_FILE)

required_columns = [
    "user_id",
    "hour",
    "files_accessed",
    "data_downloaded_mb",
    "failed_logins",
    "login_attempts",
    "synthetic_label"
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

df = df.dropna(subset=required_columns).copy()

# ---------------- SIDEBAR ----------------

st.sidebar.header("⚙️ Detection Settings")

contamination = st.sidebar.slider(
    "Expected anomaly proportion",
    min_value=0.01,
    max_value=0.20,
    value=0.03,
    step=0.01
)

show_injected = st.sidebar.checkbox(
    "Show synthetic ground-truth labels",
    value=True
)

# ---------------- TRAIN ANOMALY MODEL ----------------

features = [
    "hour",
    "files_accessed",
    "data_downloaded_mb",
    "failed_logins",
    "login_attempts"
]

model = IsolationForest(
    n_estimators=200,
    contamination=contamination,
    random_state=42
)

df["model_prediction"] = model.fit_predict(df[features])

# Isolation Forest: -1 = anomaly, 1 = normal
df["is_anomaly"] = df["model_prediction"] == -1

df["anomaly_score"] = -model.score_samples(df[features])

# =====================================================
# USER-SPECIFIC BEHAVIORAL BASELINES
# =====================================================

baseline_features = [
    "files_accessed",
    "data_downloaded_mb",
    "failed_logins",
    "login_attempts"
]

# Calculate the 95th percentile for each user.
for feature in baseline_features:
    threshold_column = f"{feature}_user_threshold"

    df[threshold_column] = (
        df.groupby("user_id")[feature]
        .transform(lambda values: values.quantile(0.95))
    )

# Identify activity exceeding individual user thresholds.
df["unusual_user_download"] = (
    df["data_downloaded_mb"]
    > df["data_downloaded_mb_user_threshold"]
)

df["unusual_user_file_access"] = (
    df["files_accessed"]
    > df["files_accessed_user_threshold"]
)

df["unusual_user_failed_logins"] = (
    df["failed_logins"]
    > df["failed_logins_user_threshold"]
)

df["unusual_user_login_attempts"] = (
    df["login_attempts"]
    > df["login_attempts_user_threshold"]
)

# Count behavioral signals for each event.
baseline_signals = [
    "unusual_user_download",
    "unusual_user_file_access",
    "unusual_user_failed_logins",
    "unusual_user_login_attempts"
]

df["baseline_signal_count"] = (
    df[baseline_signals].astype(int).sum(axis=1)
)

# Flag events with at least two unusual signals.
df["baseline_flag"] = df["baseline_signal_count"] >= 2

# Combine model predictions with behavioral-baseline signals.
df["investigation_flag"] = (
    df["is_anomaly"] | df["baseline_flag"]
)

# ---------------- BEHAVIORAL BASELINE VERIFICATION ----------------

with st.expander("🧠 Behavioral Baseline Verification", expanded=False):
    st.caption(
        "Shows which events exceed each user's calculated 95th-percentile "
        "activity thresholds."
    )

    baseline_columns = [
        "user_id",
        "unusual_user_download",
        "unusual_user_file_access",
        "unusual_user_failed_logins",
        "unusual_user_login_attempts",
        "baseline_signal_count",
        "baseline_flag",
        "investigation_flag",
    ]

    st.dataframe(
        df[baseline_columns],
        use_container_width=True,
        hide_index=True,
    )

# ---------------- KPI METRICS ----------------

total_events = len(df)
total_users = df["user_id"].nunique()
detected = int(df["is_anomaly"].sum())
detection_rate = detected / total_events * 100 if total_events else 0

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Events", f"{total_events:,}")
c2.metric("Monitored Users", f"{total_users:,}")
c3.metric("Model-Flagged Events", f"{detected:,}")
c4.metric("Flagged Event Rate", f"{detection_rate:.1f}%")

st.divider()

# ---------------- CHARTS ----------------

left, right = st.columns(2)

with left:
    st.subheader("📊 Activity by Hour")

    hourly = (
        df.groupby("hour")
        .size()
        .reset_index(name="events")
    )

    fig = px.bar(
        hourly,
        x="hour",
        y="events",
        title="Number of Events per Hour",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("📥 Data Downloads by User")

    downloads = (
        df.groupby("user_id")["data_downloaded_mb"]
        .sum()
        .reset_index()
        .sort_values("data_downloaded_mb", ascending=False)
        .head(10)
    )

    fig = px.bar(
        downloads,
        x="user_id",
        y="data_downloaded_mb",
        title="Top 10 Users by Download Volume",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------- ANOMALY VISUALIZATION ----------------

st.subheader("🔍 Normal vs Anomalous Activity")

fig = px.scatter(
    df,
    x="files_accessed",
    y="data_downloaded_mb",
    color="is_anomaly",
    size="failed_logins",
    hover_data=["user_id", "hour", "login_attempts"],
    labels={
        "is_anomaly": "Model Flagged",
        "files_accessed": "Files Accessed",
        "data_downloaded_mb": "Data Downloaded (MB)"
    },
    color_discrete_map={
        False: "#22c55e",
        True: "#ef4444"
    },
    template="plotly_dark",
    title="Activity Anomaly Map"
)

st.plotly_chart(fig, use_container_width=True)

# ---------------- FILTERS ----------------

st.subheader("🚨 Activity Investigation")

users = sorted(df["user_id"].unique().tolist())

selected_users = st.multiselect(
    "Filter by user",
    users,
    default=users
)

event_type = st.selectbox(
    "Filter activity",
    [
        "All events",
        "Model-flagged anomalies",
        "Behavioral baseline flags",
        "All investigation flags",
        "Model-classified normal events",
    ],
)

filtered = df[df["user_id"].isin(selected_users)].copy()

if event_type == "Model-flagged anomalies":
    filtered = filtered[filtered["is_anomaly"]]

elif event_type == "Behavioral baseline flags":
    filtered = filtered[filtered["baseline_flag"]]

elif event_type == "All investigation flags":
    filtered = filtered[filtered["investigation_flag"]]

elif event_type == "Model-classified normal events":
    filtered = filtered[~filtered["is_anomaly"]]

if not show_injected:
    filtered = filtered.drop(columns=["synthetic_label"])

st.dataframe(
    filtered.sort_values("anomaly_score", ascending=False),
    use_container_width=True,
    hide_index=True
)

# ---------------- MODEL EVALUATION ----------------

if show_injected:
    st.divider()
    st.subheader("🧪 Synthetic Dataset Evaluation")

    st.caption(
        "The injected labels are used only for evaluation, "
        "not as model input features."
    )

    actual = df["synthetic_label"].astype(int)
    predicted = df["is_anomaly"].astype(int)

    tp = int(((actual == 1) & (predicted == 1)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())
    tn = int(((actual == 0) & (predicted == 0)).sum())

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    a, b, c = st.columns(3)

    a.metric("Precision", f"{precision:.1%}")
    b.metric("Recall", f"{recall:.1%}")
    c.metric("F1 Score", f"{f1:.1%}")

    st.write({
        "True Positives": tp,
        "False Positives": fp,
        "False Negatives": fn,
        "True Negatives": tn
    })

# ---------------- EXPORT ----------------

st.divider()

st.subheader("📄 Export Investigation Results")

csv_data = filtered.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Filtered Events CSV",
    data=csv_data,
    file_name="insiderwatch_investigation.csv",
    mime="text/csv"
)

st.caption(
    "InsiderWatch AI | Synthetic security activity analysis | "
    "Model predictions require human review."
)