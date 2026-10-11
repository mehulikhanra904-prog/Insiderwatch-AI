import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from pathlib import Path
from detection import DataValidationError, evaluate_predictions, load_activity_data, run_detection

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

uploaded_file = st.sidebar.file_uploader(
    "Try your own activity CSV (optional)",
    type=["csv"],
    help="The CSV must contain the seven required columns listed in the README."
)

if uploaded_file is not None:
    data_source = uploaded_file
    source_name = uploaded_file.name
else:
    data_source = DATA_FILE
    source_name = DATA_FILE.name

if uploaded_file is None and not DATA_FILE.exists():
    st.error("activity_data.csv was not found in the project directory.")
    st.info("Add activity_data.csv beside app.py or upload a compatible CSV in the sidebar.")
    st.stop()

try:
    df = load_activity_data(data_source)
except DataValidationError as exc:
    st.error(f"Unable to load activity data: {exc}")
    st.stop()

dropped_rows = int(df.attrs.get("rows_dropped", 0))
if dropped_rows:
    st.warning(f"Skipped {dropped_rows} incomplete or non-numeric row(s) while loading the dataset.")

st.caption(f"Data source: {source_name} · {len(df):,} valid event(s) · {df['user_id'].nunique():,} user(s)")

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

try:
    df = run_detection(df, contamination=contamination)
except (DataValidationError, ValueError) as exc:
    st.error(f"Unable to run anomaly detection: {exc}")
    st.stop()

# ---------------- BEHAVIORAL BASELINE VERIFICATION ----------------

with st.expander("🧠 Behavioral Baseline Verification", expanded=False):
    st.caption(
        "Shows which events exceed each user's calculated 95th-percentile "
        "activity thresholds."
    )

    baseline_columns = [
        "user_id",
        "alert_reason",
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
baseline_detected = int(df["baseline_flag"].sum())
investigation_count = int(df["investigation_flag"].sum())
detection_rate = detected / total_events * 100 if total_events else 0

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Total Events", f"{total_events:,}")
c2.metric("Monitored Users", f"{total_users:,}")
c3.metric("Model Anomalies", f"{detected:,}")
c4.metric("Baseline Flags", f"{baseline_detected:,}")
c5.metric("Investigation Queue", f"{investigation_count:,}")

st.caption(
    f"Isolation Forest flags {detection_rate:.1f}% of events at the selected "
    "contamination setting. This is a model setting, not a measured real-world "
    "incident rate."
)

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
    metrics = evaluate_predictions(actual, predicted)

    a, b, c, d = st.columns(4)
    a.metric("Precision", f"{metrics['precision']:.1%}")
    b.metric("Recall", f"{metrics['recall']:.1%}")
    c.metric("F1 Score", f"{metrics['f1']:.1%}")
    d.metric("Accuracy", f"{metrics['accuracy']:.1%}")

    st.write({
        "True Positives": metrics["true_positives"],
        "False Positives": metrics["false_positives"],
        "False Negatives": metrics["false_negatives"],
        "True Negatives": metrics["true_negatives"],
        "Specificity": f"{metrics['specificity']:.1%}",
    })

    st.info(
        "These scores evaluate this synthetic dataset only. They do not "
        "estimate performance on live enterprise activity."
    )

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