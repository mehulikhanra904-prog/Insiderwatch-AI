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
        "Model-classified normal events"
    ]
)

filtered = df[df["user_id"].isin(selected_users)].copy()

if event_type == "Model-flagged anomalies":
    filtered = filtered[filtered["is_anomaly"]]

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

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="InsiderWatch AI | Security Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# PROFESSIONAL DARK UI
# =====================================================

st.markdown("""
<style>
.stApp {
    background-color: #0b1120;
    color: #e2e8f0;
}

[data-testid="stSidebar"] {
    background-color: #101a2d;
}

[data-testid="stMetric"] {
    background-color: #151f32;
    border: 1px solid #263550;
    padding: 18px;
    border-radius: 12px;
}

[data-testid="stMetricValue"] {
    color: #38bdf8;
}

h1, h2, h3 {
    color: #f1f5f9;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #263550;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# HEADER
# =====================================================

st.title("🛡️ InsiderWatch AI")

st.markdown(
    "### AI-Powered Insider Threat Detection & Security Intelligence"
)

st.caption(
    "Behavioral analytics • Anomaly detection • Risk assessment"
)

st.divider()


# =====================================================
# LOAD DATASET
# =====================================================

DATA_FILE = Path(__file__).parent / "activity_data.csv"

if not DATA_FILE.exists():
    st.error("Dataset not found: activity_data.csv")
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

if df.empty:
    st.error("The dataset contains no records.")
    st.stop()

if df[required_columns].isna().any().any():
    st.error(
        "Some required fields contain missing values. "
        "Please clean the dataset before running detection."
    )
    st.stop()

if not set(df["synthetic_label"].unique()).issubset({0, 1}):
    st.error("synthetic_label must contain only 0 and 1.")
    st.stop()

df = df.copy()

numeric_columns = [
    "hour",
    "files_accessed",
    "data_downloaded_mb",
    "failed_logins",
    "login_attempts"
]

if not np.isfinite(
    df[numeric_columns].to_numpy(dtype=float)
).all():
    st.error("The dataset contains infinite or invalid numeric values.")
    st.stop()

if (df["data_downloaded_mb"] < 0).any():
    st.error("Download volumes cannot be negative.")
    st.stop()


# =====================================================
# SIDEBAR SETTINGS
# =====================================================

st.sidebar.title("⚙️ Detection Center")

contamination = st.sidebar.slider(
    "Expected anomaly proportion",
    min_value=0.01,
    max_value=0.20,
    value=0.03,
    step=0.01,
    help="Controls the expected proportion of anomalies flagged by Isolation Forest."
)

high_risk_threshold = st.sidebar.slider(
    "High-risk threshold",
    min_value=50,
    max_value=95,
    value=75,
    help="Risk scores at or above this value are marked high risk."
)

st.sidebar.caption(
    "Thresholds control investigation priorities, "
    "not proof of malicious behavior."
)


# =====================================================
# ANOMALY DETECTION
# =====================================================

FEATURES = [
    "hour",
    "files_accessed",
    "data_downloaded_mb",
    "failed_logins",
    "login_attempts"
]

# Fit on observed activity only.
# synthetic_label is intentionally excluded.

model = IsolationForest(
    n_estimators=200,
    contamination=contamination,
    random_state=42
)

model.fit(df[FEATURES])

df["model_prediction"] = model.predict(df[FEATURES])

df["is_anomaly"] = df["model_prediction"] == -1

# Larger scores indicate more unusual model behavior.
df["anomaly_score"] = -model.score_samples(df[FEATURES])


# =====================================================
# EXPLAINABLE AI
# =====================================================

# Use robust, dataset-relative reference thresholds.
# These rules are illustrative explanations, not causal proof.

download_threshold = df["data_downloaded_mb"].quantile(0.95)

files_threshold = df["files_accessed"].quantile(0.95)

failed_login_threshold = max(
    2,
    int(np.ceil(df["failed_logins"].quantile(0.95)))
)

login_threshold = max(
    4,
    int(np.ceil(df["login_attempts"].quantile(0.95)))
)

df["explanation"] = ""

for idx, row in df.iterrows():

    reasons = []

    if row["data_downloaded_mb"] > download_threshold:
        reasons.append("Unusually large data download")

    if row["files_accessed"] > files_threshold:
        reasons.append("High file-access volume")

    if row["failed_logins"] >= failed_login_threshold:
        reasons.append("Repeated failed login attempts")

    if row["login_attempts"] >= login_threshold:
        reasons.append("High login-attempt count")

    if not 7 <= row["hour"] <= 19:
        reasons.append("Activity outside the reference hours")

    if row["is_anomaly"] and not reasons:
        reasons.append(
            "Unusual combination of observed activity features"
        )

    if reasons:
        df.at[idx, "explanation"] = "; ".join(reasons)
    else:
        df.at[idx, "explanation"] = (
            "No configured behavioral rule was triggered"
        )


# =====================================================
# RISK SCORING
# =====================================================

# Convert anomaly scores into percentile-based scores.
# This creates a relative score within the current dataset.

df["model_risk"] = (
    df["anomaly_score"]
    .rank(method="average", pct=True) * 100
)

# Additional transparent behavioral signals.

df["download_risk"] = (
    df["data_downloaded_mb"] >= download_threshold
).astype(int)

df["login_risk"] = (
    df["failed_logins"] >= failed_login_threshold
).astype(int)

df["access_risk"] = (
    df["files_accessed"] >= files_threshold
).astype(int)

df["Risk Score"] = (
    0.70 * df["model_risk"]
    + 10 * df["download_risk"]
    + 10 * df["login_risk"]
    + 10 * df["access_risk"]
).clip(0, 100).round(1)

df["Risk Level"] = pd.cut(
    df["Risk Score"],
    bins=[-0.01, 34, 74, 100],
    labels=["Low", "Medium", "High"]
).astype(str)

df["Risk Level"] = df["Risk Level"].astype("object")

df.loc[
    df["Risk Score"] >= high_risk_threshold,
    "Risk Level"
] = "High"


# =====================================================
# USER RISK PROFILING
# =====================================================

user_profiles = (
    df.groupby("user_id")
    .agg(
        Total_Events=("user_id", "size"),
        Flagged_Events=("is_anomaly", "sum"),
        Average_Risk=("Risk Score", "mean"),
        Maximum_Risk=("Risk Score", "max"),
        Total_Download_MB=("data_downloaded_mb", "sum"),
        Failed_Logins=("failed_logins", "sum")
    )
    .reset_index()
)

user_profiles["Average_Risk"] = (
    user_profiles["Average_Risk"].round(1)
)

user_profiles["Maximum_Risk"] = (
    user_profiles["Maximum_Risk"].round(1)
)

user_profiles["Flagged_Percentage"] = (
    user_profiles["Flagged_Events"]
    / user_profiles["Total_Events"]
    * 100
).round(1)

user_profiles["Profile"] = np.select(
    [
        user_profiles["Average_Risk"] >= 75,
        user_profiles["Average_Risk"] >= 40
    ],
    [
        "High Attention",
        "Needs Review"
    ],
    default="Routine Monitoring"
)

user_profiles = user_profiles.sort_values(
    "Average_Risk",
    ascending=False
)


# =====================================================
# OVERVIEW METRICS
# =====================================================

total_events = len(df)

total_users = df["user_id"].nunique()

anomaly_count = int(df["is_anomaly"].sum())

high_risk_count = int(
    (df["Risk Level"] == "High").sum()
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Total Events",
    f"{total_events:,}"
)

c2.metric(
    "Monitored Users",
    f"{total_users:,}"
)

c3.metric(
    "Model-Flagged Events",
    f"{anomaly_count:,}"
)

c4.metric(
    "High-Risk Events",
    f"{high_risk_count:,}"
)

st.divider()


# =====================================================
# SECURITY ANALYTICS
# =====================================================

st.subheader("📊 Security Analytics")

left, right = st.columns(2)

with left:

    hourly = (
        df.groupby("hour")
        .size()
        .reset_index(name="Events")
    )

    fig = px.bar(
        hourly,
        x="hour",
        y="Events",
        title="Activity Distribution by Hour",
        template="plotly_dark"
    )

    fig.update_layout(
        xaxis_title="Hour of Day",
        yaxis_title="Number of Events"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with right:

    risk_distribution = (
        df["Risk Level"]
        .value_counts()
        .reindex(["Low", "Medium", "High"], fill_value=0)
        .rename_axis("Risk Level")
        .reset_index(name="Events")
    )

    fig = px.bar(
        risk_distribution,
        x="Risk Level",
        y="Events",
        color="Risk Level",
        color_discrete_map={
            "Low": "#22c55e",
            "Medium": "#f59e0b",
            "High": "#ef4444"
        },
        title="Risk Classification",
        template="plotly_dark"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =====================================================
# USER RISK LEADERBOARD
# =====================================================

st.divider()

st.subheader("👤 User Risk Intelligence")

st.caption(
    "Ranked by average event risk. "
    "High scores indicate activity requiring closer review."
)

fig = px.bar(
    user_profiles.head(10),
    x="user_id",
    y="Average_Risk",
    color="Average_Risk",
    color_continuous_scale="RdYlGn_r",
    title="Users with Highest Average Risk Scores",
    hover_data=[
        "Total_Events",
        "Flagged_Events",
        "Failed_Logins",
        "Profile"
    ],
    template="plotly_dark"
)

fig.update_layout(
    xaxis_title="User",
    yaxis_title="Average Risk Score",
    yaxis_range=[0, 100]
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.dataframe(
    user_profiles,
    use_container_width=True,
    hide_index=True
)


# =====================================================
# EXPLAINABLE AI INVESTIGATION
# =====================================================

st.divider()

st.subheader("🧠 Explainable AI Investigation")

st.write(
    "Inspect model-flagged events, their relative risk scores, "
    "and the behavioral signals that may warrant investigation."
)

filter_mode = st.selectbox(
    "Choose events to inspect",
    [
        "All events",
        "Model-flagged anomalies",
        "High-risk events",
        "Repeated failed logins",
        "Large data downloads"
    ]
)

filtered = df.copy()

if filter_mode == "Model-flagged anomalies":
    filtered = filtered[filtered["is_anomaly"]]

elif filter_mode == "High-risk events":
    filtered = filtered[
        filtered["Risk Level"] == "High"
    ]

elif filter_mode == "Repeated failed logins":
    filtered = filtered[
        filtered["failed_logins"] >= failed_login_threshold
    ]

elif filter_mode == "Large data downloads":
    filtered = filtered[
        filtered["data_downloaded_mb"] > download_threshold
    ]


selected_users = st.multiselect(
    "Filter by user",
    options=sorted(df["user_id"].unique()),
    default=sorted(df["user_id"].unique())
)

filtered = filtered[
    filtered["user_id"].isin(selected_users)
]

display_columns = [
    "user_id",
    "hour",
    "files_accessed",
    "data_downloaded_mb",
    "failed_logins",
    "login_attempts",
    "is_anomaly",
    "Risk Score",
    "Risk Level",
    "explanation"
]

st.dataframe(
    filtered[display_columns].sort_values(
        "Risk Score",
        ascending=False
    ),
    use_container_width=True,
    hide_index=True
)

if filtered.empty:
    st.info("No events match the selected filters.")


# =====================================================
# EVENT INVESTIGATION
# =====================================================

st.divider()

st.subheader("🔎 Investigate an Individual Event")

if not filtered.empty:

    event_index = st.selectbox(
        "Select an event",
        options=filtered.index.tolist(),
        format_func=lambda i: (
            f"Event {i} | {filtered.loc[i, 'user_id']} | "
            f"Risk: {filtered.loc[i, 'Risk Score']:.1f}"
        )
    )

    event = filtered.loc[event_index]

    a, b, c = st.columns(3)

    a.metric(
        "Risk Score",
        f"{event['Risk Score']:.1f}/100"
    )

    b.metric(
        "Risk Classification",
        event["Risk Level"]
    )

    c.metric(
        "Model Flagged",
        "Yes" if event["is_anomaly"] else "No"
    )

    st.markdown("**Potential reasons for review**")

    st.info(event["explanation"])

    st.write("Observed activity details")

    st.json({
        "User": str(event["user_id"]),
        "Hour": int(event["hour"]),
        "Files Accessed": int(event["files_accessed"]),
        "Data Downloaded MB": round(
            float(event["data_downloaded_mb"]), 2
        ),
        "Failed Logins": int(event["failed_logins"]),
        "Login Attempts": int(event["login_attempts"])
    })


# =====================================================
# SYNTHETIC DATASET EVALUATION
# =====================================================

st.divider()

st.subheader("🧪 Synthetic Dataset Evaluation")

st.caption(
    "Synthetic labels are used only for evaluation, "
    "never as model input features."
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


# =====================================================
# EXPORT SECURITY REPORTS
# =====================================================

st.divider()

st.subheader("📄 Export Security Reports")

report_columns = [
    "user_id",
    "hour",
    "files_accessed",
    "data_downloaded_mb",
    "failed_logins",
    "login_attempts",
    "is_anomaly",
    "anomaly_score",
    "Risk Score",
    "Risk Level",
    "explanation"
]

report = filtered[report_columns].copy()

st.download_button(
    label="⬇️ Download Filtered Investigation CSV",
    data=report.to_csv(index=False).encode("utf-8"),
    file_name="insiderwatch_investigation.csv",
    mime="text/csv"
)

high_risk_report = df[
    df["Risk Level"] == "High"
][report_columns]

st.download_button(
    label="⬇️ Download High-Risk Events CSV",
    data=high_risk_report.to_csv(index=False).encode("utf-8"),
    file_name="insiderwatch_high_risk_events.csv",
    mime="text/csv"
)


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "InsiderWatch AI | AI/ML Cybersecurity Research Project | "
    "Synthetic data only | Human investigation required"
)