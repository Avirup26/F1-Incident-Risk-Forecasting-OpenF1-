"""
Main Streamlit dashboard for F1 Incident Risk Forecasting.

Features:
  - Session selector (year → meeting → session)
  - Risk timeline plot with SC/VSC event overlays
  - Race control message drill-down table
  - Feature importance visualization
  - Model card with no-leakage statement
"""
import sys
from pathlib import Path

# Ensure the project root is on sys.path so both `src` and `app` are importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.config import cfg
from components.risk_plot import render_risk_plot
from components.message_table import render_message_table
from components.feature_importance import render_feature_importance

st.set_page_config(
    page_title="F1 SC/VSC Risk Forecaster",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main { background-color: #0f0f0f; }
    .stMetric { background: #1a1a1a; border-radius: 8px; padding: 12px; }
    .stMetric label { color: #888; }
    h1, h2, h3 { color: #E8002D; }
    .stSidebar { background-color: #111; }
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────────────────
st.title("🏎️ F1 Safety Car Risk Forecaster")
st.caption("Predicts SC/VSC deployment probability every 30 seconds using OpenF1 data.")

# ── Sidebar: Session Selector ────────────────────────────────────────────────
with st.sidebar:
    st.header("📡 Session Selector")

    # Discover available sessions from silver
    @st.cache_data
    def get_available_sessions() -> pd.DataFrame:
        silver_root = cfg.paths.silver
        sessions = []
        if silver_root.exists():
            for year_dir in sorted(silver_root.iterdir()):
                if not year_dir.is_dir():
                    continue
                for session_dir in sorted(year_dir.iterdir()):
                    feat_path = session_dir / "features.parquet"
                    if feat_path.exists():
                        df = pd.read_parquet(feat_path, columns=["session_key", "meeting_key", "meeting_name", "year"])
                        if not df.empty:
                            row = df.iloc[0]
                            sessions.append({
                                "year": int(row["year"]),
                                "meeting_name": row["meeting_name"],
                                "session_key": int(row["session_key"]),
                                "meeting_key": int(row["meeting_key"]),
                            })
        return pd.DataFrame(sessions)

    sessions_df = get_available_sessions()

    if sessions_df.empty:
        st.warning("No processed sessions found. Run `make ingest` then `make features`.")
        st.stop()

    years = sorted(sessions_df["year"].unique(), reverse=True)
    selected_year = st.selectbox("Year", years)

    year_sessions = sessions_df[sessions_df["year"] == selected_year]
    meetings = year_sessions["meeting_name"].unique().tolist()
    selected_meeting = st.selectbox("Race Weekend", meetings)

    session_row = year_sessions[year_sessions["meeting_name"] == selected_meeting].iloc[0]
    session_key = int(session_row["session_key"])

    threshold = st.slider("Alert Threshold", 0.1, 0.9, 0.3, 0.05)

    st.divider()
    st.caption("🔴 Red line = risk score | 🟡 Dashed = SC/VSC event")

# ── Load Session Data ────────────────────────────────────────────────────────
@st.cache_data
def load_session_features(session_key: int) -> pd.DataFrame:
    year = session_row["year"]
    path = cfg.paths.silver / str(year) / str(session_key) / "features.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_race_control(session_key: int) -> pd.DataFrame:
    year = session_row["year"]
    path = cfg.paths.bronze / str(year) / str(session_key) / "race_control.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_resource
def load_lgbm_model():
    from src.models.lgbm_model import LGBMModel
    model_path = cfg.paths.models / "lgbm_model.joblib"
    if model_path.exists():
        return LGBMModel.load(str(model_path))
    return None


features_df = load_session_features(session_key)
race_control_df = load_race_control(session_key)
lgbm_model = load_lgbm_model()

if features_df.empty:
    st.error("No feature data found for this session.")
    st.stop()

# Compute risk scores
if lgbm_model is not None:
    try:
        features_df["risk_score"] = lgbm_model.predict_proba(features_df)
    except Exception as e:
        st.warning(f"Could not compute risk scores: {e}")
        features_df["risk_score"] = 0.0
else:
    st.warning("⚠️ No trained model found. Run `make train` first. Showing dummy scores.")
    features_df["risk_score"] = 0.0

# ── Main Content ─────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Grid Points", f"{len(features_df):,}")
with col2:
    sc_events = features_df["y_sc_5m"].sum()
    st.metric("SC/VSC Windows", f"{sc_events:,}")
with col3:
    max_risk = features_df["risk_score"].max()
    st.metric("Peak Risk Score", f"{max_risk:.1%}")

st.divider()

# Risk Timeline
st.subheader("📈 Risk Timeline")
fig = render_risk_plot(features_df, session_name=selected_meeting, threshold=threshold)
st.plotly_chart(fig, use_container_width=True)

# Time selector for message drill-down
st.subheader("📻 Race Control Messages")
if not features_df.empty:
    ts_min = features_df["timestamp"].min()
    ts_max = features_df["timestamp"].max()
    selected_idx = st.slider(
        "Select time point",
        min_value=0,
        max_value=len(features_df) - 1,
        value=len(features_df) // 2,
    )
    selected_time = features_df.iloc[selected_idx]["timestamp"]
    st.caption(f"Showing messages ±5 min around {pd.Timestamp(selected_time).strftime('%H:%M:%S UTC')}")
    render_message_table(race_control_df, selected_time=pd.Timestamp(selected_time), window_seconds=300)

st.divider()

# Feature Importance
st.subheader("🔍 Feature Importance")
fi_path = cfg.paths.models / "feature_importance.csv"
if fi_path.exists():
    fi_df = pd.read_csv(fi_path)
    render_feature_importance(fi_df, top_n=20)
else:
    st.info("Feature importance not available. Run `make train` first.")

st.divider()

# Model Card
with st.expander("📋 Model Card & No-Leakage Statement"):
    st.markdown("""
    **Model**: LightGBM with TF-IDF + TruncatedSVD text features + numeric race dynamics features.

    **Data**: OpenF1 API — race_control, weather, position, intervals endpoints.

    **Prediction task**: Binary classification — will a Safety Car or VSC be deployed
    in the next 5 minutes? Evaluated every 30 seconds during a race.

    **No-leakage guarantee**:
    - All features use strict as-of semantics (only data ≤ t is used at time t)
    - Train/test splits are by `meeting_key` (race weekend) — no weekend appears in both sets
    - No random splits are used anywhere in the pipeline

    **Limitations**:
    - SC/VSC events are rare (~5–15% of grid points) → imbalanced classification
    - Model trained on historical data; performance may vary on new circuits or regulations
    - Race control message latency may differ in real-time vs. historical data
    """)
