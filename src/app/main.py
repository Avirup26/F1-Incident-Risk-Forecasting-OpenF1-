
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from pathlib import Path

from src.config import cfg
from src.features.feature_pipeline import process_session
from src.app.components.risk_plot import render_risk_plot
from src.app.components.message_table import render_message_table
from src.app.components.feature_importance import render_feature_importance

# Page Config
st.set_page_config(
    page_title="F1 Risk Forecasting",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .main {
        padding: 2rem;
    }
    h1 {
        color: #FF4B4B;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_master_timeline():
    path = cfg.paths.data / "gold" / "master_timeline.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)

@st.cache_resource
def load_model():
    path = cfg.paths.models / "lgbm_model.joblib"
    if not path.exists():
        return None
    return joblib.load(path)

@st.cache_data
def load_race_control(year, session_key):
    path = cfg.paths.bronze / str(year) / str(session_key) / "race_control.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def main():
    st.title("🏎️ F1 Incident Risk Forecasting")
    st.markdown("Predicting Safety Car (SC) and Virtual Safety Car (VSC) probability in real-time.")

    # Load Data & Model
    with st.spinner("Loading data..."):
        master_df = load_master_timeline()
        model = load_model()

    if master_df.empty:
        st.error("No data found in data/gold/master_timeline.parquet. Please run feature pipeline first.")
        return

    if model is None:
        st.error("No model found. Please run training first.")
        return

    # Sidebar: Selection
    st.sidebar.header("Session Selector")
    
    # Year
    years = sorted(master_df["year"].unique())
    selected_year = st.sidebar.selectbox("Year", years, index=len(years)-1)
    
    # Meeting
    meetings = master_df[master_df["year"] == selected_year]["meeting_name"].unique()
    selected_meeting = st.sidebar.selectbox("Meeting", meetings)
    
    # Session Key (hidden or derived)
    # We assume one race session per meeting for now, or filter by meeting key
    subset = master_df[(master_df["year"] == selected_year) & (master_df["meeting_name"] == selected_meeting)]
    
    if subset.empty:
        st.warning("No data for this session.")
        return
        
    session_key = subset["session_key"].iloc[0]
    st.sidebar.write(f"**Session Key:** {session_key}")

    # Inference
    # We need predictions for this session.
    # The gold data has features. We need to run the model on them.
    # Identify feature columns
    feature_cols = [c for c in subset.columns if c not in [
        "session_key", "meeting_key", "year", "meeting_name", "timestamp", 
        "y_sc_5m", "time_to_sc_seconds"
    ]]
    
    X = subset[feature_cols]
    
    # Predict
    try:
        # Check for missing columns expected by model?
        # LightGBM handles missing cols if configured, but let's assume features match.
        y_prob = model.predict_proba(X)[:, 1]
        
        # Add to dataframe for plotting
        plot_df = subset.copy()
        plot_df["prob_lgbm"] = y_prob
        
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return

    # --- Main Dashboard ---
    
    # 1. Risk Plot
    st.subheader("⚠️ Risk Timeline")
    threshold = st.sidebar.slider("Risk Threshold", 0.0, 1.0, 0.5, 0.05)
    
    fig = render_risk_plot(plot_df, threshold=threshold)
    
    # Interactive filtering based on click? 
    # For now, just use hover.
    st.plotly_chart(fig, use_container_width=True)
    
    # 2. Tabs for Details
    tab1, tab2, tab3 = st.tabs(["Race Control Messages", "Model Insights", "Debug Data"])
    
    with tab1:
        st.write("### Race Control Messages")
        messages_df = load_race_control(selected_year, session_key)
        
        # Add a time slider to filter messages?
        # Or just show them all.
        # Let's add a filter for "Show only near high risk"?
        render_message_table(plot_df, messages_df)

    with tab2:
        st.write("### Feature Importance")
        # Global importance
        render_feature_importance(model, feature_names=feature_cols)
        
        # Local importance (SHAP) could be added here later

    with tab3:
        st.write("### Raw Data")
        st.dataframe(plot_df.head(100))

if __name__ == "__main__":
    main()
