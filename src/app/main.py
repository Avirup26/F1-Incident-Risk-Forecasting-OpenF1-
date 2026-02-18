
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from pathlib import Path
from datetime import timedelta

from src.config import cfg
from src.features.feature_pipeline import process_session
from src.app.components.risk_plot import render_risk_plot
from src.app.components.message_table import render_message_table
from src.app.components.feature_importance import render_feature_importance
from src.app.components.track_map import render_track_map  # New component

# Page Config (Must be first)
st.set_page_config(
    page_title="F1 Incident Risk Forecasting",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Attractive" UI
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
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stMetric {
        background-color: #1E1E1E;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #333;
    }
    /* Highlight the simulation slider */
    .stSlider > div > div > div > div {
        background-color: #FF4B4B;
    }
</style>
""", unsafe_allow_html=True)


# --- Data Loading -----------------------------------------------------------

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

@st.cache_data
def load_position_data(year, session_key):
    """Load driver position data for track map."""
    path = cfg.paths.bronze / str(year) / str(session_key) / "position.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)

@st.cache_data
def load_drivers(year, session_key):
    """Load driver mapping (number -> abbr) from results."""
    path = cfg.paths.bronze / str(year) / str(session_key) / "results.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    # Create dict: "44" -> "HAM"
    return dict(zip(df["driver_number"].astype(str), df["driver_abbr"]))


def main():
    st.title("🏎️ F1 Incident Risk Forecasting")
    
    # Load Data & Model
    with st.spinner("Loading aggregated data..."):
        master_df = load_master_timeline()
        model = load_model()

    if master_df.empty:
        st.error("No data found. Please run: `python -m src.cli feature_pipeline`")
        return

    if model is None:
        st.error("No model found. Please run: `python -m src.cli train`")
        return

    # --- Sidebar: Configuration ---------------------------------------------
    st.sidebar.header("Session Selection")
    
    # Year
    years = sorted(master_df["year"].unique())
    selected_year = st.sidebar.selectbox("Year", years, index=len(years)-1)
    
    # Meeting
    meetings = master_df[master_df["year"] == selected_year]["meeting_name"].unique()
    selected_meeting = st.sidebar.selectbox("Meeting", meetings)
    
    # Filter Data for Session
    subset = master_df[(master_df["year"] == selected_year) & (master_df["meeting_name"] == selected_meeting)].copy()
    
    if subset.empty:
        st.warning("No data for this session.")
        return
        
    session_key = subset["session_key"].iloc[0]
    st.sidebar.markdown(f"**Session Key:** `{session_key}`")

    # --- Simulation Mode ----------------------------------------------------
    st.sidebar.divider()
    st.sidebar.header("⏱️ Real-Time Simulation")
    
    use_simulation = st.sidebar.checkbox("Enable Replay Mode", value=False)
    
    min_time = subset["timestamp"].min()
    max_time = subset["timestamp"].max()
    
    # Default: Show full session
    current_time = max_time
    
    if use_simulation:
        # User selects time point
        # Convert to datetime for slider? Streamlit sliders support datetime.
        # But convert to python datetime first.
        start_dt = min_time.to_pydatetime()
        end_dt = max_time.to_pydatetime()
        
        current_time = st.sidebar.slider(
            "Race Time (UTC)",
            min_value=start_dt,
            max_value=end_dt,
            value=start_dt,
            format="HH:mm"
        )
        current_time = pd.Timestamp(current_time).tz_localize("UTC") if pd.Timestamp(current_time).tzinfo is None else pd.Timestamp(current_time)
        
        st.sidebar.info("Drag the slider to replay the race!")
    else:
        st.sidebar.caption("Showing full post-race analysis.")

    # --- Data Filtering -----------------------------------------------------
    
    # Filter Risk Data
    # For plot: show up to current_time (history)
    plot_df = subset[subset["timestamp"] <= current_time].copy()
    
    # Inference for *current* moment (last row of plot_df)
    current_risk_prob = 0.0
    
    # We need to compute predictions for the WHOLE subset first (or at least up to now)
    # efficiently. Since the model is fast, let's predict for plot_df.
    feature_cols = [c for c in subset.columns if c not in [
        "session_key", "meeting_key", "year", "meeting_name", "timestamp", 
        "y_sc_5m", "time_to_sc_seconds"
    ]]
    
    if not plot_df.empty:
        try:
            X = plot_df[feature_cols]
            y_prob = model.predict_proba(X)
            plot_df["prob_lgbm"] = y_prob
            current_risk_prob = y_prob[-1] if len(y_prob) > 0 else 0.0
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return
    else:
        st.warning("Simulation time is before start of data.")
        return


    # --- Dashboard Layout ---------------------------------------------------

    # Top Metrics Row
    m1, m2, m3 = st.columns(3)
    
    with m1:
        st.metric(
            label="Current Incident Risk",
            value=f"{current_risk_prob:.1%}",
            delta=f"{current_risk_prob - (plot_df['prob_lgbm'].iloc[-2] if len(plot_df) > 1 else 0.0):.1%}",
            delta_color="inverse"
        )
    
    with m2:
        # Time to SC? Only if label known (but in inference we don't know)
        # Display current time
        st.metric("Simulation Time", current_time.strftime("%H:%M:%S UTC"))
        
    with m3:
        # Latest Message Category?
        st.metric("Session", f"{selected_meeting}")


    # Main View: Risk Plot + Track Map
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("⚠️ Risk Timeline")
        threshold = 0.5 # Fixed or from slider?
        fig_risk = render_risk_plot(plot_df, threshold=threshold)
        # Update layout to zoom into recent window?
        # Maybe show full history up to now.
        st.plotly_chart(fig_risk, use_container_width=True)
        
    with col_right:
        st.subheader("📍 Track Map")
        if use_simulation:
            # Load heavy bronze data only if needed
            pos_df = load_position_data(selected_year, session_key)
            drivers = load_drivers(selected_year, session_key)
            
            if not pos_df.empty:
                render_track_map(pos_df, drivers, current_time)
            else:
                st.info("No position data available for this session.")
        else:
            st.info("Enable Simulation Mode to see live track map.")


    # Detailed Data Tabs
    st.divider()
    tab1, tab2, tab3 = st.tabs(["Race Control Messages", "Model Insights", "Debug Data"])
    
    with tab1:
        messages_df = load_race_control(selected_year, session_key)
        if not messages_df.empty:
            # Filter messages up to current time
            messages_df = messages_df[messages_df["date"] <= current_time]
            # Show latest first
            messages_df = messages_df.sort_values("date", ascending=False)
            render_message_table(plot_df, messages_df)
    
    with tab2:
        render_feature_importance(model, feature_names=feature_cols)

    with tab3:
        st.dataframe(plot_df.tail(10))

if __name__ == "__main__":
    main()
