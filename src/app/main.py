
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
from src.app.components.track_map import render_track_map

# Page Config (Must be first)
st.set_page_config(
    page_title="F1 Incident Risk Forecasting",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- F1 Style CSS (High Legibility) -----------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Titillium+Web:wght@400;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Titillium Web', sans-serif;
        font-size: 18px; /* Base font size increase */
        color: #E0E0E0;
    }

    .reportview-container {
        background: #15151E;
    }
    
    .main {
        background-color: #15151E;
    }

    /* Headings */
    h1, h2, h3 {
        color: white;
        text-transform: uppercase;
        font-style: italic;
        font-weight: 700;
        letter-spacing: 2px;
    }
    
    h1 {
        font-size: 3.5rem; /* Larger Title */
        border-bottom: 5px solid #FF1801;
        padding-bottom: 15px;
        margin-bottom: 30px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    h2 {
        font-size: 2rem;
        margin-top: 30px;
        border-left: 5px solid #FF1801;
        padding-left: 15px;
    }
    
    h3 {
        font-size: 1.5rem;
        color: #CCCCCC;
    }

    /* Metrics Cards */
    div[data-testid="stMetric"] {
        background-color: #1F1F27;
        border-right: 5px solid #FF1801;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.4);
        transition: transform 0.2s;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: scale(1.02);
        border-right: 5px solid #FFFFFF;
        background-color: #252530;
    }

    div[data-testid="stMetricLabel"] {
        color: #EEEEEE !important; /* Brighter label */
        font-size: 1.2rem !important;
        font-weight: 600;
        text-transform: uppercase;
    }

    div[data-testid="stMetricValue"] {
        color: white !important;
        font-size: 3rem !important; /* Bigger Risk % */
        font-weight: 700;
        text-shadow: 0 0 10px rgba(255, 24, 1, 0.3);
    }
    
    div[data-testid="stMetricDelta"] {
        font-size: 1.1rem !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1A1A22;
        border-right: 2px solid #333;
    }
    
    section[data-testid="stSidebar"] .stMarkdown h2 {
        font-size: 1.8rem;
        border-left: none;
        padding-left: 0;
        color: #FF1801;
    }
    
    .stSelectbox label {
        font-size: 1.2rem;
        color: white;
    }

    /* Buttons */
    .stButton > button {
        background-color: #FF1801;
        color: white;
        border: none;
        border-radius: 4px;
        font-size: 1.2rem;
        font-weight: 700;
        text-transform: uppercase;
        padding: 10px 24px;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #FF4040;
        box-shadow: 0 0 15px rgba(255, 24, 1, 0.6);
    }

    /* Slider */
    .stSlider > div > div > div > div {
        background-color: #FF1801;
    }
    
    /* Checkbox */
    .stCheckbox label {
        font-size: 1.2rem;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1F1F27;
        border-radius: 4px 4px 0 0;
        color: #CCCCCC;
        font-size: 1.2rem; /* Larger tabs */
        padding: 12px 30px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF1801;
        color: white;
        font-weight: bold;
    }
    
    /* Table */
    .stDataFrame {
        font-size: 1.1rem;
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
    return dict(zip(df["driver_number"].astype(str), df["driver_abbr"]))


def main():
    st.title("F1 INCIDENT RISK FORECASTING")
    
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
    st.sidebar.header("Real-Time Simulation")
    
    use_simulation = st.sidebar.checkbox("Enable Replay Mode", value=False)
    
    min_time = subset["timestamp"].min()
    max_time = subset["timestamp"].max()
    
    current_time = max_time
    
    if use_simulation:
        start_dt = min_time.to_pydatetime()
        end_dt = max_time.to_pydatetime()
        
        current_time = st.sidebar.slider(
            "RACE TIME (UTC)",
            min_value=start_dt,
            max_value=end_dt,
            value=start_dt,
            format="HH:mm"
        )
        current_time = pd.Timestamp(current_time).tz_localize("UTC") if pd.Timestamp(current_time).tzinfo is None else pd.Timestamp(current_time)
        
        st.sidebar.info("Drag the slider to replay the race!")
    else:
        st.sidebar.caption("Show Full Post-Race Analysis")

    # --- Data Filtering -----------------------------------------------------
    
    plot_df = subset[subset["timestamp"] <= current_time].copy()
    current_risk_prob = 0.0
    
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
            label="INCIDENT RISK",
            value=f"{current_risk_prob:.1%}",
            delta=f"{current_risk_prob - (plot_df['prob_lgbm'].iloc[-2] if len(plot_df) > 1 else 0.0):.1%}",
            delta_color="inverse"
        )
    
    with m2:
        st.metric("SIMULATION TIME", current_time.strftime("%H:%M:%S UTC"))
        
    with m3:
        st.metric("SESSION", f"{selected_meeting}")


    # Main View
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        # Title handled by plot layout
        threshold = 0.5 
        fig_risk = render_risk_plot(plot_df, threshold=threshold)
        st.plotly_chart(fig_risk, use_container_width=True)
        
    with col_right:
        st.subheader("TRACK MAP")
        if use_simulation:
            pos_df = load_position_data(selected_year, session_key)
            drivers = load_drivers(selected_year, session_key)
            
            if not pos_df.empty:
                render_track_map(pos_df, drivers, current_time)
            else:
                st.info("No position data available.")
        else:
            st.info("Enable Replay Mode in sidebar to view live track map.")


    # Detailed Data Tabs
    st.divider()
    tab1, tab2, tab3 = st.tabs(["RACE CONTROL", "MODEL INSIGHTS", "DEBUG DATA"])
    
    with tab1:
        messages_df = load_race_control(selected_year, session_key)
        if not messages_df.empty:
            messages_df = messages_df[messages_df["date"] <= current_time]
            messages_df = messages_df.sort_values("date", ascending=False)
            render_message_table(plot_df, messages_df)
    
    with tab2:
        render_feature_importance(model, feature_names=feature_cols)

    with tab3:
        st.dataframe(plot_df.tail(10))

if __name__ == "__main__":
    main()
