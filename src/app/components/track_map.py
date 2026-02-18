
import streamlit as st
import plotly.express as px
import pandas as pd

def render_track_map(position_df: pd.DataFrame, driver_map: dict, current_time: pd.Timestamp):
    """
    Render a track map with car positions at the given time.
    
    Args:
        position_df: DataFrame with X, Y, Z, date, driver_number.
        driver_map: Dict mapping driver_number (str) -> driver_abbr (str).
        current_time: The simulation timestamp to plot.
    """
    if position_df.empty:
        st.warning("No position data available.")
        return

    # Filter for the nearest timestamp (e.g., within 1 second?)
    # FastF1 position data is high frequency.
    # We can use asof merge or just simple nearest search if small.
    # But position_df is huge. We should have loaded only relevant columns.
    
    # Assuming position_df is already filtered or we filter it here.
    # To be efficient, main.py should pass a slice or we optimize here.
    # For now, let's find the slice near current_time.
    
    # Define a window: current_time +/- 1 second coverage
    # Actually, we just want the *last known position* up to current_time 
    # for each driver? Or exactly at current_time?
    
    # Let's take the snapshot at current_time (nearest)
    # Using merge_asof logic ideally, but here we iterate?
    
    # Improve: Filter close to time
    # FastF1 data usually has gaps.
    # Let's subset 5 seconds around current time and take nearest for each driver
    
    window = pd.Timedelta(seconds=2)
    subset = position_df[
        (position_df["date"] >= current_time - window) & 
        (position_df["date"] <= current_time + window)
    ]
    
    if subset.empty:
        st.info(f"No cars on track at {current_time.time()}")
        return

    # Get nearest row for each driver
    records = []
    subset = subset.sort_values("date")
    for drv_num in subset["driver_number"].unique():
        drv_data = subset[subset["driver_number"] == drv_num]
        
        # Find row closest to current_time
        # Abs difference
        idx = (drv_data["date"] - current_time).abs().idxmin()
        row = drv_data.loc[idx]
        
        records.append({
            "driver_abbr": driver_map.get(str(drv_num), str(drv_num)),
            "x": row["x"],
            "y": row["y"],
            "date": row["date"]
        })
        
    plot_data = pd.DataFrame(records)
    
    # Plot
    # We want a fixed range for the map so it doesn't jump around.
    # The range should be determined by the full session min/max X/Y.
    # passed in or calculated from full df? 
    # Calculating from full position_df is expensive if huge.
    # For now, let's let autoscaling happen or calculate once in main.
    
    # To make it "attractive", pitch black background, neon dots.
    
    fig = px.scatter(
        plot_data,
        x="x",
        y="y",
        text="driver_abbr",
        color="driver_abbr",
        title=f"Track Map at {current_time.strftime('%H:%M:%S')}",
        template="plotly_dark",
        height=500,
        width=500 
    )
    
    fig.update_traces(
        marker=dict(size=12, line=dict(width=1, color="white")),
        textposition="top center",
        textfont=dict(size=10, color="white")
    )
    
    # Fix axes to prevent jitter
    # We need global min/max. 
    # Let's assume the component caller handles axes range? 
    # Or we calculate it from the full passed df?
    # If we pass full df, we can get min/max.
    
    x_min, x_max = position_df["x"].min(), position_df["x"].max()
    y_min, y_max = position_df["y"].min(), position_df["y"].max()
    
    padding = 2000 # FastF1 units
    fig.update_xaxes(range=[x_min - padding, x_max + padding], showgrid=False, visible=False)
    fig.update_yaxes(range=[y_min - padding, y_max + padding], showgrid=False, visible=False, scaleanchor="x", scaleratio=1)
    
    st.plotly_chart(fig, use_container_width=True)
