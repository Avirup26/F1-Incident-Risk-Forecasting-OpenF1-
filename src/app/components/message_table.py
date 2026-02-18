
import pandas as pd
import streamlit as st
import numpy as np

def render_message_table(timeline_df: pd.DataFrame, messages_df: pd.DataFrame, selected_timestamp: np.datetime64 = None):
    """
    Render a table of race control messages, filtered by selected timestamp if applicable.
    """
    
    # 1. Prepare View
    display_df = messages_df.copy()
    
    # Filter
    if selected_timestamp is not None:
        # Show messages up to this timestamp, or in a window around it?
        # Typically we want to see messages leading up to the risk spike.
        # Let's show messages in the last 5 minutes leading up to the selected time.
        window = pd.Timedelta(minutes=5)
        mask = (display_df["date"] <= selected_timestamp) & (display_df["date"] >= selected_timestamp - window)
        display_df = display_df[mask]
        st.caption(f"Showing messages from 5 min window ending at {selected_timestamp}")
    else:
        st.caption("Showing all race control messages")

    if display_df.empty:
        st.info("No messages in this window.")
        return

    # Style
    # Highlight specific keywords?
    def highlight_category(val):
        color = ""
        if "SafetyCar" in str(val):
            color = "background-color: #550000"
        elif "VirtualSafetyCar" in str(val):
            color = "background-color: #554400"
        return color

    st.dataframe(
        display_df[["date", "category", "message", "flag", "driver_number"]]
        .sort_values("date", ascending=False)
        .style.applymap(highlight_category, subset=["category"]),
        use_container_width=True,
        hide_index=True
    )
