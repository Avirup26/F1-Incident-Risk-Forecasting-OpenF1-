"""
Race control message table component.

Displays race_control messages for a selected time window with
highlighting for SC/VSC-related messages.
"""
import pandas as pd
import streamlit as st


def render_message_table(
    race_control_df: pd.DataFrame,
    selected_time: pd.Timestamp | None = None,
    window_seconds: int = 300,
) -> None:
    """
    Render a styled race control message table.

    Args:
        race_control_df: DataFrame with 'date', 'message', 'category', 'flag'.
        selected_time: Center time for the display window.
        window_seconds: Half-window size in seconds.
    """
    if race_control_df.empty:
        st.info("No race control messages available.")
        return

    df = race_control_df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.sort_values("date")

    if selected_time is not None:
        lo = selected_time - pd.Timedelta(seconds=window_seconds)
        hi = selected_time + pd.Timedelta(seconds=window_seconds)
        df = df[(df["date"] >= lo) & (df["date"] <= hi)]

    if df.empty:
        st.info("No messages in selected time window.")
        return

    display_cols = [c for c in ["date", "category", "flag", "message"] if c in df.columns]
    display_df = df[display_cols].copy()
    display_df["date"] = display_df["date"].dt.strftime("%H:%M:%S")

    # Highlight SC/VSC rows
    def highlight_sc(row: pd.Series) -> list[str]:
        cat = str(row.get("category", "")).lower()
        msg = str(row.get("message", "")).lower()
        if any(kw in cat or kw in msg for kw in ["safety car", "vsc", "virtual"]):
            return ["background-color: rgba(255, 215, 0, 0.2)"] * len(row)
        return [""] * len(row)

    st.dataframe(
        display_df.style.apply(highlight_sc, axis=1),
        use_container_width=True,
        height=300,
    )
