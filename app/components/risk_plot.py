"""
Plotly risk timeline component.

Renders the SC/VSC risk score over time with actual event markers.
"""
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def render_risk_plot(
    timeline_df: pd.DataFrame,
    session_name: str = "",
    threshold: float = 0.3,
) -> go.Figure:
    """
    Create an interactive Plotly risk timeline.

    Args:
        timeline_df: DataFrame with 'timestamp', 'risk_score', 'y_sc_5m'.
        session_name: Display name for the session.
        threshold: Alert threshold line to draw.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()

    # Risk score line
    fig.add_trace(go.Scatter(
        x=timeline_df["timestamp"],
        y=timeline_df["risk_score"],
        mode="lines",
        name="SC/VSC Risk Score",
        line=dict(color="#E8002D", width=2),
        fill="tozeroy",
        fillcolor="rgba(232, 0, 45, 0.1)",
    ))

    # Actual SC/VSC events (vertical markers)
    sc_events = timeline_df[timeline_df["y_sc_5m"] == 1]["timestamp"]
    if not sc_events.empty:
        # Find event start timestamps (first in each contiguous block)
        ts_series = timeline_df["y_sc_5m"]
        event_starts = timeline_df[
            (ts_series == 1) & (ts_series.shift(1, fill_value=0) == 0)
        ]["timestamp"]

        for ts in event_starts:
            # Convert to ISO string — add_vline annotation has a midpoint arithmetic bug
            # with datetime strings in newer Plotly+pandas, so use add_shape + add_annotation
            ts_str = pd.Timestamp(ts).isoformat() if not isinstance(ts, str) else ts
            fig.add_shape(
                type="line",
                x0=ts_str, x1=ts_str,
                y0=0, y1=1,
                xref="x", yref="paper",
                line=dict(color="#FFD700", width=2, dash="dash"),
            )
            fig.add_annotation(
                x=ts_str, y=1,
                xref="x", yref="paper",
                text="SC/VSC",
                showarrow=False,
                font=dict(color="#FFD700", size=10),
                yanchor="bottom",
            )

    # Threshold line
    fig.add_hline(
        y=threshold,
        line_dash="dot",
        line_color="rgba(255,255,255,0.5)",
        annotation_text=f"Alert threshold ({threshold:.0%})",
        annotation_position="right",
        annotation_font_color="rgba(255,255,255,0.7)",
    )

    fig.update_layout(
        title=f"🏎️ SC/VSC Risk Timeline — {session_name}",
        xaxis_title="Race Time (UTC)",
        yaxis_title="Risk Score",
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        template="plotly_dark",
        height=400,
        margin=dict(l=60, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )

    return fig
