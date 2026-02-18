
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def render_risk_plot(timeline_df: pd.DataFrame, threshold: float = 0.5) -> go.Figure:
    """
    Render an interactive Plotly timeline of SC/VSC risk probability with F1 aesthetics and High Legibility.

    Args:
        timeline_df: DataFrame containing 'timestamp', 'prob_lgbm', 'y_sc_5m'.
        threshold: Decision threshold for high risk highlighting.

    Returns:
        Plotly Figure object.
    """
    df = timeline_df.sort_values("timestamp").copy()
    
    # F1 Color Palette (High Contrast)
    F1_RED = "#FF1801" # Brighter Official Red
    
    fig = go.Figure()

    # 1. Risk Score Line with Fill
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["prob_lgbm"],
        mode="lines",
        name="SC PROBABILITY",
        line=dict(color=F1_RED, width=4), # Thicker line
        fill='tozeroy',
        fillcolor='rgba(255, 24, 1, 0.2)', 
        hovertemplate="<b>Time</b>: %{x}<br><b>Risk</b>: %{y:.1%}<extra></extra>"
    ))

    # 2. Threshold Line
    fig.add_hline(
        y=threshold, 
        line_dash="dot", 
        line_color="white", 
        line_width=2,
        annotation_text=f"THRESHOLD ({threshold:.2f})",
        annotation_position="bottom right",
        annotation_font=dict(color="white", size=14, family="Titillium Web, sans-serif")
    )

    # 3. High Risk Markers
    # Highlight points above threshold
    high_risk = df[df["prob_lgbm"] > threshold]
    if not high_risk.empty:
        fig.add_trace(go.Scatter(
            x=high_risk["timestamp"],
            y=high_risk["prob_lgbm"],
            mode="markers",
            name="HIGH RISK ALERT",
            marker=dict(color="white", size=6, line=dict(color=F1_RED, width=2)),
            showlegend=False,
            hoverinfo="skip"
        ))

    # 4. Actual SC/VSC Windows (Ground Truth)
    true_pos = df[df["y_sc_5m"] == 1]
    if not true_pos.empty:
         fig.add_trace(go.Scatter(
            x=true_pos["timestamp"],
            y=[1.02] * len(true_pos),
            mode="markers",
            name="ACTUAL SC/VSC",
            marker=dict(color="#FFD700", size=8, symbol="diamond"), 
            hovertemplate="Actual SC Window<extra></extra>"
        ))


    # Layout updates for "F1 Style" & Legibility
    fig.update_layout(
        title=dict(
            text="<b>SAFETY CAR PROBABILITY</b>",
            font=dict(family="Titillium Web, sans-serif", size=28, color="white")
        ),
        xaxis_title="RACE TIME (UTC)",
        yaxis_title="PROBABILITY",
        yaxis=dict(
            range=[-0.05, 1.1], 
            gridcolor="#555", # Brighter grid
            zerolinecolor="#777",
            tickfont=dict(size=16, color="white"),
            title_font=dict(size=20, color="white")
        ),
        xaxis=dict(
            gridcolor="#555",
            tickfont=dict(size=16, color="white"),
            title_font=dict(size=20, color="white")
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)', 
        height=550, # Slightly taller
        hovermode="x unified",
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="right", 
            x=1,
            font=dict(family="Titillium Web, sans-serif", size=16, color="white"),
            bgcolor="rgba(0,0,0,0.5)"
        ),
        margin=dict(l=60, r=20, t=80, b=60),
        font=dict(family="Titillium Web, sans-serif", color="white")
    )

    return fig
