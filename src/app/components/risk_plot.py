
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def render_risk_plot(timeline_df: pd.DataFrame, threshold: float = 0.5) -> go.Figure:
    """
    Render an interactive Plotly timeline of SC/VSC risk probability.

    Args:
        timeline_df: DataFrame containing 'timestamp', 'prob_lgbm', 'y_sc_5m'.
        threshold: Decision threshold for high risk highlighting.

    Returns:
        Plotly Figure object.
    """
    df = timeline_df.sort_values("timestamp").copy()
    
    # Create figure
    fig = go.Figure()

    # 1. Risk Score Line
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["prob_lgbm"],
        mode="lines",
        name="SC Probability",
        line=dict(color="#FF4B4B", width=2),
        hovertemplate="Time: %{x}<br>Risk: %{y:.1%}<extra></extra>"
    ))

    # 2. Threshold Line
    fig.add_hline(
        y=threshold, 
        line_dash="dot", 
        line_color="gray", 
        annotation_text=f"Threshold ({threshold:.2f})",
        annotation_position="bottom right"
    )

    # 3. Actual SC/VSC Events (Markers)
    # Filter for moments where a new SC event starts (y_sc_5m transitions/is active)
    # A cleaner way: use known SC start events from race control if available,
    # otherwise infer from positive labels.
    # Here we highlight regions where y_sc_5m is 1 (Positive Label Window)
    
    # Use fill for positive label regions
    # Create a boolean mask for positive labels
    mask = df["y_sc_5m"] == 1
    if mask.any():
        # Add a trace that is only present where mask is True
        # We can use fill='tozeroy' with specific x/y, but gaps are tricky.
        # simpler: add markers on the line or a shaded region.
        
        # Let's add markers for high risk moments
        high_risk = df[df["prob_lgbm"] > threshold]
        if not high_risk.empty:
            fig.add_trace(go.Scatter(
                x=high_risk["timestamp"],
                y=high_risk["prob_lgbm"],
                mode="markers",
                name="High Risk Alert",
                marker=dict(color="red", size=6, symbol="circle"),
                showlegend=False
            ))

        # Add vertical bands for actual SC events labels (where we predict 1)
        # This requires identifying contiguous blocks.
        # For simplicity in this version, we'll just plot the binary label on a secondary y-axis or as a bar.
        
        # Actually, let's just use markers for actual positive labels (SC window) at y=1.05
        true_pos = df[df["y_sc_5m"] == 1]
        if not true_pos.empty:
             fig.add_trace(go.Scatter(
                x=true_pos["timestamp"],
                y=[1.02] * len(true_pos),
                mode="markers",
                name="Actual SC/VSC Window",
                marker=dict(color="yellow", size=5, symbol="square"),
                hovertemplate="Actual SC Window<extra></extra>"
            ))


    # Layout updates
    fig.update_layout(
        title="Real-time Safety Car Risk Prediction",
        xaxis_title="Race Time (UTC)",
        yaxis_title="Probability",
        yaxis=dict(range=[-0.05, 1.1]),
        template="plotly_dark",
        height=500,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig
