"""
Feature importance visualization component.
"""
import pandas as pd
import plotly.express as px
import streamlit as st


def render_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
) -> None:
    """
    Render a horizontal bar chart of feature importances.

    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns.
        top_n: Number of top features to display.
    """
    if importance_df.empty:
        st.info("Feature importance not available.")
        return

    top = importance_df.head(top_n).sort_values("importance")

    fig = px.bar(
        top,
        x="importance",
        y="feature",
        orientation="h",
        title=f"Top {top_n} Feature Importances (LightGBM)",
        color="importance",
        color_continuous_scale="Reds",
        template="plotly_dark",
    )
    fig.update_layout(
        height=500,
        yaxis_title="",
        xaxis_title="Importance Score",
        coloraxis_showscale=False,
        margin=dict(l=20, r=20, t=50, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)
