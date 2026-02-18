
import pandas as pd
import streamlit as st
import plotly.express as px

def render_feature_importance(model, feature_names: list[str], top_n: int = 20):
    """
    Render feature importance bar chart from a trained LightGBM model.
    """
    if not hasattr(model, "feature_importance"):
        st.warning("Model does not expose feature importance.")
        return

    # Get importance
    importance = model.feature_importance(importance_type="gain")
    
    # Create DataFrame
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values("importance", ascending=True).tail(top_n)

    # Plot
    fig = px.bar(
        df, 
        x="importance", 
        y="feature", 
        orientation="h",
        title=f"Top {top_n} Features (gain)",
        template="plotly_dark",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
