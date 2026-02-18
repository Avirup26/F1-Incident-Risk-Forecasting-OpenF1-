"""
Alert policy analysis at different probability thresholds.

Simulates an alert system: when risk score exceeds a threshold,
an alert is raised. Measures:
  - Alerts per race
  - Median lead-time to actual SC/VSC events
  - False positive rate
"""
import numpy as np
import pandas as pd

from src.utils.logger import logger


def analyze_alert_policy(
    test_df: pd.DataFrame,
    y_prob: np.ndarray,
    thresholds: list[float] | None = None,
    session_col: str = "session_key",
    label_col: str = "y_sc_5m",
    time_col: str = "timestamp",
    time_to_event_col: str = "time_to_sc_seconds",
) -> pd.DataFrame:
    """
    Analyze alert policy performance at multiple thresholds.

    Args:
        test_df: Test set DataFrame.
        y_prob: Predicted probabilities.
        thresholds: List of threshold values to evaluate.
        session_col: Column identifying sessions.
        label_col: Binary label column.
        time_col: Timestamp column.
        time_to_event_col: Seconds to next SC/VSC event.

    Returns:
        DataFrame with one row per threshold and columns:
        threshold, alerts_per_race, median_lead_time_s,
        true_positive_rate, false_positive_rate, precision.
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    df = test_df.copy()
    df["_prob"] = y_prob
    df["_y"] = df[label_col].values

    n_sessions = df[session_col].nunique()
    results = []

    for thresh in thresholds:
        df["_alert"] = (df["_prob"] >= thresh).astype(int)

        # Alerts per race
        alerts_per_race = df.groupby(session_col)["_alert"].sum().mean()

        # True positives: alert raised when label is 1
        tp = ((df["_alert"] == 1) & (df["_y"] == 1)).sum()
        fp = ((df["_alert"] == 1) & (df["_y"] == 0)).sum()
        fn = ((df["_alert"] == 0) & (df["_y"] == 1)).sum()
        tn = ((df["_alert"] == 0) & (df["_y"] == 0)).sum()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # Median lead time: time_to_sc_seconds for true positive alerts
        tp_mask = (df["_alert"] == 1) & (df["_y"] == 1)
        if tp_mask.any() and time_to_event_col in df.columns:
            median_lead = df.loc[tp_mask, time_to_event_col].median()
        else:
            median_lead = np.nan

        results.append({
            "threshold": thresh,
            "alerts_per_race": round(alerts_per_race, 2),
            "median_lead_time_s": round(median_lead, 1) if not np.isnan(median_lead) else None,
            "true_positive_rate": round(tpr, 4),
            "false_positive_rate": round(fpr, 4),
            "precision": round(precision, 4),
        })

    result_df = pd.DataFrame(results)
    logger.info(f"Alert policy analysis complete for {len(thresholds)} thresholds.")
    return result_df
