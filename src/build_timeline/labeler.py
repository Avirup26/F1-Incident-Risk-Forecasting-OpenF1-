"""
Label assignment: applies SC/VSC binary labels to the timeline grid.

For each grid timestamp t, we ask:
  "Will a Safety Car or VSC be deployed in the next 5 minutes?"

Labels:
  y_sc_5m: 1 if any SC/VSC event starts in (t, t + 5min], else 0
  time_to_sc_seconds: seconds until next SC/VSC event (capped at 1800s)
"""
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from src.config import cfg
from src.utils.logger import logger


def assign_labels(
    timeline_df: pd.DataFrame,
    sc_events: list[datetime],
    horizon_seconds: int | None = None,
    cap_seconds: int | None = None,
) -> pd.DataFrame:
    """
    Assign SC/VSC prediction labels to a timeline grid.

    Args:
        timeline_df: Timeline DataFrame with 'timestamp' column (UTC-aware).
        sc_events: List of UTC-aware SC/VSC start datetimes.
        horizon_seconds: Prediction horizon in seconds (default from config).
        cap_seconds: Cap for time_to_sc_seconds (default from config).

    Returns:
        timeline_df with two new columns:
          - y_sc_5m: binary label (int)
          - time_to_sc_seconds: float, seconds to next event (capped)
    """
    horizon = horizon_seconds or cfg.features.prediction_horizon_seconds
    cap = cap_seconds or cfg.features.time_to_event_cap

    df = timeline_df.copy()

    if not sc_events:
        df["y_sc_5m"] = 0
        df["time_to_sc_seconds"] = float(cap)
        return df

    # Convert events to UTC-aware pandas Timestamps for vectorized ops
    event_ts = pd.DatetimeIndex(
        [pd.Timestamp(e).tz_localize("UTC") if e.tzinfo is None else pd.Timestamp(e) for e in sc_events]
    )

    timestamps = df["timestamp"]

    # Vectorized label computation
    y_labels = np.zeros(len(df), dtype=int)
    time_to_next = np.full(len(df), float(cap))

    for i, t in enumerate(timestamps):
        # Events strictly after t
        future_events = event_ts[event_ts > t]
        if len(future_events) == 0:
            continue

        next_event = future_events[0]
        delta_seconds = (next_event - t).total_seconds()

        time_to_next[i] = min(delta_seconds, cap)

        if delta_seconds <= horizon:
            y_labels[i] = 1

    df["y_sc_5m"] = y_labels
    df["time_to_sc_seconds"] = time_to_next

    positive_rate = y_labels.mean() * 100
    logger.debug(
        f"Labels assigned: {y_labels.sum()} positive / {len(y_labels)} total "
        f"({positive_rate:.1f}% positive rate)"
    )
    return df
