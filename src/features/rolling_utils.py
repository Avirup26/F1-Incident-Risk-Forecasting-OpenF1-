"""
Rolling window utilities for irregular time series.

These helpers compute rolling aggregations over a time-based window
(e.g., "count of events in the last 60 seconds") using as-of semantics:
only data with timestamp strictly <= t is included.
"""
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.utils.logger import logger


def rolling_count(
    timeline_ts: pd.Series,
    event_ts: pd.Series,
    window_seconds: int,
) -> np.ndarray:
    """
    For each timestamp in timeline_ts, count events in event_ts
    that fall in (t - window_seconds, t].

    Args:
        timeline_ts: Sorted UTC timestamps for the grid (n,).
        event_ts: Sorted UTC timestamps of events to count.
        window_seconds: Rolling window size in seconds.

    Returns:
        Array of counts, shape (n,).
    """
    timeline_arr = np.asarray(timeline_ts, dtype="datetime64[ns]")
    event_arr = np.asarray(event_ts.dropna(), dtype="datetime64[ns]")
    window_ns = np.timedelta64(window_seconds, "s")

    counts = np.zeros(len(timeline_arr), dtype=int)
    for i, t in enumerate(timeline_arr):
        lo = t - window_ns
        mask = (event_arr > lo) & (event_arr <= t)
        counts[i] = mask.sum()

    return counts


def rolling_agg(
    timeline_ts: pd.Series,
    event_df: pd.DataFrame,
    ts_col: str,
    value_col: str,
    window_seconds: int,
    agg_fn: str = "mean",
) -> np.ndarray:
    """
    For each timeline timestamp, aggregate a value column over a rolling window.

    Args:
        timeline_ts: Sorted UTC timestamps for the grid.
        event_df: DataFrame with timestamp and value columns.
        ts_col: Name of the timestamp column in event_df.
        value_col: Name of the value column to aggregate.
        window_seconds: Rolling window size in seconds.
        agg_fn: Aggregation function: 'mean', 'max', 'min', 'std', 'sum'.

    Returns:
        Array of aggregated values, shape (n,). NaN where no data in window.
    """
    agg_map: dict[str, Callable] = {
        "mean": np.nanmean,
        "max": np.nanmax,
        "min": np.nanmin,
        "std": np.nanstd,
        "sum": np.nansum,
    }
    if agg_fn not in agg_map:
        raise ValueError(f"Unknown agg_fn '{agg_fn}'. Choose from {list(agg_map)}.")

    fn = agg_map[agg_fn]
    timeline_arr = np.asarray(timeline_ts, dtype="datetime64[ns]")
    event_ts_arr = np.asarray(event_df[ts_col].dropna(), dtype="datetime64[ns]")
    values = event_df[value_col].values
    window_ns = np.timedelta64(window_seconds, "s")

    result = np.full(len(timeline_arr), np.nan)
    for i, t in enumerate(timeline_arr):
        lo = t - window_ns
        mask = (event_ts_arr > lo) & (event_ts_arr <= t)
        if mask.any():
            result[i] = fn(values[mask])

    return result


def rolling_unique_count(
    timeline_ts: pd.Series,
    event_df: pd.DataFrame,
    ts_col: str,
    value_col: str,
    window_seconds: int,
) -> np.ndarray:
    """
    Count unique values in a column over a rolling window.

    Args:
        timeline_ts: Sorted UTC timestamps for the grid.
        event_df: DataFrame with timestamp and categorical value columns.
        ts_col: Name of the timestamp column.
        value_col: Name of the categorical column.
        window_seconds: Rolling window size in seconds.

    Returns:
        Array of unique counts, shape (n,).
    """
    timeline_arr = np.asarray(timeline_ts, dtype="datetime64[ns]")
    event_ts_arr = np.asarray(event_df[ts_col].dropna(), dtype="datetime64[ns]")
    cat_values = event_df[value_col].values
    window_ns = np.timedelta64(window_seconds, "s")

    result = np.zeros(len(timeline_arr), dtype=int)
    for i, t in enumerate(timeline_arr):
        lo = t - window_ns
        mask = (event_ts_arr > lo) & (event_ts_arr <= t)
        result[i] = len(set(cat_values[mask]))

    return result
