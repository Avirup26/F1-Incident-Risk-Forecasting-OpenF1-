"""
Race dynamics features from /position and /intervals endpoints.

Features:
  - position_changes_120s: total position swaps across all drivers in last 2 min
  - driver_position_volatility_300s: avg |position change| per driver in last 5 min
  - gap_std_120s: std dev of gap_to_leader across the field in last 2 min
  - pack_density_1_5s: fraction of drivers with gap < 1.5s to car ahead
  - pack_density_3_0s: fraction of drivers with gap < 3.0s to car ahead
"""
import numpy as np
import pandas as pd

from src.config import cfg
from src.features.asof_join import asof_join_by_session
from src.utils.logger import logger
from src.utils.time_utils import parse_timestamp_series


def _position_changes_in_window(
    pos_df: pd.DataFrame,
    t: np.datetime64,
    window_ns: np.timedelta64,
) -> int:
    """Count total position swaps across all drivers in a time window."""
    lo = t - window_ns
    window = pos_df[(pos_df["_ts"] > lo) & (pos_df["_ts"] <= t)]
    if window.empty:
        return 0
    # For each driver, count position changes
    changes = (
        window.sort_values("_ts")
        .groupby("driver_number")["position"]
        .apply(lambda s: (s.diff().abs() > 0).sum())
        .sum()
    )
    return int(changes)


def _position_volatility_in_window(
    pos_df: pd.DataFrame,
    t: np.datetime64,
    window_ns: np.timedelta64,
) -> float:
    """Avg absolute position change per driver in a time window."""
    lo = t - window_ns
    window = pos_df[(pos_df["_ts"] > lo) & (pos_df["_ts"] <= t)]
    if window.empty:
        return 0.0
    per_driver = (
        window.sort_values("_ts")
        .groupby("driver_number")["position"]
        .apply(lambda s: s.diff().abs().mean())
    )
    return float(per_driver.mean()) if not per_driver.empty else 0.0


def build_dynamics_features(
    timeline_df: pd.DataFrame,
    position_df: pd.DataFrame,
    intervals_df: pd.DataFrame,
    session_col: str = "session_key",
) -> pd.DataFrame:
    """
    Compute race dynamics features for each timeline grid point.

    Args:
        timeline_df: Timeline grid with 'timestamp' and 'session_key'.
        position_df: Driver positions with 'date', 'driver_number', 'position'.
        intervals_df: Intervals with 'date', 'driver_number', 'gap_to_leader',
            'interval'.

    Returns:
        timeline_df with dynamics feature columns added.
    """
    result_frames = []
    sessions = timeline_df[session_col].unique()

    for session_key in sessions:
        tl = timeline_df[timeline_df[session_col] == session_key].copy()
        pos = position_df[position_df[session_col] == session_key].copy() if not position_df.empty else pd.DataFrame()
        ivl = intervals_df[intervals_df[session_col] == session_key].copy() if not intervals_df.empty else pd.DataFrame()

        # Default NaN columns
        tl["position_changes_120s"] = 0
        tl["driver_position_volatility_300s"] = 0.0
        tl["gap_std_120s"] = np.nan
        tl["pack_density_1_5s"] = np.nan
        tl["pack_density_3_0s"] = np.nan

        # --- Position features ---
        if not pos.empty and "date" in pos.columns and "position" in pos.columns:
            pos["date"] = parse_timestamp_series(pos["date"])
            pos["_ts"] = np.asarray(pos["date"], dtype="datetime64[ns]")
            pos["position"] = pd.to_numeric(pos["position"], errors="coerce")
            pos = pos.dropna(subset=["_ts", "position"])

            tl_ts_arr = np.asarray(tl["timestamp"], dtype="datetime64[ns]")
            w120 = np.timedelta64(cfg.features.dynamics_window, "s")
            w300 = np.timedelta64(cfg.features.dynamics_long_window, "s")

            tl["position_changes_120s"] = [
                _position_changes_in_window(pos, t, w120) for t in tl_ts_arr
            ]
            tl["driver_position_volatility_300s"] = [
                _position_volatility_in_window(pos, t, w300) for t in tl_ts_arr
            ]

        # --- Interval features ---
        if not ivl.empty and "date" in ivl.columns:
            ivl["date"] = parse_timestamp_series(ivl["date"])
            ivl["_ts"] = np.asarray(ivl["date"], dtype="datetime64[ns]")
            ivl["gap_to_leader"] = pd.to_numeric(ivl.get("gap_to_leader", np.nan), errors="coerce")
            ivl["interval"] = pd.to_numeric(ivl.get("interval", np.nan), errors="coerce")
            ivl = ivl.dropna(subset=["_ts"])

            tl_ts_arr = np.asarray(tl["timestamp"], dtype="datetime64[ns]")
            w120_ns = np.timedelta64(cfg.features.dynamics_window, "s")

            gap_stds = []
            pack_1_5 = []
            pack_3_0 = []

            for t in tl_ts_arr:
                lo = t - w120_ns
                window = ivl[(ivl["_ts"] > lo) & (ivl["_ts"] <= t)]

                if window.empty:
                    gap_stds.append(np.nan)
                    pack_1_5.append(np.nan)
                    pack_3_0.append(np.nan)
                    continue

                # Latest snapshot per driver
                latest = window.sort_values("_ts").groupby("driver_number").last()
                gaps = latest["gap_to_leader"].dropna()
                intervals = latest["interval"].dropna()

                gap_stds.append(gaps.std() if len(gaps) > 1 else np.nan)

                n_drivers = len(intervals)
                if n_drivers > 0:
                    pack_1_5.append((intervals < 1.5).sum() / n_drivers)
                    pack_3_0.append((intervals < 3.0).sum() / n_drivers)
                else:
                    pack_1_5.append(np.nan)
                    pack_3_0.append(np.nan)

            tl["gap_std_120s"] = gap_stds
            tl["pack_density_1_5s"] = pack_1_5
            tl["pack_density_3_0s"] = pack_3_0

        result_frames.append(tl)

    return pd.concat(result_frames, ignore_index=True) if result_frames else timeline_df
