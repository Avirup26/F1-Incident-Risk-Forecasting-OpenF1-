"""
Race dynamics features from position and laps/intervals data.

Works with both FastF1 (pos_data) and legacy OpenF1 (position + intervals).

Features:
  - position_changes_120s: total position swaps across all drivers in last 2 min
  - driver_position_volatility_300s: avg |position change| per driver in last 5 min
  - gap_std_120s: std dev of gap_to_leader across the field in last 2 min
    (only when gap_to_leader column is present — not available in FastF1 laps)
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
        position_df: Driver positions with 'date', 'driver_number'.
                     FastF1 pos_data has X/Y/Z coordinates; OpenF1 has 'position'.
        intervals_df: Intervals/laps with 'date', 'driver_number', optionally
                      'gap_to_leader', 'interval'. FastF1 laps don't have
                      gap_to_leader — interval features are skipped if absent.

    Returns:
        timeline_df with dynamics feature columns added.
    """
    result_frames = []
    sessions = timeline_df[session_col].unique()

    for session_key in sessions:
        tl = timeline_df[timeline_df[session_col] == session_key].copy()

        # Filter position data — handle missing session_col gracefully
        if not position_df.empty and session_col in position_df.columns:
            pos = position_df[position_df[session_col] == session_key].copy()
        elif not position_df.empty:
            pos = position_df.copy()  # single-session fallback
        else:
            pos = pd.DataFrame()

        # Filter intervals/laps data — handle missing session_col gracefully
        if not intervals_df.empty and session_col in intervals_df.columns:
            ivl = intervals_df[intervals_df[session_col] == session_key].copy()
        elif not intervals_df.empty:
            ivl = intervals_df.copy()  # single-session fallback
        else:
            ivl = pd.DataFrame()

        # Default columns
        tl["position_changes_120s"] = 0
        tl["driver_position_volatility_300s"] = 0.0
        tl["gap_std_120s"] = np.nan
        tl["pack_density_1_5s"] = np.nan
        tl["pack_density_3_0s"] = np.nan

        # --- Position features ---
        # FastF1 pos_data has X/Y/Z (GPS coords), not race position.
        # Race position is in laps data. Use 'position' column if available.
        if not pos.empty and "date" in pos.columns and "position" in pos.columns:
            pos["date"] = parse_timestamp_series(pos["date"])
            pos["_ts"] = np.asarray(pos["date"], dtype="datetime64[ns]")
            pos["position"] = pd.to_numeric(pos["position"], errors="coerce")
            pos = pos.dropna(subset=["_ts", "position"])

            if not pos.empty:
                tl_ts_arr = np.asarray(tl["timestamp"], dtype="datetime64[ns]")
                w120 = np.timedelta64(cfg.features.dynamics_window, "s")
                w300 = np.timedelta64(cfg.features.dynamics_long_window, "s")

                tl["position_changes_120s"] = [
                    _position_changes_in_window(pos, t, w120) for t in tl_ts_arr
                ]
                tl["driver_position_volatility_300s"] = [
                    _position_volatility_in_window(pos, t, w300) for t in tl_ts_arr
                ]

        # --- Interval / gap features ---
        # FastF1 laps don't have gap_to_leader — skip gracefully.
        has_gap = (
            not ivl.empty
            and "date" in ivl.columns
            and "gap_to_leader" in ivl.columns
        )
        if has_gap:
            ivl["date"] = parse_timestamp_series(ivl["date"])
            ivl["_ts"] = np.asarray(ivl["date"], dtype="datetime64[ns]")
            ivl["gap_to_leader"] = pd.to_numeric(ivl.get("gap_to_leader", np.nan), errors="coerce")
            ivl["interval"] = pd.to_numeric(ivl.get("interval", np.nan), errors="coerce")
            ivl = ivl.dropna(subset=["_ts"])

            tl_ts_arr = np.asarray(tl["timestamp"], dtype="datetime64[ns]")
            w120_ns = np.timedelta64(cfg.features.dynamics_window, "s")

            gap_stds, pack_1_5, pack_3_0 = [], [], []

            for t in tl_ts_arr:
                lo = t - w120_ns
                window = ivl[(ivl["_ts"] > lo) & (ivl["_ts"] <= t)]

                if window.empty:
                    gap_stds.append(np.nan)
                    pack_1_5.append(np.nan)
                    pack_3_0.append(np.nan)
                    continue

                latest = window.sort_values("_ts").groupby("driver_number").last()
                gaps = latest["gap_to_leader"].dropna()
                intervals_vals = latest["interval"].dropna()

                gap_stds.append(gaps.std() if len(gaps) > 1 else np.nan)

                n_drivers = len(intervals_vals)
                if n_drivers > 0:
                    pack_1_5.append((intervals_vals < 1.5).sum() / n_drivers)
                    pack_3_0.append((intervals_vals < 3.0).sum() / n_drivers)
                else:
                    pack_1_5.append(np.nan)
                    pack_3_0.append(np.nan)

            tl["gap_std_120s"] = gap_stds
            tl["pack_density_1_5s"] = pack_1_5
            tl["pack_density_3_0s"] = pack_3_0

        result_frames.append(tl)

    return pd.concat(result_frames, ignore_index=True) if result_frames else timeline_df
