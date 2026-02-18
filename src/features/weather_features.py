"""
Weather features: as-of join + rolling aggregations.

Features:
  - rainfall, track_temperature, air_temperature, wind_speed, humidity, pressure
    (as-of joined — last known value at time t)
  - max_rainfall_5m: max rainfall in last 5 minutes
  - track_temp_delta_5m: track_temperature change over last 5 minutes
"""
import numpy as np
import pandas as pd

from src.config import cfg
from src.features.asof_join import asof_join_by_session
from src.features.rolling_utils import rolling_agg
from src.utils.logger import logger
from src.utils.time_utils import parse_timestamp_series

WEATHER_COLS = [
    "rainfall", "track_temperature", "air_temperature",
    "wind_speed", "humidity", "pressure",
]


def build_weather_features(
    timeline_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    session_col: str = "session_key",
) -> pd.DataFrame:
    """
    Compute weather features for each timeline grid point.

    Args:
        timeline_df: Timeline grid with 'timestamp' and 'session_key'.
        weather_df: Weather readings with 'date', weather columns, 'session_key'.

    Returns:
        timeline_df with weather feature columns added.
    """
    if weather_df.empty:
        logger.warning("Weather DataFrame is empty — skipping weather features.")
        for col in WEATHER_COLS + ["max_rainfall_5m", "track_temp_delta_5m"]:
            timeline_df[col] = np.nan
        return timeline_df

    weather_df = weather_df.copy()
    weather_df["date"] = parse_timestamp_series(weather_df["date"])
    weather_df = weather_df.rename(columns={"date": "timestamp"})

    # As-of join for raw weather values
    right_cols = ["timestamp", session_col] + [c for c in WEATHER_COLS if c in weather_df.columns]
    weather_slim = weather_df[right_cols].dropna(subset=["timestamp"])

    merged = asof_join_by_session(
        timeline_df,
        weather_slim,
        on="timestamp",
        session_col=session_col,
        tolerance="10min",
    )

    # Rolling derived features per session
    result_frames = []
    for session_key in merged[session_col].unique():
        tl = merged[merged[session_col] == session_key].copy()
        w = weather_slim[weather_slim[session_col] == session_key].copy()

        if w.empty or "rainfall" not in w.columns:
            tl["max_rainfall_5m"] = np.nan
            tl["track_temp_delta_5m"] = np.nan
            result_frames.append(tl)
            continue

        # max rainfall in last 5 minutes
        tl["max_rainfall_5m"] = rolling_agg(
            tl["timestamp"], w, "timestamp", "rainfall",
            window_seconds=cfg.features.prediction_horizon_seconds,
            agg_fn="max",
        )

        # track temp delta: current - min in last 5 minutes
        if "track_temperature" in w.columns:
            min_temp = rolling_agg(
                tl["timestamp"], w, "timestamp", "track_temperature",
                window_seconds=cfg.features.prediction_horizon_seconds,
                agg_fn="min",
            )
            tl["track_temp_delta_5m"] = tl["track_temperature"].values - min_temp
        else:
            tl["track_temp_delta_5m"] = np.nan

        result_frames.append(tl)

    return pd.concat(result_frames, ignore_index=True) if result_frames else merged
