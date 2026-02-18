"""
Timeline builder: generates a uniform 30-second time grid for each session.

All timestamps are UTC-aware. The grid serves as the backbone for feature
engineering and label assignment.
"""
from typing import Optional
import pandas as pd

from src.config import cfg
from src.utils.logger import logger
from src.utils.time_utils import parse_openf1_timestamp


def build_session_timeline(
    session: dict,
    interval_seconds: int | None = None,
) -> pd.DataFrame:
    """
    Generate a uniform time grid for a single session.

    Args:
        session: Session metadata dict (must have 'date_start' and 'date_end').
        interval_seconds: Grid spacing in seconds (default from config).

    Returns:
        DataFrame with columns: ['session_key', 'meeting_key', 'year',
        'meeting_name', 'timestamp']
    """
    interval = interval_seconds or cfg.features.grid_interval_seconds

    date_start = parse_openf1_timestamp(session.get("date_start", ""))
    date_end = parse_openf1_timestamp(session.get("date_end", ""))

    if date_start is None or date_end is None:
        logger.warning(
            f"Session {session.get('session_key')} missing date_start/date_end — skipping."
        )
        return pd.DataFrame()

    # Build UTC-aware timestamp grid
    timestamps = pd.date_range(
        start=date_start,
        end=date_end,
        freq=f"{interval}s",
        tz="UTC",
    )

    df = pd.DataFrame({"timestamp": timestamps})
    df["session_key"] = session.get("session_key")
    df["meeting_key"] = session.get("meeting_key")
    df["year"] = session.get("year")
    df["meeting_name"] = session.get("meeting_name", "")

    logger.debug(
        f"Session {session.get('session_key')}: {len(df)} grid points "
        f"({date_start} → {date_end}, {interval}s interval)"
    )
    return df[["session_key", "meeting_key", "year", "meeting_name", "timestamp"]]


def build_all_timelines(sessions: list[dict], interval_seconds: int | None = None) -> pd.DataFrame:
    """
    Build and concatenate timelines for all sessions.

    Args:
        sessions: List of session metadata dicts.
        interval_seconds: Grid spacing in seconds.

    Returns:
        Combined DataFrame of all session timelines, sorted by
        (session_key, timestamp).
    """
    frames = []
    for session in sessions:
        df = build_session_timeline(session, interval_seconds)
        if not df.empty:
            frames.append(df)

    if not frames:
        logger.warning("No timelines built — check session date fields.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["session_key", "timestamp"]).reset_index(drop=True)
    logger.info(f"Built timeline: {len(combined)} total grid points across {len(frames)} sessions.")
    return combined
