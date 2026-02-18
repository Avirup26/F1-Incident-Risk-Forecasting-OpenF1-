"""
SC/VSC event detector from race_control messages.

This is a critical component: it identifies the *start* of each Safety Car
or Virtual Safety Car deployment from OpenF1 race_control data.

Strategy:
1. Check `category` and `flag` fields first (structured data)
2. Fall back to keyword matching in the `message` text field
3. Group contiguous messages into events, take the first timestamp per event
"""
import re
from datetime import datetime
from typing import Optional

import pandas as pd

from src.utils.logger import logger
from src.utils.time_utils import parse_timestamp_series

# Keywords that indicate a SC/VSC deployment (not a period end)
SC_KEYWORDS = re.compile(
    r"\b(SAFETY CAR|VIRTUAL SAFETY CAR|VSC)\b.*\b(DEPLOYED|OUT|PERIOD)\b",
    re.IGNORECASE,
)
SC_CATEGORIES = {"SafetyCar", "Vsc", "VirtualSafetyCar"}
SC_FLAGS = {"SC", "VSC", "SAFETY CAR", "VIRTUAL SAFETY CAR"}

# Keywords that indicate the *end* of a SC/VSC period (to exclude)
END_KEYWORDS = re.compile(
    r"\b(ENDING|WITHDRAWN|IN THIS LAP|RESUME|CLEAR)\b",
    re.IGNORECASE,
)


def _is_sc_start(row: pd.Series) -> bool:
    """
    Determine if a race_control row represents a SC/VSC deployment start.

    Args:
        row: A single race_control record as a pandas Series.

    Returns:
        True if this message signals a SC/VSC start.
    """
    category = str(row.get("category", "") or "")
    flag = str(row.get("flag", "") or "")
    message = str(row.get("message", "") or "")

    # Exclude clear/end messages
    if END_KEYWORDS.search(message):
        return False

    # Structured fields first
    if category in SC_CATEGORIES or flag in SC_FLAGS:
        return True

    # Text fallback
    if SC_KEYWORDS.search(message):
        return True

    return False


def detect_sc_events(race_control_df: pd.DataFrame) -> list[datetime]:
    """
    Detect SC/VSC deployment start timestamps from race_control data.

    Groups contiguous SC messages into single events and returns the
    first timestamp of each event window.

    Args:
        race_control_df: DataFrame of race_control messages for one session.
            Must have 'date' (UTC datetime) and 'message' columns.

    Returns:
        Sorted list of UTC-aware datetimes for each SC/VSC start event.
    """
    if race_control_df.empty:
        return []

    df = race_control_df.copy()

    # Ensure UTC timestamps
    if "date" in df.columns:
        df["date"] = parse_timestamp_series(df["date"])
    else:
        logger.warning("race_control DataFrame missing 'date' column.")
        return []

    df = df.sort_values("date").reset_index(drop=True)

    # Flag SC/VSC rows
    df["_is_sc"] = df.apply(_is_sc_start, axis=1)

    sc_rows = df[df["_is_sc"]].copy()
    if sc_rows.empty:
        logger.debug("No SC/VSC events detected in race_control data.")
        return []

    # Group contiguous SC messages (within 5-minute windows) into single events
    events: list[datetime] = []
    last_event_time: Optional[pd.Timestamp] = None
    merge_window_seconds = 300  # 5 minutes

    for _, row in sc_rows.iterrows():
        ts = row["date"]
        if last_event_time is None or (ts - last_event_time).total_seconds() > merge_window_seconds:
            events.append(ts.to_pydatetime())
            last_event_time = ts

    logger.info(f"Detected {len(events)} SC/VSC event(s).")
    return sorted(events)
