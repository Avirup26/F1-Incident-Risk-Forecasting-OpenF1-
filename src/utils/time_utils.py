"""
Timezone-aware datetime utilities.
All timestamps in this project are stored and processed in UTC.
"""
from datetime import datetime, timezone
from typing import Optional
import pandas as pd


def utc_now() -> datetime:
    """Return current UTC datetime (timezone-aware)."""
    return datetime.now(tz=timezone.utc)


def to_utc(dt: datetime) -> datetime:
    """
    Convert a datetime to UTC. If naive, assume it is already UTC.

    Args:
        dt: Input datetime (aware or naive).

    Returns:
        UTC-aware datetime.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_openf1_timestamp(ts: str) -> Optional[datetime]:
    """
    Parse an OpenF1 API timestamp string to a UTC-aware datetime.

    OpenF1 returns ISO 8601 strings like '2024-03-02T15:00:00.000000+00:00'.

    Args:
        ts: Timestamp string from OpenF1 API.

    Returns:
        UTC-aware datetime, or None if parsing fails.
    """
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
        return to_utc(dt)
    except (ValueError, TypeError):
        return None


def parse_timestamp_series(series: pd.Series) -> pd.Series:
    """
    Parse a pandas Series of timestamp strings to UTC-aware datetimes.

    Args:
        series: Series of ISO 8601 timestamp strings.

    Returns:
        Series of UTC-aware pandas Timestamps.
    """
    parsed = pd.to_datetime(series, utc=True, errors="coerce")
    return parsed


def floor_to_grid(ts: pd.Timestamp, interval_seconds: int = 30) -> pd.Timestamp:
    """
    Floor a timestamp to the nearest grid interval.

    Args:
        ts: Input timestamp.
        interval_seconds: Grid interval in seconds (default 30).

    Returns:
        Floored timestamp.
    """
    freq = f"{interval_seconds}s"
    return ts.floor(freq)
