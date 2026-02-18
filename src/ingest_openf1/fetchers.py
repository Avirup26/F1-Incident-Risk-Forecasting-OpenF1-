"""
Endpoint-specific fetchers for the OpenF1 API.

Each fetcher returns a list of raw record dicts for a given session.
"""
from typing import Optional
import pandas as pd

from src.ingest_openf1.api_client import OpenF1Client
from src.utils.logger import logger


def fetch_sessions(
    client: OpenF1Client,
    year: int,
    session_name: str = "Race",
) -> list[dict]:
    """
    Fetch all race sessions for a given year.

    Args:
        client: OpenF1Client instance.
        year: F1 season year (e.g. 2024).
        session_name: Session type to filter (default 'Race').

    Returns:
        List of session metadata dicts.
    """
    logger.info(f"Fetching sessions for {year} ({session_name})...")
    records = client.get("/sessions", params={"year": year, "session_name": session_name})
    logger.info(f"Found {len(records)} sessions.")
    return records


def fetch_race_control(
    client: OpenF1Client,
    session_key: int,
) -> list[dict]:
    """
    Fetch race control messages for a session.

    These contain Safety Car, VSC, flag, and incident messages.

    Args:
        client: OpenF1Client instance.
        session_key: Unique session identifier.

    Returns:
        List of race control message dicts.
    """
    logger.debug(f"Fetching race_control for session {session_key}...")
    return client.get("/race_control", params={"session_key": session_key})


def fetch_weather(
    client: OpenF1Client,
    session_key: int,
) -> list[dict]:
    """
    Fetch weather data for a session.

    Args:
        client: OpenF1Client instance.
        session_key: Unique session identifier.

    Returns:
        List of weather reading dicts (minute-level).
    """
    logger.debug(f"Fetching weather for session {session_key}...")
    return client.get("/weather", params={"session_key": session_key})


def fetch_position(
    client: OpenF1Client,
    session_key: int,
) -> list[dict]:
    """
    Fetch driver position data for a session.

    Args:
        client: OpenF1Client instance.
        session_key: Unique session identifier.

    Returns:
        List of position records (driver × timestamp).
    """
    logger.debug(f"Fetching position for session {session_key}...")
    return client.get("/position", params={"session_key": session_key})


def fetch_intervals(
    client: OpenF1Client,
    session_key: int,
) -> list[dict]:
    """
    Fetch interval (gap to leader) data for a session.

    Args:
        client: OpenF1Client instance.
        session_key: Unique session identifier.

    Returns:
        List of interval records.
    """
    logger.debug(f"Fetching intervals for session {session_key}...")
    return client.get("/intervals", params={"session_key": session_key})


def fetch_drivers(
    client: OpenF1Client,
    session_key: int,
) -> list[dict]:
    """
    Fetch driver metadata for a session.

    Args:
        client: OpenF1Client instance.
        session_key: Unique session identifier.

    Returns:
        List of driver metadata dicts.
    """
    logger.debug(f"Fetching drivers for session {session_key}...")
    return client.get("/drivers", params={"session_key": session_key})


def fetch_all_for_session(
    client: OpenF1Client,
    session: dict,
) -> dict[str, list[dict]]:
    """
    Fetch all relevant endpoints for a single session.

    Args:
        client: OpenF1Client instance.
        session: Session metadata dict (must contain 'session_key').

    Returns:
        Dict mapping endpoint name → list of records.
    """
    session_key = session["session_key"]
    logger.info(f"Fetching all endpoints for session {session_key} ({session.get('meeting_name', '?')})...")

    return {
        "race_control": fetch_race_control(client, session_key),
        "weather": fetch_weather(client, session_key),
        "position": fetch_position(client, session_key),
        "intervals": fetch_intervals(client, session_key),
        "drivers": fetch_drivers(client, session_key),
    }
