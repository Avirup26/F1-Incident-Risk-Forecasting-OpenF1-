"""
Ingestion pipeline orchestration.

Fetches all OpenF1 data for a season, saves raw JSON, and converts
to Parquet bronze tables for fast downstream processing.
"""
import json
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from src.config import cfg
from src.ingest_openf1.api_client import OpenF1Client
from src.ingest_openf1.fetchers import fetch_sessions, fetch_all_for_session
from src.utils.logger import logger
from src.utils.time_utils import parse_timestamp_series


def _save_raw_json(data: list[dict], path: Path) -> None:
    """Save raw API response as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, default=str)


def _records_to_parquet(records: list[dict], path: Path) -> None:
    """Convert list of dicts to a Parquet file."""
    if not records:
        logger.warning(f"No records to save → {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_parquet(path, index=False)
    logger.debug(f"Saved {len(df)} rows → {path.name}")


def _parse_timestamps_inplace(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Parse known timestamp columns to UTC-aware datetimes."""
    for col in cols:
        if col in df.columns:
            df[col] = parse_timestamp_series(df[col])
    return df


TIMESTAMP_COLS: dict[str, list[str]] = {
    "race_control": ["date"],
    "weather": ["date"],
    "position": ["date"],
    "intervals": ["date"],
    "drivers": [],
}


def run_ingestion_pipeline(
    year: int = 2024,
    limit: Optional[int] = None,
    force_refresh: bool = False,
) -> None:
    """
    Main ingestion pipeline entry point.

    For each race session in the given year:
    1. Fetches all endpoint data from OpenF1 API
    2. Saves raw JSON to data/raw/{year}/{session_key}/
    3. Converts to Parquet bronze tables in data/bronze/{year}/{session_key}/

    Args:
        year: F1 season year.
        limit: Max number of sessions to process (useful for testing).
        force_refresh: If True, bypass cache and re-fetch all data.
    """
    client = OpenF1Client(use_cache=not force_refresh)

    # Fetch session list
    sessions = fetch_sessions(client, year=year)
    if not sessions:
        logger.warning(f"No sessions found for {year}. Exiting.")
        return

    if limit is not None:
        sessions = sessions[:limit]
        logger.info(f"Limiting to {limit} sessions.")

    logger.info(f"Processing {len(sessions)} sessions for {year}...")

    for session in tqdm(sessions, desc="Sessions", unit="session"):
        session_key = session["session_key"]
        meeting_name = session.get("meeting_name", "Unknown")
        logger.info(f"→ {meeting_name} (session_key={session_key})")

        # Save session metadata
        raw_session_dir = cfg.paths.raw / str(year) / str(session_key)
        _save_raw_json([session], raw_session_dir / "session.json")

        # Fetch all endpoints
        try:
            endpoint_data = fetch_all_for_session(client, session)
        except Exception as e:
            logger.error(f"Failed to fetch session {session_key}: {e}")
            continue

        # Save raw JSON + convert to bronze Parquet
        bronze_dir = cfg.paths.bronze / str(year) / str(session_key)
        for endpoint_name, records in endpoint_data.items():
            # Raw JSON
            _save_raw_json(records, raw_session_dir / f"{endpoint_name}.json")

            # Bronze Parquet
            df = pd.DataFrame(records) if records else pd.DataFrame()
            if not df.empty:
                ts_cols = TIMESTAMP_COLS.get(endpoint_name, [])
                df = _parse_timestamps_inplace(df, ts_cols)
                # Attach session metadata
                df["session_key"] = session_key
                df["meeting_key"] = session.get("meeting_key")
                df["meeting_name"] = meeting_name
                df["year"] = year

            _records_to_parquet(df.to_dict("records") if not df.empty else [], bronze_dir / f"{endpoint_name}.parquet")

        # Save session metadata as bronze too
        _records_to_parquet([session], bronze_dir / "session.parquet")

    logger.info(f"✅ Ingestion complete for {year}. {len(sessions)} sessions processed.")
