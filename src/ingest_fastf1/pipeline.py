"""
FastF1 pipeline: fetches race session data and saves bronze Parquet tables.

FastF1 is completely FREE — no API key or subscription required.
It caches all data locally in .cache/fastf1/ automatically.

Data flow:
  FastF1 API → session.load() → bronze/{year}/{session_key}/*.parquet

Bronze tables produced per session:
  - session.parquet        (session metadata)
  - race_control.parquet   (SC/VSC messages, flags, incidents)
  - weather.parquet        (track/air temperature, rainfall, wind)
  - laps.parquet           (lap times, tyre info, pit stops)
  - position.parquet       (driver positions over time)
  - car_data.parquet       (speed, throttle, brake telemetry)
  - results.parquet        (final race results)
"""
from pathlib import Path
from typing import Optional

import fastf1
import pandas as pd
from tqdm import tqdm

from src.config import cfg
from src.utils.logger import logger
from src.utils.time_utils import parse_timestamp_series


# Enable FastF1 disk cache (avoids re-downloading data)
_CACHE_DIR = cfg.paths.cache / "fastf1"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(_CACHE_DIR))


# ── Session discovery ─────────────────────────────────────────────────────────

def get_race_sessions(year: int) -> list[dict]:
    """
    Get all Race sessions for a given year using FastF1's event schedule.

    Args:
        year: F1 season year (2018+).

    Returns:
        List of session metadata dicts.
    """
    logger.info(f"Fetching {year} race schedule...")
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    sessions = []

    for _, event in schedule.iterrows():
        # Skip future events (no data yet)
        try:
            session = fastf1.get_session(year, event["RoundNumber"], "R")
            sessions.append({
                "session_key": int(f"{year}{event['RoundNumber']:02d}"),
                "meeting_key": int(f"{year}{event['RoundNumber']:02d}0"),
                "year": year,
                "round_number": int(event["RoundNumber"]),
                "meeting_name": str(event.get("EventName", event.get("OfficialEventName", "Unknown"))),
                "circuit_short_name": str(event.get("Location", "")),
                "country": str(event.get("Country", "")),
                "session_name": "Race",
                "session_type": "Race",
                "_fastf1_session": session,  # keep reference for loading
            })
        except Exception as e:
            logger.warning(f"Could not get session for round {event['RoundNumber']}: {e}")

    logger.info(f"Found {len(sessions)} race sessions for {year}.")
    return sessions


# ── Data extraction helpers ───────────────────────────────────────────────────

def _add_session_meta(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Attach session_key, meeting_key, year, meeting_name to a DataFrame."""
    df = df.copy()
    df["session_key"] = meta["session_key"]
    df["meeting_key"] = meta["meeting_key"]
    df["year"] = meta["year"]
    df["meeting_name"] = meta["meeting_name"]
    return df


def _extract_race_control(session: fastf1.core.Session, meta: dict) -> pd.DataFrame:
    """Extract race control messages → standardized DataFrame."""
    rc = session.race_control_messages
    if rc is None or rc.empty:
        return pd.DataFrame()

    df = rc.copy().reset_index(drop=True)

    # Standardize column names to match our pipeline
    rename = {
        "Time": "date",
        "Category": "category",
        "Message": "message",
        "Flag": "flag",
        "Scope": "scope",
        "Sector": "sector",
        "RacingNumber": "driver_number",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Convert Time (timedelta from session start) → absolute UTC timestamp
    if "date" in df.columns and hasattr(session, "date"):
        if pd.api.types.is_timedelta64_dtype(df["date"]):
            session_start = pd.Timestamp(session.date).tz_localize("UTC") if session.date.tzinfo is None else pd.Timestamp(session.date).tz_convert("UTC")
            df["date"] = session_start + df["date"]
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")

    return _add_session_meta(df, meta)


def _extract_weather(session: fastf1.core.Session, meta: dict) -> pd.DataFrame:
    """Extract weather data → standardized DataFrame."""
    weather = session.weather_data
    if weather is None or weather.empty:
        return pd.DataFrame()

    df = weather.copy().reset_index(drop=True)

    rename = {
        "Time": "date",
        "AirTemp": "air_temperature",
        "TrackTemp": "track_temperature",
        "Humidity": "humidity",
        "Pressure": "pressure",
        "WindSpeed": "wind_speed",
        "WindDirection": "wind_direction",
        "Rainfall": "rainfall",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    if "date" in df.columns and pd.api.types.is_timedelta64_dtype(df["date"]):
        session_start = pd.Timestamp(session.date).tz_localize("UTC") if session.date.tzinfo is None else pd.Timestamp(session.date).tz_convert("UTC")
        df["date"] = session_start + df["date"]
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")

    # Normalize rainfall: FastF1 returns bool or float
    if "rainfall" in df.columns:
        df["rainfall"] = pd.to_numeric(df["rainfall"].map({True: 1.0, False: 0.0}).fillna(df["rainfall"]), errors="coerce").fillna(0.0)

    return _add_session_meta(df, meta)


def _extract_laps(session: fastf1.core.Session, meta: dict) -> pd.DataFrame:
    """Extract lap data → standardized DataFrame."""
    laps = session.laps
    if laps is None or laps.empty:
        return pd.DataFrame()

    df = laps.copy().reset_index(drop=True)

    rename = {
        "Time": "date",
        "Driver": "driver_abbr",
        "DriverNumber": "driver_number",
        "LapNumber": "lap_number",
        "LapTime": "lap_time",
        "Sector1Time": "sector1_time",
        "Sector2Time": "sector2_time",
        "Sector3Time": "sector3_time",
        "Compound": "tyre_compound",
        "TyreLife": "tyre_life",
        "Stint": "stint",
        "PitInTime": "pit_in_time",
        "PitOutTime": "pit_out_time",
        "Position": "position",
        "TrackStatus": "track_status",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    if "date" in df.columns and pd.api.types.is_timedelta64_dtype(df["date"]):
        session_start = pd.Timestamp(session.date).tz_localize("UTC") if session.date.tzinfo is None else pd.Timestamp(session.date).tz_convert("UTC")
        df["date"] = session_start + df["date"]
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")

    # Convert timedelta columns to seconds
    for col in ["lap_time", "sector1_time", "sector2_time", "sector3_time"]:
        if col in df.columns and pd.api.types.is_timedelta64_dtype(df[col]):
            df[col] = df[col].dt.total_seconds()

    return _add_session_meta(df, meta)


def _extract_position(session: fastf1.core.Session, meta: dict) -> pd.DataFrame:
    """Extract driver position data → standardized DataFrame.

    FastF1 pos_data is a dict of {driver_number: Telemetry} where each
    Telemetry has columns: ['Date', 'Status', 'X', 'Y', 'Z', 'Source'].
    'Date' is already an absolute UTC datetime.
    """
    pos = session.pos_data
    if pos is None:
        return pd.DataFrame()

    # pos_data is always a dict in FastF1
    if not isinstance(pos, dict) or len(pos) == 0:
        return pd.DataFrame()

    frames = []
    for drv, drv_df in pos.items():
        try:
            drv_copy = drv_df.copy()
            drv_copy["driver_number"] = str(drv)
            frames.append(drv_copy)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # FastF1 Telemetry uses 'Date' (absolute UTC) not 'Time' (timedelta)
    rename = {"Date": "date", "X": "x", "Y": "y", "Z": "z", "Status": "status", "Source": "source"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")

    return _add_session_meta(df, meta)


def _extract_car_data(session: fastf1.core.Session, meta: dict) -> pd.DataFrame:
    """Extract car telemetry (speed, throttle, brake, gear, RPM).

    FastF1 car_data is a dict of {driver_number: Telemetry} where each
    Telemetry has columns: ['Date', 'RPM', 'Speed', 'nGear', 'Throttle', 'Brake', 'DRS'].
    'Date' is already an absolute UTC datetime.
    """
    car = session.car_data
    if car is None:
        return pd.DataFrame()

    if not isinstance(car, dict) or len(car) == 0:
        return pd.DataFrame()

    frames = []
    for drv, drv_df in car.items():
        try:
            drv_copy = drv_df.copy()
            drv_copy["driver_number"] = str(drv)
            frames.append(drv_copy)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # FastF1 Telemetry uses 'Date' (absolute UTC) not 'Time' (timedelta)
    rename = {
        "Date": "date", "Speed": "speed", "Throttle": "throttle",
        "Brake": "brake", "nGear": "gear", "RPM": "rpm", "DRS": "drs",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")

    return _add_session_meta(df, meta)


def _extract_results(session: fastf1.core.Session, meta: dict) -> pd.DataFrame:
    """Extract race results."""
    results = session.results
    if results is None or results.empty:
        return pd.DataFrame()

    df = results.copy().reset_index(drop=True)
    rename = {
        "DriverNumber": "driver_number",
        "Abbreviation": "driver_abbr",
        "FullName": "full_name",
        "TeamName": "team_name",
        "Position": "position",
        "ClassifiedPosition": "classified_position",
        "Points": "points",
        "Status": "status",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    return _add_session_meta(df, meta)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def ingest_session(meta: dict) -> bool:
    """
    Load a single race session via FastF1 and save bronze Parquet files.

    Args:
        meta: Session metadata dict (from get_race_sessions).

    Returns:
        True if successful, False otherwise.
    """
    session_key = meta["session_key"]
    year = meta["year"]
    meeting_name = meta["meeting_name"]
    round_num = meta.get("round_number", "?")

    bronze_dir = cfg.paths.bronze / str(year) / str(session_key)
    bronze_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already ingested
    if (bronze_dir / "race_control.parquet").exists():
        logger.info(f"Already ingested: {meeting_name} (session_key={session_key}) — skipping.")
        return True

    logger.info(f"Loading: {year} Round {round_num} — {meeting_name}")

    try:
        ff1_session = meta.get("_fastf1_session")
        if ff1_session is None:
            ff1_session = fastf1.get_session(year, round_num, "R")

        # Load all data types
        ff1_session.load(
            laps=True,
            telemetry=True,
            weather=True,
            messages=True,
            livedata=None,
        )
    except Exception as e:
        logger.error(f"Failed to load session {meeting_name}: {e}")
        return False

    # Derive date_start / date_end from FastF1 session data
    # FastF1 session.date = session start datetime
    # We derive end from laps data (last lap time) or weather data
    try:
        session_start = pd.Timestamp(ff1_session.date)
        if session_start.tzinfo is None:
            session_start = session_start.tz_localize("UTC")
        else:
            session_start = session_start.tz_convert("UTC")

        # Try to get end time from laps
        session_end = None
        if ff1_session.laps is not None and not ff1_session.laps.empty:
            last_lap_time = ff1_session.laps["Time"].dropna().max()
            if pd.notna(last_lap_time):
                session_end = session_start + last_lap_time

        # Fallback: use weather data max time
        if session_end is None and ff1_session.weather_data is not None and not ff1_session.weather_data.empty:
            last_weather = ff1_session.weather_data["Time"].dropna().max()
            if pd.notna(last_weather):
                session_end = session_start + last_weather

        # Final fallback: assume 2-hour race
        if session_end is None:
            session_end = session_start + pd.Timedelta(hours=2)

        meta["date_start"] = session_start.isoformat()
        meta["date_end"] = session_end.isoformat()
        logger.debug(f"  Session window: {meta['date_start']} → {meta['date_end']}")
    except Exception as e:
        logger.warning(f"  Could not determine session window for {meeting_name}: {e}")
        # Use a default 2-hour window starting from a placeholder
        meta["date_start"] = ""
        meta["date_end"] = ""

    # Save session metadata (with date_start/date_end)
    session_df = pd.DataFrame([{k: v for k, v in meta.items() if k != "_fastf1_session"}])
    session_df.to_parquet(bronze_dir / "session.parquet", index=False)

    # Extract and save each data type
    extractors = {
        "race_control": _extract_race_control,
        "weather": _extract_weather,
        "laps": _extract_laps,
        "position": _extract_position,
        "car_data": _extract_car_data,
        "results": _extract_results,
    }

    for name, extractor in extractors.items():
        try:
            df = extractor(ff1_session, meta)
            if not df.empty:
                out_path = bronze_dir / f"{name}.parquet"
                df.to_parquet(out_path, index=False)
                logger.debug(f"  Saved {name}.parquet: {len(df)} rows")
            else:
                logger.warning(f"  No data for {name} in {meeting_name}")
        except Exception as e:
            logger.warning(f"  Failed to extract {name} for {meeting_name}: {e}")

    logger.info(f"  ✅ {meeting_name} → {bronze_dir}")
    return True


def run_fastf1_pipeline(year: int, limit: Optional[int] = None) -> None:
    """
    Run the full FastF1 ingestion pipeline for a season.

    Args:
        year: F1 season year (2018+).
        limit: Optional max number of sessions to ingest (for quick testing).
    """
    sessions = get_race_sessions(year)

    if limit:
        sessions = sessions[:limit]
        logger.info(f"Limiting to {limit} sessions.")

    logger.info(f"Ingesting {len(sessions)} sessions for {year}...")
    success = 0
    for meta in tqdm(sessions, desc=f"FastF1 ingest {year}", unit="session"):
        if ingest_session(meta):
            success += 1

    logger.info(f"✅ Ingestion complete: {success}/{len(sessions)} sessions saved to {cfg.paths.bronze}")
