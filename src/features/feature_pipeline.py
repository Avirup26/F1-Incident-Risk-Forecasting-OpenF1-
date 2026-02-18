"""
Feature pipeline orchestration.

Loads bronze Parquet tables, builds timeline + labels, applies all feature
modules, and saves:
  - Silver: per-session feature DataFrames in data/silver/{year}/{session_key}/
  - Gold: combined master_timeline.parquet in data/gold/
"""
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.config import cfg
from src.build_timeline.timeline_builder import build_session_timeline
from src.build_timeline.label_detector import detect_sc_events
from src.build_timeline.labeler import assign_labels
from src.features.text_features import build_text_features
from src.features.weather_features import build_weather_features
from src.features.dynamics_features import build_dynamics_features
from src.utils.logger import logger


def _load_bronze(year: int, session_key: int, endpoint: str) -> pd.DataFrame:
    """Load a bronze Parquet file for a session endpoint."""
    path = cfg.paths.bronze / str(year) / str(session_key) / f"{endpoint}.parquet"
    if not path.exists():
        logger.warning(f"Bronze file not found: {path}")
        return pd.DataFrame()
    return pd.read_parquet(path)


def process_session(session: dict) -> pd.DataFrame:
    """
    Build the full feature DataFrame for a single session.

    Args:
        session: Session metadata dict.

    Returns:
        Feature DataFrame for this session, or empty DataFrame on failure.
    """
    session_key = session["session_key"]
    year = session.get("year", 2023)
    meeting_name = session.get("meeting_name", "?")

    logger.info(f"Processing features: {meeting_name} (session_key={session_key})")

    # 1. Build timeline grid
    timeline = build_session_timeline(session)
    if timeline.empty:
        logger.warning(f"Empty timeline for session {session_key} — skipping.")
        return pd.DataFrame()

    # 2. Load bronze data (FastF1 provides laps instead of intervals)
    race_control = _load_bronze(year, session_key, "race_control")
    weather = _load_bronze(year, session_key, "weather")
    position = _load_bronze(year, session_key, "position")
    laps = _load_bronze(year, session_key, "laps")

    # 3. Detect SC/VSC events and assign labels
    sc_events = detect_sc_events(race_control) if not race_control.empty else []
    timeline = assign_labels(timeline, sc_events)
    logger.info(f"  SC events: {len(sc_events)} | Positive labels: {timeline['y_sc_5m'].sum()}")

    # 4. Build features
    timeline = build_text_features(timeline, race_control)
    timeline = build_weather_features(timeline, weather)
    # Pass laps as the "intervals" proxy for dynamics (position changes from laps)
    timeline = build_dynamics_features(timeline, position, laps)

    return timeline


def run_feature_pipeline() -> None:
    """
    Run the full feature pipeline for all ingested sessions.

    Reads session metadata from bronze, processes each session, saves
    silver per-session files, and concatenates to gold master_timeline.parquet.
    """
    # Discover all sessions from bronze
    bronze_root = cfg.paths.bronze
    if not bronze_root.exists():
        logger.error(f"Bronze directory not found: {bronze_root}. Run `ingest` first.")
        return

    all_sessions = []
    for year_dir in sorted(bronze_root.iterdir()):
        if not year_dir.is_dir():
            continue
        year = int(year_dir.name)
        for session_dir in sorted(year_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            session_parquet = session_dir / "session.parquet"
            if session_parquet.exists():
                session_df = pd.read_parquet(session_parquet)
                if not session_df.empty:
                    session = session_df.iloc[0].to_dict()
                    session["year"] = year
                    all_sessions.append(session)

    if not all_sessions:
        logger.error("No sessions found in bronze. Run `ingest` first.")
        return

    logger.info(f"Building features for {len(all_sessions)} sessions...")

    gold_frames = []
    for session in tqdm(all_sessions, desc="Feature pipeline", unit="session"):
        session_key = session["session_key"]
        year = session.get("year", 2024)

        try:
            df = process_session(session)
        except Exception as e:
            logger.error(f"Failed to process session {session_key}: {e}")
            continue

        if df.empty:
            continue

        # Save silver
        silver_path = cfg.paths.silver / str(year) / str(session_key) / "features.parquet"
        silver_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(silver_path, index=False)
        logger.debug(f"Saved silver: {silver_path}")

        gold_frames.append(df)

    if not gold_frames:
        logger.error("No feature DataFrames produced. Check bronze data.")
        return

    # Save gold master timeline
    master = pd.concat(gold_frames, ignore_index=True)
    master = master.sort_values(["session_key", "timestamp"]).reset_index(drop=True)

    gold_path = cfg.paths.gold / "master_timeline.parquet"
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    master.to_parquet(gold_path, index=False)

    logger.info(
        f"✅ Gold master timeline saved: {len(master)} rows × {len(master.columns)} cols → {gold_path}"
    )
    logger.info(f"   Positive rate: {master['y_sc_5m'].mean()*100:.2f}%")
