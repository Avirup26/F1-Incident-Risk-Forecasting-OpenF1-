"""
Pytest fixtures for F1 Risk Forecasting tests.
"""
from datetime import datetime, timezone, timedelta

import pandas as pd
import pytest


@pytest.fixture
def sample_race_control() -> pd.DataFrame:
    """Synthetic race_control messages with known SC events."""
    base = datetime(2024, 3, 2, 15, 0, 0, tzinfo=timezone.utc)
    return pd.DataFrame([
        {"date": base + timedelta(minutes=10), "message": "SAFETY CAR DEPLOYED", "category": "SafetyCar", "flag": "SC"},
        {"date": base + timedelta(minutes=11), "message": "SAFETY CAR PERIOD", "category": "SafetyCar", "flag": "SC"},
        {"date": base + timedelta(minutes=25), "message": "TRACK LIMITS REMINDER", "category": "TrackLimits", "flag": ""},
        {"date": base + timedelta(minutes=40), "message": "VIRTUAL SAFETY CAR DEPLOYED", "category": "Vsc", "flag": "VSC"},
        {"date": base + timedelta(minutes=41), "message": "VIRTUAL SAFETY CAR ENDING", "category": "Vsc", "flag": "VSC"},
    ])


@pytest.fixture
def sample_timeline() -> pd.DataFrame:
    """30-second grid for a 1-hour session."""
    base = datetime(2024, 3, 2, 15, 0, 0, tzinfo=timezone.utc)
    timestamps = pd.date_range(start=base, periods=120, freq="30s", tz="UTC")
    return pd.DataFrame({
        "timestamp": timestamps,
        "session_key": 9158,
        "meeting_key": 1229,
        "year": 2024,
        "meeting_name": "Test GP",
    })


@pytest.fixture
def sample_weather() -> pd.DataFrame:
    """Synthetic weather readings."""
    base = datetime(2024, 3, 2, 15, 0, 0, tzinfo=timezone.utc)
    return pd.DataFrame([
        {"date": base + timedelta(minutes=i), "rainfall": 0.0, "track_temperature": 35.0 + i * 0.1,
         "air_temperature": 25.0, "wind_speed": 5.0, "humidity": 60.0, "pressure": 1013.0,
         "session_key": 9158}
        for i in range(60)
    ])
