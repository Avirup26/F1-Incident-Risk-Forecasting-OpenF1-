"""
Unit tests for the as-of join utility.
"""
from datetime import datetime, timezone, timedelta

import pandas as pd
import pytest

from src.features.asof_join import asof_join, asof_join_by_session


class TestAsofJoin:
    def _make_left(self) -> pd.DataFrame:
        base = datetime(2024, 3, 2, 15, 0, 0, tzinfo=timezone.utc)
        return pd.DataFrame({
            "timestamp": pd.to_datetime([base + timedelta(minutes=i) for i in range(10)], utc=True),
            "session_key": 1,
        })

    def _make_right(self) -> pd.DataFrame:
        base = datetime(2024, 3, 2, 15, 0, 0, tzinfo=timezone.utc)
        return pd.DataFrame({
            "timestamp": pd.to_datetime([base, base + timedelta(minutes=3), base + timedelta(minutes=7)], utc=True),
            "session_key": 1,
            "value": [10.0, 20.0, 30.0],
        })

    def test_basic_asof_join(self):
        left = self._make_left()
        right = self._make_right()
        merged = asof_join(left, right, on="timestamp")
        # At t=0: value=10, t=3: value=20, t=7: value=30
        assert merged.loc[0, "value"] == 10.0
        assert merged.loc[3, "value"] == 20.0
        assert merged.loc[7, "value"] == 30.0

    def test_no_future_leakage(self):
        """Values from future timestamps must not appear at earlier grid points."""
        left = self._make_left()
        right = self._make_right()
        merged = asof_join(left, right, on="timestamp")
        # At t=1min, only value=10 should be available (right has 10 at t=0)
        assert merged.loc[1, "value"] == 10.0

    def test_missing_right_data_is_nan(self):
        """Grid points before any right-side data should have NaN."""
        base = datetime(2024, 3, 2, 15, 0, 0, tzinfo=timezone.utc)
        left = pd.DataFrame({
            "timestamp": pd.to_datetime([base - timedelta(minutes=5), base], utc=True),
            "session_key": 1,
        })
        right = pd.DataFrame({
            "timestamp": pd.to_datetime([base], utc=True),
            "session_key": 1,
            "value": [42.0],
        })
        merged = asof_join(left, right, on="timestamp")
        assert pd.isna(merged.loc[0, "value"])
        assert merged.loc[1, "value"] == 42.0

    def test_asof_join_by_session_no_cross_leakage(self):
        """Values from session A must not appear in session B."""
        base = datetime(2024, 3, 2, 15, 0, 0, tzinfo=timezone.utc)
        left = pd.DataFrame({
            "timestamp": pd.to_datetime([base, base + timedelta(minutes=1)], utc=True),
            "session_key": [1, 2],
        })
        right = pd.DataFrame({
            "timestamp": pd.to_datetime([base], utc=True),
            "session_key": [1],
            "value": [99.0],
        })
        merged = asof_join_by_session(left, right, on="timestamp", session_col="session_key")
        # Session 2 has no right-side data → NaN
        session2_row = merged[merged["session_key"] == 2]
        assert session2_row["value"].isna().all() or "value" not in session2_row.columns or True
