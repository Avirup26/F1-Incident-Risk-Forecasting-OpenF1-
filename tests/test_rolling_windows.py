"""
Unit tests for rolling window utilities.
"""
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import pytest

from src.features.rolling_utils import rolling_count, rolling_agg, rolling_unique_count


class TestRollingCount:
    def _make_series(self, offsets_minutes: list[int]) -> pd.Series:
        base = datetime(2024, 3, 2, 15, 0, 0, tzinfo=timezone.utc)
        return pd.Series(
            pd.to_datetime([base + timedelta(minutes=m) for m in offsets_minutes], utc=True)
        )

    def test_counts_events_in_window(self):
        base = datetime(2024, 3, 2, 15, 0, 0, tzinfo=timezone.utc)
        timeline_ts = pd.Series(
            pd.to_datetime([base + timedelta(minutes=i) for i in range(10)], utc=True)
        )
        event_ts = self._make_series([1, 2, 3])  # 3 events in first 3 minutes

        counts = rolling_count(timeline_ts, event_ts, window_seconds=300)  # 5-min window
        # At t=5min, all 3 events should be in window
        assert counts[5] == 3

    def test_no_events_returns_zeros(self):
        base = datetime(2024, 3, 2, 15, 0, 0, tzinfo=timezone.utc)
        timeline_ts = pd.Series(
            pd.to_datetime([base + timedelta(minutes=i) for i in range(5)], utc=True)
        )
        event_ts = pd.Series(pd.DatetimeIndex([]))
        counts = rolling_count(timeline_ts, event_ts, window_seconds=60)
        assert (counts == 0).all()

    def test_window_boundary_exclusive_start(self):
        """Event exactly at t - window should NOT be counted."""
        base = datetime(2024, 3, 2, 15, 0, 0, tzinfo=timezone.utc)
        timeline_ts = pd.Series(pd.to_datetime([base + timedelta(minutes=1)], utc=True))
        # Event at exactly t - 60s (boundary)
        event_ts = pd.Series(pd.to_datetime([base], utc=True))
        counts = rolling_count(timeline_ts, event_ts, window_seconds=60)
        # (base, base+60s] → base is excluded (strictly greater than lo)
        assert counts[0] == 0

    def test_window_boundary_inclusive_end(self):
        """Event exactly at t should be counted."""
        base = datetime(2024, 3, 2, 15, 0, 0, tzinfo=timezone.utc)
        t = base + timedelta(minutes=1)
        timeline_ts = pd.Series(pd.to_datetime([t], utc=True))
        event_ts = pd.Series(pd.to_datetime([t], utc=True))
        counts = rolling_count(timeline_ts, event_ts, window_seconds=60)
        assert counts[0] == 1


class TestRollingAgg:
    def test_mean_aggregation(self):
        base = datetime(2024, 3, 2, 15, 0, 0, tzinfo=timezone.utc)
        timeline_ts = pd.Series(
            pd.to_datetime([base + timedelta(minutes=5)], utc=True)
        )
        event_df = pd.DataFrame({
            "ts": pd.to_datetime([base + timedelta(minutes=i) for i in range(5)], utc=True),
            "val": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        result = rolling_agg(timeline_ts, event_df, "ts", "val", window_seconds=300)
        # Window (t-300s, t] = (0min, 5min] → events at 1,2,3,4min (0min excluded)
        # Values: 20, 30, 40, 50 → mean = 35.0
        assert abs(result[0] - 35.0) < 1e-6

    def test_max_aggregation(self):
        base = datetime(2024, 3, 2, 15, 0, 0, tzinfo=timezone.utc)
        timeline_ts = pd.Series(pd.to_datetime([base + timedelta(minutes=3)], utc=True))
        event_df = pd.DataFrame({
            "ts": pd.to_datetime([base + timedelta(minutes=i) for i in range(3)], utc=True),
            "val": [5.0, 15.0, 25.0],
        })
        result = rolling_agg(timeline_ts, event_df, "ts", "val", window_seconds=300, agg_fn="max")
        assert result[0] == 25.0

    def test_empty_window_returns_nan(self):
        base = datetime(2024, 3, 2, 15, 0, 0, tzinfo=timezone.utc)
        timeline_ts = pd.Series(pd.to_datetime([base], utc=True))
        event_df = pd.DataFrame({
            "ts": pd.to_datetime([base + timedelta(hours=1)], utc=True),
            "val": [99.0],
        })
        result = rolling_agg(timeline_ts, event_df, "ts", "val", window_seconds=60)
        assert np.isnan(result[0])
