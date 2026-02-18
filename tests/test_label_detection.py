"""
Unit tests for SC/VSC label detection.
"""
from datetime import datetime, timezone, timedelta

import pandas as pd
import pytest

from src.build_timeline.label_detector import detect_sc_events, _is_sc_start
from src.build_timeline.labeler import assign_labels


class TestIsScStart:
    def test_detects_safety_car_category(self):
        row = pd.Series({"category": "SafetyCar", "flag": "", "message": "SAFETY CAR DEPLOYED"})
        assert _is_sc_start(row) is True

    def test_detects_vsc_category(self):
        row = pd.Series({"category": "Vsc", "flag": "VSC", "message": "VIRTUAL SAFETY CAR DEPLOYED"})
        assert _is_sc_start(row) is True

    def test_excludes_ending_messages(self):
        row = pd.Series({"category": "SafetyCar", "flag": "SC", "message": "SAFETY CAR ENDING"})
        assert _is_sc_start(row) is False

    def test_excludes_clear_messages(self):
        row = pd.Series({"category": "", "flag": "", "message": "TRACK CLEAR"})
        assert _is_sc_start(row) is False

    def test_detects_text_fallback(self):
        row = pd.Series({"category": "", "flag": "", "message": "SAFETY CAR DEPLOYED"})
        assert _is_sc_start(row) is True

    def test_ignores_unrelated_messages(self):
        row = pd.Series({"category": "TrackLimits", "flag": "", "message": "TRACK LIMITS REMINDER"})
        assert _is_sc_start(row) is False


class TestDetectScEvents:
    def test_detects_two_events(self, sample_race_control):
        events = detect_sc_events(sample_race_control)
        assert len(events) == 2

    def test_events_are_sorted(self, sample_race_control):
        events = detect_sc_events(sample_race_control)
        assert events == sorted(events)

    def test_empty_dataframe_returns_empty(self):
        events = detect_sc_events(pd.DataFrame())
        assert events == []

    def test_no_sc_messages_returns_empty(self):
        df = pd.DataFrame([
            {"date": "2024-03-02T15:00:00+00:00", "message": "TRACK LIMITS", "category": "TrackLimits", "flag": ""}
        ])
        events = detect_sc_events(df)
        assert events == []

    def test_contiguous_messages_grouped_as_one_event(self):
        base = datetime(2024, 3, 2, 15, 0, 0, tzinfo=timezone.utc)
        df = pd.DataFrame([
            {"date": base, "message": "SAFETY CAR DEPLOYED", "category": "SafetyCar", "flag": "SC"},
            {"date": base + timedelta(minutes=1), "message": "SAFETY CAR PERIOD", "category": "SafetyCar", "flag": "SC"},
            {"date": base + timedelta(minutes=2), "message": "SAFETY CAR PERIOD", "category": "SafetyCar", "flag": "SC"},
        ])
        events = detect_sc_events(df)
        assert len(events) == 1  # All within 5-minute window → one event


class TestAssignLabels:
    def test_label_assigned_within_horizon(self, sample_timeline):
        base = datetime(2024, 3, 2, 15, 0, 0, tzinfo=timezone.utc)
        sc_events = [base + timedelta(minutes=3)]  # 3 min into session
        labeled = assign_labels(sample_timeline, sc_events, horizon_seconds=300)
        # Grid points at t=0, t=30s, t=1min, t=2min, t=2.5min should be labeled 1
        assert labeled["y_sc_5m"].sum() > 0

    def test_no_events_all_zeros(self, sample_timeline):
        labeled = assign_labels(sample_timeline, sc_events=[])
        assert labeled["y_sc_5m"].sum() == 0

    def test_time_to_sc_capped(self, sample_timeline):
        labeled = assign_labels(sample_timeline, sc_events=[], cap_seconds=1800)
        assert (labeled["time_to_sc_seconds"] == 1800.0).all()

    def test_time_to_sc_decreases_toward_event(self, sample_timeline):
        base = datetime(2024, 3, 2, 15, 30, 0, tzinfo=timezone.utc)  # 30 min in
        sc_events = [base]
        labeled = assign_labels(sample_timeline, sc_events)
        # time_to_sc should be monotonically decreasing up to the event
        event_ts = pd.Timestamp(base)  # already tz-aware
        before_event = labeled[labeled["timestamp"] < event_ts]
        if len(before_event) > 1:
            diffs = before_event["time_to_sc_seconds"].diff().dropna()
            assert (diffs <= 0).all()
