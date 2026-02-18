"""
Text features extracted from race_control messages.

All features use strict as-of semantics: only messages with timestamp <= t
are used when computing features for grid point t.

Features:
  - recent_messages_concat: raw text for TF-IDF
  - msg_count_60s, msg_count_180s, msg_count_600s: rolling message counts
  - category_entropy_180s: Shannon entropy of message categories
  - unique_categories_180s: number of distinct categories
  - Keyword flags: debris, crash, stopped, rain, yellow, red, track_limits, investigation
"""
import re
from math import log2

import numpy as np
import pandas as pd

from src.config import cfg
from src.features.rolling_utils import rolling_count, rolling_unique_count
from src.utils.logger import logger
from src.utils.time_utils import parse_timestamp_series

# Keyword patterns (applied to last 180s of messages)
KEYWORD_FLAGS = {
    "debris_flag": re.compile(r"\bDEBRIS\b", re.IGNORECASE),
    "crash_flag": re.compile(r"\b(CRASH|ACCIDENT|COLLISION|INCIDENT)\b", re.IGNORECASE),
    "stopped_flag": re.compile(r"\bSTOPPED\b", re.IGNORECASE),
    "rain_flag": re.compile(r"\b(RAIN|WET|SLIPPERY|AQUAPLANING)\b", re.IGNORECASE),
    "yellow_flag": re.compile(r"\bYELLOW\b", re.IGNORECASE),
    "red_flag": re.compile(r"\bRED FLAG\b", re.IGNORECASE),
    "track_limits_flag": re.compile(r"\bTRACK LIMITS\b", re.IGNORECASE),
    "investigation_flag": re.compile(r"\bINVESTIGATION\b", re.IGNORECASE),
}


def _shannon_entropy(values: list[str]) -> float:
    """Compute Shannon entropy of a list of category strings."""
    if not values:
        return 0.0
    counts: dict[str, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    total = len(values)
    return -sum((c / total) * log2(c / total) for c in counts.values())


def build_text_features(
    timeline_df: pd.DataFrame,
    race_control_df: pd.DataFrame,
    session_col: str = "session_key",
) -> pd.DataFrame:
    """
    Compute text-based features for each timeline grid point.

    Args:
        timeline_df: Timeline grid with 'timestamp' and 'session_key'.
        race_control_df: Race control messages with 'date', 'message',
            'category', 'session_key'.

    Returns:
        timeline_df with text feature columns added.
    """
    result_frames = []
    sessions = timeline_df[session_col].unique()

    for session_key in sessions:
        tl = timeline_df[timeline_df[session_col] == session_key].copy()
        rc = race_control_df[race_control_df[session_col] == session_key].copy()

        if rc.empty:
            logger.warning(f"No race_control data for session {session_key}.")
            for col in list(KEYWORD_FLAGS.keys()) + [
                "recent_messages_concat", "msg_count_60s", "msg_count_180s",
                "msg_count_600s", "category_entropy_180s", "unique_categories_180s"
            ]:
                tl[col] = 0 if col.endswith("_flag") or col.startswith("msg_count") else ""
            result_frames.append(tl)
            continue

        rc["date"] = parse_timestamp_series(rc["date"])
        rc = rc.sort_values("date").reset_index(drop=True)

        tl_ts = tl["timestamp"]
        rc_ts = rc["date"]
        rc_ts_arr = np.asarray(rc_ts, dtype="datetime64[ns]")
        messages = rc["message"].fillna("").values
        categories = rc["category"].fillna("").values

        # Rolling message counts
        tl["msg_count_60s"] = rolling_count(tl_ts, rc_ts, cfg.features.short_window)
        tl["msg_count_180s"] = rolling_count(tl_ts, rc_ts, cfg.features.medium_window)
        tl["msg_count_600s"] = rolling_count(tl_ts, rc_ts, cfg.features.long_window)

        # Per-row text features (180s window)
        window_ns = np.timedelta64(cfg.features.medium_window, "s")
        tl_ts_arr = np.asarray(tl_ts, dtype="datetime64[ns]")

        recent_msgs_list = []
        entropy_list = []
        unique_cat_list = []
        flag_arrays = {k: np.zeros(len(tl), dtype=int) for k in KEYWORD_FLAGS}

        for i, t in enumerate(tl_ts_arr):
            lo = t - window_ns
            mask = (rc_ts_arr > lo) & (rc_ts_arr <= t)
            window_msgs = messages[mask]
            window_cats = categories[mask]

            # Concatenated text
            recent_msgs_list.append(" | ".join(window_msgs))

            # Category entropy
            entropy_list.append(_shannon_entropy(list(window_cats)))
            unique_cat_list.append(len(set(window_cats)))

            # Keyword flags
            combined_text = " ".join(window_msgs)
            for flag_name, pattern in KEYWORD_FLAGS.items():
                if pattern.search(combined_text):
                    flag_arrays[flag_name][i] = 1

        tl["recent_messages_concat"] = recent_msgs_list
        tl["category_entropy_180s"] = entropy_list
        tl["unique_categories_180s"] = unique_cat_list
        for flag_name, arr in flag_arrays.items():
            tl[flag_name] = arr

        result_frames.append(tl)

    return pd.concat(result_frames, ignore_index=True) if result_frames else timeline_df
