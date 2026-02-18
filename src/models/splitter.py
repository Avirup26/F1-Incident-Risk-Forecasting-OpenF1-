"""
Time-series safe train/test splits.

Uses GroupKFold by meeting_key to ensure no race weekend appears in both
train and test sets. This prevents data leakage from temporal correlation.
"""
from typing import Generator, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.utils.logger import logger


def get_meeting_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    meeting_col: str = "meeting_key",
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Generate train/test index splits grouped by meeting_key.

    Each split ensures all rows from a given race weekend are either
    entirely in train or entirely in test — never split across both.

    Args:
        df: Master timeline DataFrame.
        n_splits: Number of cross-validation folds.
        meeting_col: Column to group by (default 'meeting_key').

    Returns:
        List of (train_indices, test_indices) tuples.
    """
    groups = df[meeting_col].values
    unique_meetings = np.unique(groups)
    n_splits = min(n_splits, len(unique_meetings))

    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(df, groups=groups))

    logger.info(
        f"Created {n_splits} GroupKFold splits over {len(unique_meetings)} meetings."
    )
    return splits


def temporal_train_test_split(
    df: pd.DataFrame,
    test_meetings: Optional[list] = None,
    test_fraction: float = 0.2,
    meeting_col: str = "meeting_key",
    time_col: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data temporally: last N meetings go to test set.

    Args:
        df: Master timeline DataFrame.
        test_meetings: Explicit list of meeting_keys for test set.
            If None, uses the last `test_fraction` of meetings chronologically.
        test_fraction: Fraction of meetings to use as test (if test_meetings is None).
        meeting_col: Column identifying race weekends.
        time_col: Timestamp column for ordering meetings.

    Returns:
        (train_df, test_df) tuple.
    """
    if test_meetings is not None:
        test_mask = df[meeting_col].isin(test_meetings)
        train_df = df[~test_mask].copy()
        test_df = df[test_mask].copy()
    else:
        # Order meetings by their earliest timestamp
        meeting_order = (
            df.groupby(meeting_col)[time_col].min().sort_values().index.tolist()
        )
        n_test = max(1, int(len(meeting_order) * test_fraction))
        test_meetings_auto = meeting_order[-n_test:]
        test_mask = df[meeting_col].isin(test_meetings_auto)
        train_df = df[~test_mask].copy()
        test_df = df[test_mask].copy()

    logger.info(
        f"Train: {len(train_df)} rows ({train_df[meeting_col].nunique()} meetings) | "
        f"Test: {len(test_df)} rows ({test_df[meeting_col].nunique()} meetings)"
    )
    return train_df, test_df
