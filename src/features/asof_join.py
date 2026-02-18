"""
Generic as-of join utility.

An as-of join merges two DataFrames such that for each row in the left
DataFrame (at time t), we find the most recent row in the right DataFrame
with timestamp <= t. This prevents any future data leakage.
"""
import pandas as pd

from src.utils.logger import logger


def asof_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str = "timestamp",
    by: str | list[str] | None = None,
    tolerance: str | None = None,
    direction: str = "backward",
) -> pd.DataFrame:
    """
    Perform an as-of (last-known-value) join between two DataFrames.

    Args:
        left: Left DataFrame (timeline grid). Must be sorted by `on`.
        right: Right DataFrame (feature source). Must be sorted by `on`.
        on: Column name for the timestamp merge key.
        by: Optional grouping key(s) for per-group as-of joins
            (e.g. 'session_key').
        tolerance: Optional max time distance (e.g. '10min'). If the
            nearest right timestamp is farther than this, the row is NaN.
        direction: 'backward' (default) = last value <= t.

    Returns:
        Left DataFrame with right columns merged in.
    """
    # Ensure both are sorted
    left = left.sort_values(on).reset_index(drop=True)
    right = right.sort_values(on).reset_index(drop=True)

    # Validate timestamp dtype compatibility
    if not pd.api.types.is_datetime64_any_dtype(left[on]):
        raise ValueError(f"left['{on}'] must be datetime dtype.")
    if not pd.api.types.is_datetime64_any_dtype(right[on]):
        raise ValueError(f"right['{on}'] must be datetime dtype.")

    # Drop shared columns from right (except the merge key and by-columns)
    # to prevent pd.merge_asof from creating _x/_y suffixes or dropping them.
    by_cols = [by] if isinstance(by, str) else (list(by) if by else [])
    keep_in_right = {on} | set(by_cols)
    shared_cols = (set(left.columns) & set(right.columns)) - keep_in_right
    if shared_cols:
        right = right.drop(columns=list(shared_cols))

    kwargs: dict = {"on": on, "direction": direction}
    if by is not None:
        kwargs["by"] = by
    if tolerance is not None:
        kwargs["tolerance"] = pd.Timedelta(tolerance)

    merged = pd.merge_asof(left, right, **kwargs)

    right_only_cols = [c for c in merged.columns if c not in left.columns]
    n_matched = merged[right_only_cols].notna().any(axis=1).sum() if right_only_cols else 0
    logger.debug(f"as-of join: {n_matched}/{len(merged)} rows matched.")
    return merged


def asof_join_by_session(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str = "timestamp",
    session_col: str = "session_key",
    tolerance: str | None = None,
) -> pd.DataFrame:
    """
    Perform per-session as-of joins and concatenate results.

    Splits by session_key, runs asof_join for each, then recombines.
    This avoids cross-session leakage.

    Args:
        left: Timeline grid DataFrame.
        right: Feature source DataFrame.
        on: Timestamp column name.
        session_col: Column to group by (default 'session_key').
        tolerance: Optional max time distance.

    Returns:
        Merged DataFrame with all sessions combined.
    """
    sessions = left[session_col].unique()
    frames = []

    for session_key in sessions:
        left_s = left[left[session_col] == session_key].copy()
        right_s = right[right[session_col] == session_key].copy() if session_col in right.columns else right.copy()

        if right_s.empty:
            logger.warning(f"No right-side data for session {session_key}.")
            frames.append(left_s)
            continue

        merged = asof_join(left_s, right_s, on=on, tolerance=tolerance)
        frames.append(merged)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
