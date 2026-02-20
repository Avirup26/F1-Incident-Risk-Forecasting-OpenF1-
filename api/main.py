"""
FastAPI backend for F1 Incident Risk Forecasting.
Serves all data from parquet files and trained models as JSON to the Next.js frontend.
"""
import sys
import json
from pathlib import Path
from functools import lru_cache
from typing import Optional

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root on sys.path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.config import cfg

app = FastAPI(title="F1 Risk API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Data Loaders (cached) ────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_master() -> pd.DataFrame:
    path = cfg.paths.gold / "master_timeline.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    # Normalize timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


@lru_cache(maxsize=1)
def _load_lgbm():
    try:
        from src.models.lgbm_model import LGBMModel
        model_path = cfg.paths.models / "lgbm_model.joblib"
        if model_path.exists():
            return LGBMModel.load(str(model_path))
    except Exception:
        pass
    return None


@lru_cache(maxsize=1)
def _load_feature_importance() -> pd.DataFrame:
    path = cfg.paths.models / "feature_importance.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@lru_cache(maxsize=200)
def _load_session_features(session_key: int) -> pd.DataFrame:
    master = _load_master()
    if master.empty:
        return pd.DataFrame()
    df = master[master["session_key"] == session_key].copy()
    return df


@lru_cache(maxsize=200)
def _load_race_control(year: int, session_key: int) -> pd.DataFrame:
    path = cfg.paths.bronze / str(year) / str(session_key) / "race_control.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    return df


def _clean(obj):
    """Recursively clean NaN/Inf/Timestamp for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


# ── Sessions ─────────────────────────────────────────────────────────────────

@app.get("/api/sessions")
def list_sessions(year: Optional[int] = None):
    """List all available sessions grouped by year."""
    master = _load_master()
    if master.empty:
        return []

    cols = [c for c in ["session_key", "meeting_key", "meeting_name", "year"] if c in master.columns]
    sessions = master[cols].drop_duplicates(subset=["session_key"])

    if year:
        sessions = sessions[sessions["year"] == year]

    sessions = sessions.sort_values(["year", "session_key"], ascending=[False, True])
    return _clean(sessions.to_dict(orient="records"))


@app.get("/api/sessions/{session_key}/risk")
def session_risk(session_key: int):
    """Risk timeline + SC events for a session."""
    df = _load_session_features(session_key)
    if df.empty:
        raise HTTPException(404, f"Session {session_key} not found")

    # Compute risk scores
    model = _load_lgbm()
    if model is not None:
        try:
            df = df.copy()
            df["risk_score"] = model.predict_proba(df)
        except Exception:
            df["risk_score"] = 0.0
    else:
        df["risk_score"] = df.get("y_sc_5m", pd.Series([0.0] * len(df))).astype(float)

    # Build timeline
    timeline_cols = [c for c in ["timestamp", "risk_score", "y_sc_5m"] if c in df.columns]
    timeline = df[timeline_cols].copy()
    timeline["timestamp"] = timeline["timestamp"].astype(str)
    timeline["risk_score"] = timeline["risk_score"].round(4)

    # Find SC event starts (first point of each contiguous block)
    if "y_sc_5m" in df.columns:
        ts_series = df["y_sc_5m"].fillna(0)
        event_starts = df[
            (ts_series == 1) & (ts_series.shift(1, fill_value=0) == 0)
        ]["timestamp"].astype(str).tolist()
    else:
        event_starts = []

    # Session metadata
    meta_cols = [c for c in ["meeting_name", "year", "session_key", "meeting_key"] if c in df.columns]
    meta = df[meta_cols].iloc[0].to_dict() if not df.empty else {}

    return _clean({
        "meta": meta,
        "timeline": timeline.to_dict(orient="records"),
        "sc_event_starts": event_starts,
        "peak_risk": float(df["risk_score"].max()) if "risk_score" in df.columns else 0,
        "sc_windows": int(df["y_sc_5m"].sum()) if "y_sc_5m" in df.columns else 0,
        "grid_points": len(df),
    })


@app.get("/api/sessions/{session_key}/race-control")
def session_race_control(session_key: int, year: int = Query(...)):
    """Race control messages for a session."""
    df = _load_race_control(year, session_key)
    if df.empty:
        return []

    cols = [c for c in ["date", "category", "flag", "message", "status"] if c in df.columns]
    df = df[cols].copy()
    df["date"] = df["date"].astype(str)
    df = df.sort_values("date")
    return _clean(df.to_dict(orient="records"))


# ── Stats ─────────────────────────────────────────────────────────────────────

@app.get("/api/stats/overview")
def stats_overview():
    """Global aggregate stats."""
    master = _load_master()
    if master.empty:
        return {}

    total_sessions = master["session_key"].nunique() if "session_key" in master.columns else 0
    total_rows = len(master)
    positive_rate = float(master["y_sc_5m"].mean()) if "y_sc_5m" in master.columns else 0

    sc_events_total = int(master["y_sc_5m"].sum()) if "y_sc_5m" in master.columns else 0
    years_covered = sorted(master["year"].unique().tolist()) if "year" in master.columns else []

    return _clean({
        "total_sessions": total_sessions,
        "total_grid_points": total_rows,
        "positive_rate_pct": round(positive_rate * 100, 2),
        "sc_event_windows": sc_events_total,
        "years_covered": years_covered,
        "years_count": len(years_covered),
    })


@app.get("/api/stats/sc-rate-by-year")
def sc_rate_by_year():
    """SC/VSC rate and event count by year."""
    master = _load_master()
    if master.empty or "year" not in master.columns:
        return []

    grouped = master.groupby("year").agg(
        sessions=("session_key", "nunique"),
        sc_windows=("y_sc_5m", "sum"),
        total_windows=("y_sc_5m", "count"),
    ).reset_index()
    grouped["sc_rate_pct"] = (grouped["sc_windows"] / grouped["total_windows"] * 100).round(2)
    grouped = grouped.sort_values("year")
    return _clean(grouped.to_dict(orient="records"))


@app.get("/api/stats/sc-rate-by-circuit")
def sc_rate_by_circuit():
    """SC/VSC rate per circuit (meeting_name)."""
    master = _load_master()
    if master.empty or "meeting_name" not in master.columns:
        return []

    grouped = master.groupby("meeting_name").agg(
        sessions=("session_key", "nunique"),
        sc_windows=("y_sc_5m", "sum"),
        total_windows=("y_sc_5m", "count"),
    ).reset_index()
    grouped["sc_rate_pct"] = (grouped["sc_windows"] / grouped["total_windows"] * 100).round(2)
    grouped = grouped.sort_values("sc_rate_pct", ascending=False).head(30)
    return _clean(grouped.to_dict(orient="records"))


@app.get("/api/stats/weather-by-year")
def weather_by_year():
    """Average weather stats per year."""
    master = _load_master()
    if master.empty or "year" not in master.columns:
        return []

    weather_cols = [c for c in ["air_temperature", "track_temperature", "humidity", "rainfall"] if c in master.columns]
    if not weather_cols:
        return []

    grouped = master.groupby("year")[weather_cols].mean().round(2).reset_index()
    return _clean(grouped.to_dict(orient="records"))


# ── Model ─────────────────────────────────────────────────────────────────────

@app.get("/api/model/metrics")
def model_metrics():
    """Model evaluation metrics."""
    return {
        "baseline": {
            "name": "Baseline (TF-IDF + LogReg)",
            "pr_auc": 0.0922,
            "roc_auc": 0.6254,
            "brier": 0.1373,
        },
        "lgbm": {
            "name": "LightGBM",
            "pr_auc": 0.1059,
            "roc_auc": 0.6918,
            "brier": 0.0391,
        },
        "test_set": {
            "rows": 10464,
            "meetings": 34,
            "positive_rate_pct": 3.73,
        },
    }


@app.get("/api/model/alert-policy")
def model_alert_policy():
    """Alert policy analysis at different thresholds."""
    return [
        {"threshold": 0.1, "alerts_per_race": 31.6, "lead_time_s": 119, "tpr": 0.297, "fpr": 0.095, "precision": 0.108},
        {"threshold": 0.2, "alerts_per_race": 14.2, "lead_time_s": 93,  "tpr": 0.162, "fpr": 0.042, "precision": 0.131},
        {"threshold": 0.3, "alerts_per_race": 7.7,  "lead_time_s": 91,  "tpr": 0.120, "fpr": 0.021, "precision": 0.180},
        {"threshold": 0.4, "alerts_per_race": 4.6,  "lead_time_s": 68,  "tpr": 0.090, "fpr": 0.012, "precision": 0.223},
        {"threshold": 0.5, "alerts_per_race": 3.3,  "lead_time_s": 136, "tpr": 0.067, "fpr": 0.008, "precision": 0.234},
        {"threshold": 0.6, "alerts_per_race": 2.2,  "lead_time_s": 184, "tpr": 0.046, "fpr": 0.006, "precision": 0.243},
        {"threshold": 0.7, "alerts_per_race": 1.5,  "lead_time_s": 186, "tpr": 0.038, "fpr": 0.004, "precision": 0.288},
    ]


@app.get("/api/model/feature-importance")
def feature_importance(top_n: int = 20):
    """Top N feature importances."""
    fi = _load_feature_importance()
    if fi.empty:
        return []
    top = fi.head(top_n)
    return _clean(top.to_dict(orient="records"))


@app.get("/api/health")
def health():
    return {"status": "ok"}
