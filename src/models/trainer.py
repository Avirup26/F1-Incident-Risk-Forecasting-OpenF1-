"""
Model training pipeline.

Loads the gold master timeline, splits by meeting_key, trains both
baseline and LightGBM models, and saves them with metadata.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import cfg
from src.models.splitter import temporal_train_test_split
from src.models.baseline_model import BaselineModel
from src.models.lgbm_model import LGBMModel
from src.utils.logger import logger


def run_training() -> None:
    """
    Full training pipeline:
    1. Load gold master_timeline.parquet
    2. Temporal train/test split by meeting_key
    3. Train baseline (TF-IDF + LR) and LightGBM models
    4. Save models + metadata to data/models/
    """
    gold_path = cfg.paths.gold / "master_timeline.parquet"
    if not gold_path.exists():
        logger.error(f"Gold timeline not found: {gold_path}. Run `build_features` first.")
        return

    logger.info(f"Loading gold timeline from {gold_path}...")
    df = pd.read_parquet(gold_path)
    logger.info(f"Loaded {len(df)} rows × {len(df.columns)} columns.")
    logger.info(f"Positive rate: {df[cfg.features.label_column].mean()*100:.2f}%")

    # Temporal split
    train_df, test_df = temporal_train_test_split(df, test_fraction=0.2)

    # Further split train into train/val for LightGBM early stopping
    train_inner, val_df = temporal_train_test_split(train_df, test_fraction=0.15)

    models_dir = cfg.paths.models
    models_dir.mkdir(parents=True, exist_ok=True)

    # --- Baseline ---
    logger.info("Training baseline model (TF-IDF + Logistic Regression)...")
    baseline = BaselineModel()
    baseline.fit(train_df)
    baseline_path = models_dir / "baseline_model.joblib"
    baseline.save(str(baseline_path))

    # --- LightGBM ---
    logger.info("Training LightGBM model...")
    lgbm = LGBMModel()
    lgbm.fit(train_inner, val_df=val_df)
    lgbm_path = models_dir / "lgbm_model.joblib"
    lgbm.save(str(lgbm_path))

    # Save feature importance
    fi = lgbm.feature_importance()
    if not fi.empty:
        fi.to_csv(models_dir / "feature_importance.csv", index=False)
        logger.info(f"Top features: {fi.head(5)['feature'].tolist()}")

    # Save metadata
    metadata = {
        "trained_at": datetime.now(tz=timezone.utc).isoformat(),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_meetings": int(train_df["meeting_key"].nunique()),
        "test_meetings": int(test_df["meeting_key"].nunique()),
        "positive_rate_train": float(train_df[cfg.features.label_column].mean()),
        "positive_rate_test": float(test_df[cfg.features.label_column].mean()),
        "label_column": cfg.features.label_column,
        "prediction_horizon_seconds": cfg.features.prediction_horizon_seconds,
        "no_leakage": "Splits are by meeting_key (race weekend). No random splits used.",
    }
    with open(models_dir / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save test set for evaluation
    test_df.to_parquet(models_dir / "test_set.parquet", index=False)

    logger.info(f"✅ Training complete. Models saved to {models_dir}")
    logger.info(f"   Baseline: {baseline_path.name}")
    logger.info(f"   LightGBM: {lgbm_path.name}")
