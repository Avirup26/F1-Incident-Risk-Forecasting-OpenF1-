"""
Baseline model: TF-IDF + Logistic Regression.

Uses race_control message text (TF-IDF) combined with numeric features.
Logistic Regression with class weighting handles the class imbalance.
"""
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import hstack, csr_matrix

from src.config import cfg
from src.utils.logger import logger

NUMERIC_FEATURES = [
    "msg_count_60s", "msg_count_180s", "msg_count_600s",
    "category_entropy_180s", "unique_categories_180s",
    "debris_flag", "crash_flag", "stopped_flag", "rain_flag",
    "yellow_flag", "red_flag", "track_limits_flag", "investigation_flag",
    "rainfall", "track_temperature", "air_temperature", "wind_speed",
    "humidity", "pressure", "max_rainfall_5m", "track_temp_delta_5m",
    "position_changes_120s", "driver_position_volatility_300s",
    "gap_std_120s", "pack_density_1_5s", "pack_density_3_0s",
]


class BaselineModel:
    """
    TF-IDF + Logistic Regression baseline for SC/VSC prediction.
    """

    def __init__(self) -> None:
        self.tfidf = TfidfVectorizer(
            max_features=cfg.model.tfidf_max_features,
            ngram_range=(cfg.model.tfidf_ngram_min, cfg.model.tfidf_ngram_max),
            sublinear_tf=True,
        )
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(
            class_weight=cfg.model.class_weight,
            max_iter=1000,
            random_state=cfg.model.random_state,
            solver="lbfgs",
        )
        self._numeric_cols: list[str] = []
        self.is_fitted = False

    def _get_numeric(self, df: pd.DataFrame) -> np.ndarray:
        """Extract available numeric features, filling missing with 0."""
        cols = [c for c in NUMERIC_FEATURES if c in df.columns]
        self._numeric_cols = cols
        X_num = df[cols].fillna(0).values.astype(float)
        return X_num

    def fit(self, train_df: pd.DataFrame) -> "BaselineModel":
        """
        Fit the baseline model.

        Args:
            train_df: Training DataFrame with feature columns and 'y_sc_5m'.

        Returns:
            self
        """
        y = train_df[cfg.features.label_column].values
        text = train_df["recent_messages_concat"].fillna("").values

        X_tfidf = self.tfidf.fit_transform(text)
        X_num = self.scaler.fit_transform(self._get_numeric(train_df))
        X = hstack([X_tfidf, csr_matrix(X_num)])

        logger.info(f"Fitting baseline model: {X.shape[0]} samples, {X.shape[1]} features.")
        self.clf.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return probability of positive class."""
        text = df["recent_messages_concat"].fillna("").values
        X_tfidf = self.tfidf.transform(text)
        cols = [c for c in self._numeric_cols if c in df.columns]
        X_num = self.scaler.transform(df[cols].fillna(0).values.astype(float))
        X = hstack([X_tfidf, csr_matrix(X_num)])
        return self.clf.predict_proba(X)[:, 1]

    def save(self, path: str) -> None:
        """Serialize model to disk."""
        joblib.dump(self, path)
        logger.info(f"Baseline model saved → {path}")

    @classmethod
    def load(cls, path: str) -> "BaselineModel":
        """Load model from disk."""
        return joblib.load(path)
