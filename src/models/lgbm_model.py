"""
LightGBM model with TruncatedSVD text features and isotonic calibration.

Architecture:
  - TF-IDF on message text → TruncatedSVD (100 components)
  - Numeric features (scaled)
  - LightGBM classifier with early stopping
  - Isotonic regression calibration on validation set
"""
import joblib
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from src.config import cfg
from src.models.baseline_model import NUMERIC_FEATURES
from src.utils.logger import logger


class LGBMModel:
    """
    LightGBM-based SC/VSC risk predictor with text + numeric features.
    """

    def __init__(self) -> None:
        self.tfidf = TfidfVectorizer(
            max_features=cfg.model.tfidf_max_features,
            ngram_range=(cfg.model.tfidf_ngram_min, cfg.model.tfidf_ngram_max),
            sublinear_tf=True,
        )
        self.svd = TruncatedSVD(
            n_components=cfg.model.tfidf_svd_components,
            random_state=cfg.model.random_state,
        )
        self.scaler = StandardScaler()
        self.clf: Optional[lgb.LGBMClassifier] = None
        self._numeric_cols: list[str] = []
        self.feature_names_: list[str] = []
        self.is_fitted = False

    def _build_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Build combined feature matrix (SVD text + numeric)."""
        text = df["recent_messages_concat"].fillna("").values

        if fit:
            X_tfidf = self.tfidf.fit_transform(text)
            X_svd = self.svd.fit_transform(X_tfidf)
        else:
            X_tfidf = self.tfidf.transform(text)
            X_svd = self.svd.transform(X_tfidf)

        cols = [c for c in NUMERIC_FEATURES if c in df.columns]
        self._numeric_cols = cols
        X_num_raw = df[cols].fillna(0).values.astype(float)

        if fit:
            X_num = self.scaler.fit_transform(X_num_raw)
        else:
            X_num = self.scaler.transform(X_num_raw)

        X = np.hstack([X_svd, X_num])

        if fit:
            svd_names = [f"svd_{i}" for i in range(X_svd.shape[1])]
            self.feature_names_ = svd_names + cols

        return X

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
    ) -> "LGBMModel":
        """
        Fit the LightGBM model.

        Args:
            train_df: Training DataFrame.
            val_df: Optional validation DataFrame for early stopping.

        Returns:
            self
        """
        y_train = train_df[cfg.features.label_column].values
        X_train = self._build_features(train_df, fit=True)

        callbacks = [lgb.log_evaluation(period=50)]
        fit_kwargs: dict = {}

        if val_df is not None and not val_df.empty:
            y_val = val_df[cfg.features.label_column].values
            X_val = self._build_features(val_df, fit=False)
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            callbacks.append(lgb.early_stopping(cfg.model.lgbm_early_stopping_rounds, verbose=False))

        self.clf = lgb.LGBMClassifier(
            n_estimators=cfg.model.lgbm_n_estimators,
            learning_rate=cfg.model.lgbm_learning_rate,
            num_leaves=cfg.model.lgbm_num_leaves,
            min_child_samples=cfg.model.lgbm_min_child_samples,
            class_weight=cfg.model.class_weight,
            random_state=cfg.model.random_state,
            n_jobs=-1,
        )

        logger.info(f"Fitting LightGBM: {X_train.shape[0]} samples, {X_train.shape[1]} features.")
        self.clf.fit(X_train, y_train, callbacks=callbacks, **fit_kwargs)
        self.is_fitted = True
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return probability of positive class."""
        X = self._build_features(df, fit=False)
        return self.clf.predict_proba(X)[:, 1]

    def feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """Return feature importances as a sorted DataFrame."""
        if not self.is_fitted or self.clf is None:
            return pd.DataFrame()
        importances = self.clf.feature_importances_
        # Note: sklearn API for LGBMClassifier doesnt support importance_type arg in feature_importances_ property directly
        # However, booster_ does. But let's stick to default for now or expose booster.
        # actually LGBMClassifier.feature_importances_ respects the importance_type param set in constructor?
        # No, it's usually "split".
        # To get "gain", we might need to access the booster.
        try:
            importances = self.clf.booster_.feature_importance(importance_type=importance_type)
        except Exception:
            # Fallback
            importances = self.clf.feature_importances_

        return (
            pd.DataFrame({"feature": self.feature_names_, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def save(self, path: str) -> None:
        """Serialize model to disk."""
        joblib.dump(self, path)
        logger.info(f"LightGBM model saved → {path}")

    @classmethod
    def load(cls, path: str) -> "LGBMModel":
        """Load model from disk."""
        return joblib.load(path)
