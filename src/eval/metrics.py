"""
Evaluation metrics for SC/VSC risk prediction.

Primary metric: PR-AUC (handles class imbalance better than ROC-AUC).
Also computes ROC-AUC, Brier score, and calibration curve data.
"""
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.calibration import calibration_curve

from src.utils.logger import logger


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "model",
) -> dict:
    """
    Compute all evaluation metrics for a binary classifier.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities for positive class.
        model_name: Name for logging.

    Returns:
        Dict of metric name → value.
    """
    metrics = {}

    # Core metrics
    metrics["pr_auc"] = average_precision_score(y_true, y_prob)
    metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    metrics["brier_score"] = brier_score_loss(y_true, y_prob)
    metrics["positive_rate"] = float(y_true.mean())
    metrics["n_samples"] = int(len(y_true))
    metrics["n_positive"] = int(y_true.sum())

    logger.info(
        f"[{model_name}] PR-AUC={metrics['pr_auc']:.4f} | "
        f"ROC-AUC={metrics['roc_auc']:.4f} | "
        f"Brier={metrics['brier_score']:.4f}"
    )
    return metrics


def compute_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (precision, recall, thresholds) for PR curve plotting."""
    return precision_recall_curve(y_true, y_prob)


def compute_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (fpr, tpr, thresholds) for ROC curve plotting."""
    return roc_curve(y_true, y_prob)


def compute_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (fraction_of_positives, mean_predicted_value) for calibration plot."""
    return calibration_curve(y_true, y_prob, n_bins=n_bins)
