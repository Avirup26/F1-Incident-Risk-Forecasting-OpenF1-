"""
Evaluation report generator.

Loads trained models and test set, computes all metrics, runs alert
policy analysis, and writes a markdown report to data/models/.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import cfg
from src.eval.metrics import compute_metrics, compute_pr_curve, compute_calibration
from src.eval.alert_analysis import analyze_alert_policy
from src.models.baseline_model import BaselineModel
from src.models.lgbm_model import LGBMModel
from src.utils.logger import logger


def run_evaluation() -> None:
    """
    Full evaluation pipeline:
    1. Load test set and trained models
    2. Compute PR-AUC, ROC-AUC, Brier score for both models
    3. Run alert policy analysis
    4. Write markdown report
    """
    models_dir = cfg.paths.models
    test_path = models_dir / "test_set.parquet"

    if not test_path.exists():
        logger.error(f"Test set not found: {test_path}. Run `train` first.")
        return

    logger.info("Loading test set and models...")
    test_df = pd.read_parquet(test_path)
    y_true = test_df[cfg.features.label_column].values

    results = {}

    # Load and evaluate baseline
    baseline_path = models_dir / "baseline_model.joblib"
    if baseline_path.exists():
        baseline = BaselineModel.load(str(baseline_path))
        y_prob_baseline = baseline.predict_proba(test_df)
        results["Baseline (TF-IDF + LR)"] = {
            "metrics": compute_metrics(y_true, y_prob_baseline, "Baseline"),
            "y_prob": y_prob_baseline,
        }
    else:
        logger.warning("Baseline model not found.")

    # Load and evaluate LightGBM
    lgbm_path = models_dir / "lgbm_model.joblib"
    if lgbm_path.exists():
        lgbm = LGBMModel.load(str(lgbm_path))
        y_prob_lgbm = lgbm.predict_proba(test_df)
        results["LightGBM"] = {
            "metrics": compute_metrics(y_true, y_prob_lgbm, "LightGBM"),
            "y_prob": y_prob_lgbm,
        }
    else:
        logger.warning("LightGBM model not found.")

    if not results:
        logger.error("No models found to evaluate.")
        return

    # Alert policy analysis (use best model = LightGBM if available)
    best_key = "LightGBM" if "LightGBM" in results else list(results.keys())[0]
    alert_df = analyze_alert_policy(test_df, results[best_key]["y_prob"])

    # Write markdown report
    report_path = models_dir / "evaluation_report.md"
    _write_report(results, alert_df, test_df, report_path)
    logger.info(f"✅ Evaluation report saved → {report_path}")


def _write_report(
    results: dict,
    alert_df: pd.DataFrame,
    test_df: pd.DataFrame,
    path: Path,
) -> None:
    """Write a markdown evaluation report."""
    lines = [
        "# F1 Incident Risk Forecasting — Evaluation Report\n",
        f"**Test set**: {len(test_df)} rows | "
        f"{test_df['meeting_key'].nunique()} race weekends | "
        f"Positive rate: {test_df[cfg.features.label_column].mean()*100:.2f}%\n",
        "---\n",
        "## Model Performance\n",
        "| Model | PR-AUC | ROC-AUC | Brier Score |",
        "|-------|--------|---------|-------------|",
    ]

    for model_name, data in results.items():
        m = data["metrics"]
        lines.append(
            f"| {model_name} | {m['pr_auc']:.4f} | {m['roc_auc']:.4f} | {m['brier_score']:.4f} |"
        )

    lines += [
        "\n---\n",
        "## Alert Policy Analysis (LightGBM)\n",
        "| Threshold | Alerts/Race | Median Lead Time (s) | TPR | FPR | Precision |",
        "|-----------|-------------|----------------------|-----|-----|-----------|",
    ]

    for _, row in alert_df.iterrows():
        lead = f"{row['median_lead_time_s']:.0f}" if row["median_lead_time_s"] else "N/A"
        lines.append(
            f"| {row['threshold']:.1f} | {row['alerts_per_race']:.1f} | {lead} | "
            f"{row['true_positive_rate']:.3f} | {row['false_positive_rate']:.3f} | "
            f"{row['precision']:.3f} |"
        )

    lines += [
        "\n---\n",
        "## No-Leakage Statement\n",
        "All features are computed with strict as-of semantics: only data with "
        "timestamp ≤ t is used when computing features for grid point t. "
        "Train/test splits are by `meeting_key` (race weekend) — no race weekend "
        "appears in both train and test sets. No random splits are used.\n",
        "---\n",
        "## Limitations\n",
        "- SC/VSC events are rare (~5-15% of grid points), making this an imbalanced classification problem.\n",
        "- Model performance depends on data quality from the OpenF1 API.\n",
        "- Race control messages may have variable latency in real-time scenarios.\n",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))
