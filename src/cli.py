"""
Click-based CLI for the F1 Incident Risk Forecasting pipeline.

Usage:
    python -m src.cli setup
    python -m src.cli ingest --year 2024 --limit 5
    python -m src.cli build_features
    python -m src.cli train
    python -m src.cli evaluate
"""
import click
from src.config import cfg
from src.utils.logger import setup_logger, logger


@click.group()
def cli() -> None:
    """🏎️  F1 Incident Risk Forecasting Pipeline"""
    setup_logger(log_dir=cfg.paths.logs)


@cli.command()
def setup() -> None:
    """Initialize project directories."""
    logger.info("Setting up project directories...")
    cfg.paths.setup()
    logger.success("✅ All directories created.")


@cli.command()
@click.option("--year", default=2023, show_default=True, help="F1 season year (2018+).")
@click.option("--limit", default=None, type=int, help="Max sessions to fetch (for testing).")
def ingest(year: int, limit: int | None) -> None:
    """Fetch data via FastF1 (free, no API key needed) and save as bronze parquet tables."""
    from src.ingest_fastf1.pipeline import run_fastf1_pipeline

    cfg.paths.setup()
    logger.info(f"Starting FastF1 ingestion for {year} season (limit={limit})...")
    run_fastf1_pipeline(year=year, limit=limit)
    logger.success("✅ Ingestion complete.")


@cli.command()
def build_features() -> None:
    """Build timeline, labels, and features from bronze data."""
    from src.features.feature_pipeline import run_feature_pipeline

    cfg.paths.setup()
    logger.info("Building features...")
    run_feature_pipeline()
    logger.success("✅ Feature pipeline complete.")


@cli.command()
def train() -> None:
    """Train baseline and LightGBM models."""
    from src.models.trainer import run_training

    cfg.paths.setup()
    logger.info("Starting model training...")
    run_training()
    logger.success("✅ Training complete.")


@cli.command()
def evaluate() -> None:
    """Evaluate trained models and generate report."""
    from src.eval.report_generator import run_evaluation

    cfg.paths.setup()
    logger.info("Running evaluation...")
    run_evaluation()
    logger.success("✅ Evaluation complete.")


@cli.command()
def app() -> None:
    """Launch Streamlit dashboard."""
    import subprocess
    import sys

    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "src/app/main.py"], check=True)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    cli()
