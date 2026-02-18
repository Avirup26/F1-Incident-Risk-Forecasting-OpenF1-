"""
Structured logging setup using loguru.
Provides file + console output with rotation.
"""
import sys
from pathlib import Path
from loguru import logger as _logger


def setup_logger(log_dir: Path | None = None, level: str = "INFO") -> None:
    """
    Configure loguru logger with console and optional file sink.

    Args:
        log_dir: Directory to write log files. If None, file logging is skipped.
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR).
    """
    _logger.remove()  # Remove default handler

    # Console handler — colorized, human-readable
    _logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File handler — structured, with rotation
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        _logger.add(
            log_dir / "f1_pipeline_{time:YYYY-MM-DD}.log",
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} — {message}",
            rotation="1 day",
            retention="7 days",
            compression="gz",
        )


# Re-export the configured logger
logger = _logger
