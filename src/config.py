"""
Project-wide configuration using Pydantic Settings.
All paths, API settings, and model hyperparameters live here.
"""
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings


ROOT_DIR = Path(__file__).resolve().parent.parent


class APIConfig(BaseSettings):
    base_url: str = "https://api.openf1.org/v1"
    timeout: int = 30
    max_retries: int = 5
    backoff_factor: float = 1.5
    rate_limit_delay: float = 0.3  # seconds between requests

    model_config = {"env_prefix": "OPENF1_"}


class PathConfig(BaseSettings):
    root: Path = ROOT_DIR
    data: Path = ROOT_DIR / "data"
    raw: Path = ROOT_DIR / "data" / "raw"
    bronze: Path = ROOT_DIR / "data" / "bronze"
    silver: Path = ROOT_DIR / "data" / "silver"
    gold: Path = ROOT_DIR / "data" / "gold"
    models: Path = ROOT_DIR / "data" / "models"
    sample: Path = ROOT_DIR / "data" / "sample"
    logs: Path = ROOT_DIR / "logs"
    cache: Path = ROOT_DIR / ".cache"

    def setup(self) -> None:
        """Create all directories if they don't exist."""
        for field_name, path in self.model_dump().items():
            if isinstance(path, Path):
                path.mkdir(parents=True, exist_ok=True)

    model_config = {"env_prefix": "F1_PATH_"}


class FeatureConfig(BaseSettings):
    # Timeline
    grid_interval_seconds: int = 30
    prediction_horizon_seconds: int = 300  # 5 minutes

    # Rolling windows (seconds)
    short_window: int = 60
    medium_window: int = 180
    long_window: int = 600
    dynamics_window: int = 120
    dynamics_long_window: int = 300

    # Label
    label_column: str = "y_sc_5m"
    time_to_event_column: str = "time_to_sc_seconds"
    time_to_event_cap: int = 1800  # 30 minutes cap

    model_config = {"env_prefix": "F1_FEAT_"}


class ModelConfig(BaseSettings):
    # LightGBM
    lgbm_n_estimators: int = 500
    lgbm_learning_rate: float = 0.05
    lgbm_num_leaves: int = 63
    lgbm_min_child_samples: int = 20
    lgbm_early_stopping_rounds: int = 50

    # TF-IDF
    tfidf_max_features: int = 5000
    tfidf_ngram_min: int = 1
    tfidf_ngram_max: int = 3
    tfidf_svd_components: int = 100

    # Training
    random_state: int = 42
    class_weight: str = "balanced"

    model_config = {"env_prefix": "F1_MODEL_"}


class Config:
    """Unified project configuration."""

    api: APIConfig = APIConfig()
    paths: PathConfig = PathConfig()
    features: FeatureConfig = FeatureConfig()
    model: ModelConfig = ModelConfig()


# Singleton instance
cfg = Config()
