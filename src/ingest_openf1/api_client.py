"""
Robust OpenF1 API client with:
- Exponential backoff retry logic
- Disk-level JSON caching
- Rate limiting to avoid 429 errors
- Session-based connection pooling
"""
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlencode

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import cfg
from src.utils.logger import logger


class OpenF1Client:
    """
    HTTP client for the OpenF1 REST API.

    Handles retries, caching, and rate limiting transparently.
    """

    def __init__(
        self,
        base_url: str | None = None,
        cache_dir: Path | None = None,
        use_cache: bool = True,
    ) -> None:
        self.base_url = (base_url or cfg.api.base_url).rstrip("/")
        self.cache_dir = cache_dir or cfg.paths.cache
        self.use_cache = use_cache
        self._last_request_time: float = 0.0

        # Set up session with retry adapter
        self.session = requests.Session()
        retry_strategy = Retry(
            total=cfg.api.max_retries,
            backoff_factor=cfg.api.backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers.update({"Accept": "application/json"})

    def _cache_path(self, endpoint: str, params: dict) -> Path:
        """Generate a deterministic cache file path for a request."""
        key = f"{endpoint}?{urlencode(sorted(params.items()))}"
        digest = hashlib.md5(key.encode()).hexdigest()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / f"{digest}.json"

    def _rate_limit(self) -> None:
        """Enforce minimum delay between requests."""
        elapsed = time.monotonic() - self._last_request_time
        wait = cfg.api.rate_limit_delay - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_time = time.monotonic()

    def get(
        self,
        endpoint: str,
        params: Optional[dict[str, Any]] = None,
        force_refresh: bool = False,
    ) -> list[dict]:
        """
        Fetch data from an OpenF1 endpoint.

        Args:
            endpoint: API endpoint path (e.g. '/sessions').
            params: Query parameters dict.
            force_refresh: If True, bypass cache and re-fetch.

        Returns:
            List of records (dicts) from the API response.
        """
        params = params or {}
        cache_path = self._cache_path(endpoint, params)

        # Serve from cache if available
        if self.use_cache and not force_refresh and cache_path.exists():
            logger.debug(f"Cache hit: {endpoint} {params}")
            with open(cache_path) as f:
                return json.load(f)

        # Build URL
        url = f"{self.base_url}{endpoint}"
        logger.debug(f"Fetching: {url} params={params}")

        self._rate_limit()
        try:
            response = self.session.get(
                url, params=params, timeout=cfg.api.timeout
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error for {url}: {e}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            raise

        # Persist to cache
        if self.use_cache:
            with open(cache_path, "w") as f:
                json.dump(data, f)
            logger.debug(f"Cached response → {cache_path.name}")

        return data
