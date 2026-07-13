"""Model discovery and availability checking for the Ollama API."""

from __future__ import annotations

from typing import Any

from config import settings
from ai.models import ModelInfo
from utils import get_logger

logger = get_logger(__name__)


class ModelNotFoundError(LookupError):
    """Raised when a requested model cannot be found."""


class ModelManager:
    """Discover and validate available AI models via the Ollama API.

    Usage::

        mgr = ModelManager()
        all_models = mgr.list_models()
        info = mgr.get_model("llama3")
    """

    def list_models(self, provider: str = "") -> list[ModelInfo]:
        """Return all available models for *provider*.

        Args:
            provider: Ignored for now — only ``"ollama"`` is supported.

        Returns:
            A list of :class:`ModelInfo`. Empty when the API is
            unavailable or has no models.
        """
        return self._fetch_ollama_models()

    def get_model(self, name: str, provider: str = "") -> ModelInfo:
        """Get metadata for a specific model by name.

        Args:
            name: Model name (e.g. ``"llama3"``).
            provider: Ignored for now.

        Returns:
            :class:`ModelInfo` if found.

        Raises:
            ModelNotFoundError: If the model does not exist.
        """
        models = self.list_models()
        for m in models:
            if m.name == name:
                return m
        raise ModelNotFoundError(f"Model '{name}' not found on the API server")

    def is_available(self, provider: str = "") -> bool:
        """Check whether the Ollama API endpoint is reachable."""
        return self._ollama_reachable()

    def _fetch_ollama_models(self) -> list[ModelInfo]:
        try:
            return self._request_ollama_models()
        except Exception as exc:
            logger.debug("Failed to list Ollama models: %s", exc)
            return []

    def _request_ollama_models(self) -> list[ModelInfo]:
        try:
            import httpx
        except ImportError:
            return []

        api_key = settings.OLLAMA_API_KEY
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        base_url = settings.OLLAMA_BASE_URL.rstrip("/")
        if base_url.endswith("/api"):
            base_url = base_url[:-4]
        url = f"{base_url}/api/tags"
        with httpx.Client(timeout=10) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        models: list[ModelInfo] = []
        for item in data.get("models", []):
            name: str = item.get("name", "unknown")
            size: int = item.get("size", 0)
            models.append(
                ModelInfo(
                    name=name,
                    provider="ollama",
                    available=True,
                    path=name,
                    size_bytes=size,
                )
            )
        return models

    def _ollama_reachable(self) -> bool:
        try:
            import httpx
        except ImportError:
            return False

        api_key = settings.OLLAMA_API_KEY
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            base_url = settings.OLLAMA_BASE_URL.rstrip("/")
            if base_url.endswith("/api"):
                base_url = base_url[:-4]
            url = f"{base_url}/api/tags"
            with httpx.Client(timeout=5) as client:
                resp = client.get(url, headers=headers)
                return resp.is_success
        except Exception:
            return False
