"""Model discovery and availability checking.

Supports Ollama (HTTP API) and llama.cpp (local GGUF files).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config import settings
from ai.models import ModelInfo
from utils import get_logger

logger = get_logger(__name__)


class ModelNotFoundError(LookupError):
    """Raised when a requested model cannot be found."""


class ModelManager:
    """Discover and validate available AI models.

    Usage::

        mgr = ModelManager()
        all_models = mgr.list_models("ollama")
        info = mgr.get_model("llama3", "ollama")
    """

    def list_models(self, provider: str = "") -> list[ModelInfo]:
        """Return all available models for *provider*.

        Args:
            provider: ``"ollama"`` or ``"llama.cpp"``. Defaults to
                ``settings.LLM_PROVIDER``.

        Returns:
            A list of :class:`ModelInfo`. Empty when the provider is
            unavailable or has no models.
        """
        provider = provider or settings.LLM_PROVIDER
        if provider == "ollama":
            return self._list_ollama_models()
        if provider in ("llama.cpp", "llamacpp"):
            return self._list_llamacpp_models()
        logger.warning("Unknown provider: %s", provider)
        return []

    def get_model(self, name: str, provider: str = "") -> ModelInfo:
        """Get metadata for a specific model by name.

        Args:
            name: Model name (e.g. ``"llama3"`` or ``"llama-2-7b.gguf"``).
            provider: Provider to search in.

        Returns:
            :class:`ModelInfo` if found.

        Raises:
            ModelNotFoundError: If the model does not exist.
        """
        provider = provider or settings.LLM_PROVIDER
        models = self.list_models(provider)
        for m in models:
            if m.name == name:
                return m
        raise ModelNotFoundError(f"Model '{name}' not found for provider '{provider}'")

    def is_available(self, provider: str = "") -> bool:
        """Check whether *provider* is reachable / installed.

        For Ollama this sends a request to the API endpoint.
        For llama.cpp this checks if the Python package is importable.
        """
        provider = provider or settings.LLM_PROVIDER
        if provider == "ollama":
            return self._ollama_reachable()
        if provider in ("llama.cpp", "llamacpp"):
            return self._llamacpp_installed()
        return False

    # ── Ollama ─────────────────────────────────────────────────────────

    def _list_ollama_models(self) -> list[ModelInfo]:
        try:
            return self._fetch_ollama_models()
        except Exception as exc:
            logger.debug("Failed to list Ollama models: %s", exc)
            return []

    def _fetch_ollama_models(self) -> list[ModelInfo]:
        try:
            import httpx
        except ImportError:
            return []

        url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/tags"
        with httpx.Client(timeout=10) as client:
            resp = client.get(url)
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

        try:
            url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/tags"
            with httpx.Client(timeout=5) as client:
                resp = client.get(url)
                return resp.is_success
        except Exception:
            return False

    # ── llama.cpp ──────────────────────────────────────────────────────

    def _list_llamacpp_models(self) -> list[ModelInfo]:
        models_dir = self._llama_models_dir()
        if not models_dir.is_dir():
            return []

        models: list[ModelInfo] = []
        for fpath in sorted(models_dir.glob("*.gguf")):
            models.append(
                ModelInfo(
                    name=fpath.stem,
                    provider="llama.cpp",
                    available=True,
                    path=str(fpath),
                    size_bytes=fpath.stat().st_size,
                )
            )
        return models

    def _llama_models_dir(self) -> Path:
        raw = settings.LLAMA_MODELS_DIR
        if isinstance(raw, Path):
            return raw
        return Path(raw) if raw else settings.MODELS_DIR / "llama"

    def _llamacpp_installed(self) -> bool:
        try:
            import llama_cpp  # noqa: F401
            return True
        except ImportError:
            return False
