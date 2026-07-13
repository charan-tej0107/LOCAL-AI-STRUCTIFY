"""Inference backend factory — select provider by name."""

from __future__ import annotations

from config import settings
from ai.inference import BaseInference, OllamaInference


def create_inference(provider: str = "") -> BaseInference:
    """Create an inference backend for the named *provider*.

    Args:
        provider: ``"ollama"`` (the only supported provider).
            Defaults to ``settings.LLM_PROVIDER``.

    Returns:
        A ready-to-use :class:`BaseInference` instance.

    Raises:
        ValueError: If *provider* is not supported.
    """
    provider = provider or settings.LLM_PROVIDER

    if provider == "ollama":
        return OllamaInference()

    msg = f"Unsupported LLM provider: '{provider}'. Only 'ollama' is supported."
    raise ValueError(msg)
