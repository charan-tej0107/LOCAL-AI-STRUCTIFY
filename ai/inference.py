"""Inference backends — Ollama and llama.cpp.

Each backend implements the :class:`BaseInference` ABC with
``generate``, ``generate_stream``, ``chat``, and ``chat_stream``.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generator

from config import settings
from ai.models import AIResult
from ai.streaming import TokenCollector
from ai.retry import RetryHandler
from utils import get_logger

logger = get_logger(__name__)


class BaseInference(ABC):
    """Abstract base for an inference backend.

    Subclasses must implement :meth:`_generate` and :meth:`_chat`;
    the public methods handle retry, streaming, and result wrapping.
    """

    def __init__(self) -> None:
        self._retry = RetryHandler()

    # ── Public API ─────────────────────────────────────────────────────

    def generate(self, prompt: str, **kwargs: Any) -> AIResult:
        """Generate a completion for *prompt* (non-streaming)."""
        result = self._retry.execute(self._generate, prompt, **kwargs)
        if result is None:
            return AIResult(success=False, error="All retry attempts exhausted")
        return result

    def generate_stream(
        self, prompt: str, **kwargs: Any
    ) -> Generator[str, None, AIResult]:
        """Stream a completion for *prompt*, yielding tokens."""
        yield from self._stream_wrapper(self._generate_stream, prompt, **kwargs)

    def chat(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> AIResult:
        """Send a chat completion request (non-streaming)."""
        result = self._retry.execute(self._chat, messages, **kwargs)
        if result is None:
            return AIResult(success=False, error="All retry attempts exhausted")
        return result

    def chat_stream(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> Generator[str, None, AIResult]:
        """Stream a chat completion, yielding tokens."""
        yield from self._stream_wrapper(self._chat_stream, messages, **kwargs)

    def is_available(self) -> bool:
        """Check whether this backend is usable right now."""
        return True

    # ── Subclass hooks ─────────────────────────────────────────────────

    @abstractmethod
    def _generate(self, prompt: str, **kwargs: Any) -> AIResult:
        ...

    @abstractmethod
    def _generate_stream(
        self, prompt: str, **kwargs: Any
    ) -> Generator[str, None, None]:
        ...

    @abstractmethod
    def _chat(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> AIResult:
        ...

    @abstractmethod
    def _chat_stream(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> Generator[str, None, None]:
        ...

    # ── Helpers ────────────────────────────────────────────────────────

    def _stream_wrapper(
        self,
        stream_fn: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Generator[str, None, AIResult]:
        """Wrap a stream generator with retry and result collection."""
        collector = TokenCollector(model_name="")
        try:
            stream = stream_fn(*args, **kwargs)
            for token in stream:
                collector.add(token)
                yield token
        except Exception as exc:
            collector.set_error(str(exc))
        else:
            return collector.result()

    def _build_kwargs(self, overrides: dict[str, Any]) -> dict[str, Any]:
        return {
            "temperature": overrides.get("temperature", settings.LLM_TEMPERATURE),
            "max_tokens": overrides.get("max_tokens", settings.LLM_MAX_TOKENS),
            "top_p": overrides.get("top_p", settings.LLM_TOP_P),
        }


class OllamaInference(BaseInference):
    """Inference backend for Ollama via its HTTP API."""

    def __init__(self) -> None:
        super().__init__()
        self._base_url = settings.OLLAMA_BASE_URL.rstrip("/")

    def _model_name(self, kwargs: dict[str, Any]) -> str:
        """Return the Ollama model name."""
        return kwargs.get(
            "model",
            getattr(settings, "OLLAMA_MODEL", "llama3.2:latest"),
        )

    # ── Availability ───────────────────────────────────────────────────

    def is_available(self) -> bool:
        try:
            import httpx
        except ImportError:
            return False

        try:
            url = f"{self._base_url}/api/tags"
            with httpx.Client(timeout=5) as client:
                resp = client.get(url)
                return resp.is_success
        except Exception:
            return False

    # ── Generate (non-streaming) ───────────────────────────────────────

    def _generate(self, prompt: str, **kwargs: Any) -> AIResult:
        try:
            import httpx
        except ImportError:
            return AIResult(success=False, error="httpx is not installed. Run: pip install httpx")

        params = self._build_kwargs(kwargs)
        model = self._model_name(kwargs)
        payload = {"model": model, "prompt": prompt, "stream": False, **params}
        url = f"{self._base_url}/api/generate"
        start = time.perf_counter()

        with httpx.Client(timeout=settings.LLM_TIMEOUT) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        elapsed = time.perf_counter() - start
        return AIResult(
            success=True,
            text=data.get("response", ""),
            model_used=data.get("model", model),
            tokens_in=data.get("prompt_eval_count", 0),
            tokens_out=data.get("eval_count", 0),
            processing_time_seconds=round(elapsed, 3),
        )

    def _generate_stream(
        self, prompt: str, **kwargs: Any
    ) -> Generator[str, None, None]:
        try:
            import httpx
        except ImportError:
            return

        params = self._build_kwargs(kwargs)
        model = self._model_name(kwargs)
        payload = {"model": model, "prompt": prompt, "stream": True, **params}
        url = f"{self._base_url}/api/generate"

        with httpx.Client(timeout=settings.LLM_TIMEOUT) as client:
            with client.stream("POST", url, json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = chunk.get("response", "")
                    if token:
                        yield token
                    if chunk.get("done", False):
                        break

    # ── Chat ───────────────────────────────────────────────────────────

    def _chat(self, messages: list[dict[str, str]], **kwargs: Any) -> AIResult:
        try:
            import httpx
        except ImportError:
            return AIResult(success=False, error="httpx is not installed. Run: pip install httpx")

        params = self._build_kwargs(kwargs)
        model = self._model_name(kwargs)
        payload = {"model": model, "messages": messages, "stream": False, **params}
        url = f"{self._base_url}/api/chat"
        start = time.perf_counter()

        with httpx.Client(timeout=settings.LLM_TIMEOUT) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        elapsed = time.perf_counter() - start
        msg = data.get("message", {})
        return AIResult(
            success=True,
            text=msg.get("content", ""),
            model_used=data.get("model", model),
            tokens_in=data.get("prompt_eval_count", 0),
            tokens_out=data.get("eval_count", 0),
            processing_time_seconds=round(elapsed, 3),
        )

    def _chat_stream(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> Generator[str, None, None]:
        try:
            import httpx
        except ImportError:
            return

        params = self._build_kwargs(kwargs)
        model = self._model_name(kwargs)
        payload = {"model": model, "messages": messages, "stream": True, **params}
        url = f"{self._base_url}/api/chat"

        with httpx.Client(timeout=settings.LLM_TIMEOUT) as client:
            with client.stream("POST", url, json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    delta = chunk.get("message", {})
                    token = delta.get("content", "")
                    if token:
                        yield token
                    if chunk.get("done", False):
                        break


class LlamaCppInference(BaseInference):
    """Inference backend for llama.cpp via ``llama-cpp-python``.

    Loads a single GGUF model file.  Use :meth:`is_available` to
    check that both the package and a model file exist.
    """

    def __init__(self, model_path: str = "") -> None:
        super().__init__()
        self._model_path = model_path
        self._llama: Any = None

    # ── Availability ───────────────────────────────────────────────────

    def is_available(self) -> bool:
        try:
            import llama_cpp  # noqa: F401
        except ImportError:
            return False
        if self._model_path:
            return Path(self._model_path).is_file()
        models_dir = self._resolve_models_dir()
        return any(models_dir.glob("*.gguf"))

    # ── Generate ───────────────────────────────────────────────────────

    def _load_model(self) -> Any:
        if self._llama is not None:
            return self._llama
        if not self._model_path:
            self._model_path = self._find_first_gguf()
        if not self._model_path:
            raise FileNotFoundError(
                "No llama.cpp model found — place a .gguf file in the models directory"
            )
        try:
            import llama_cpp
        except ImportError:
            raise RuntimeError("llama-cpp-python is not installed. Run: pip install llama-cpp-python")

        self._llama = llama_cpp.Llama(
            model_path=self._model_path,
            n_threads=settings.LLAMA_CPP_N_THREADS,
            n_gpu_layers=settings.LLAMA_CPP_N_GPU_LAYERS,
            verbose=False,
        )
        return self._llama

    def _generate(self, prompt: str, **kwargs: Any) -> AIResult:
        llm = self._load_model()
        params = self._build_kwargs(kwargs)
        start = time.perf_counter()
        output = llm.create_completion(
            prompt,
            max_tokens=params["max_tokens"],
            temperature=params["temperature"],
            top_p=params["top_p"],
            stop=kwargs.get("stop", []),
            stream=False,
        )
        elapsed = time.perf_counter() - start
        choice = output.get("choices", [{}])[0]
        return AIResult(
            success=True,
            text=choice.get("text", ""),
            model_used=Path(self._model_path).stem if self._model_path else "",
            tokens_in=output.get("usage", {}).get("prompt_tokens", 0),
            tokens_out=output.get("usage", {}).get("completion_tokens", 0),
            processing_time_seconds=round(elapsed, 3),
        )

    def _generate_stream(
        self, prompt: str, **kwargs: Any
    ) -> Generator[str, None, None]:
        llm = self._load_model()
        params = self._build_kwargs(kwargs)
        for output in llm.create_completion(
            prompt,
            max_tokens=params["max_tokens"],
            temperature=params["temperature"],
            top_p=params["top_p"],
            stop=kwargs.get("stop", []),
            stream=True,
        ):
            choice = output.get("choices", [{}])[0]
            token = choice.get("text", "")
            if token:
                yield token

    # ── Chat ───────────────────────────────────────────────────────────

    def _chat(self, messages: list[dict[str, str]], **kwargs: Any) -> AIResult:
        llm = self._load_model()
        params = self._build_kwargs(kwargs)
        start = time.perf_counter()
        output = llm.create_chat_completion(
            messages,
            max_tokens=params["max_tokens"],
            temperature=params["temperature"],
            top_p=params["top_p"],
            stop=kwargs.get("stop", []),
            stream=False,
        )
        elapsed = time.perf_counter() - start
        choice = output.get("choices", [{}])[0]
        msg = choice.get("message", {})
        return AIResult(
            success=True,
            text=msg.get("content", ""),
            model_used=Path(self._model_path).stem if self._model_path else "",
            tokens_in=output.get("usage", {}).get("prompt_tokens", 0),
            tokens_out=output.get("usage", {}).get("completion_tokens", 0),
            processing_time_seconds=round(elapsed, 3),
        )

    def _chat_stream(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> Generator[str, None, None]:
        llm = self._load_model()
        params = self._build_kwargs(kwargs)
        for output in llm.create_chat_completion(
            messages,
            max_tokens=params["max_tokens"],
            temperature=params["temperature"],
            top_p=params["top_p"],
            stop=kwargs.get("stop", []),
            stream=True,
        ):
            choice = output.get("choices", [{}])[0]
            delta = choice.get("delta", {})
            token = delta.get("content", "")
            if token:
                yield token

    # ── Helpers ────────────────────────────────────────────────────────

    def _resolve_models_dir(self) -> Path:
        raw = settings.LLAMA_MODELS_DIR
        if isinstance(raw, Path):
            return raw
        return Path(raw) if raw else settings.MODELS_DIR / "llama"

    def _find_first_gguf(self) -> str:
        models_dir = self._resolve_models_dir()
        files = sorted(models_dir.glob("*.gguf"))
        return str(files[0]) if files else ""

    def set_model(self, model_path: str) -> None:
        """Set a specific GGUF model file path."""
        self._model_path = model_path
        self._llama = None
