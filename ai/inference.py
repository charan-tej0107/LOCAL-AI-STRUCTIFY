"""Inference backend for the Ollama API.

Each backend implements the :class:`BaseInference` ABC with
``generate``, ``generate_stream``, ``chat``, and ``chat_stream``.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import Any, Generator

from config import settings
from ai.models import AIResult
from ai.streaming import TokenCollector
from ai.retry import RetryHandler
from utils import get_logger

logger = get_logger(__name__)

_PERMANENT_HTTP_STATUSES = frozenset({400, 401, 403, 404, 405, 406, 422})


class BaseInference(ABC):
    """Abstract base for an inference backend.

    Subclasses must implement :meth:`_generate` and :meth:`_chat`;
    the public methods handle retry, streaming, and result wrapping.
    """

    def __init__(self) -> None:
        self._retry = RetryHandler()

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
    """Inference backend that communicates with an Ollama-compatible API.

    Supports both local and remote API endpoints.
    Authentication is handled via an API key sent as a Bearer token.
    """

    def __init__(self) -> None:
        super().__init__()
        self._base_url = settings.OLLAMA_BASE_URL.rstrip("/")
        if self._base_url.endswith("/api"):
            self._base_url = self._base_url[:-4]
        self._api_key = settings.OLLAMA_API_KEY

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _model_name(self, kwargs: dict[str, Any]) -> str:
        return kwargs.get("model", settings.OLLAMA_MODEL)

    def is_available(self) -> bool:
        try:
            import httpx
        except ImportError:
            return False

        try:
            url = f"{self._base_url}/api/tags"
            with httpx.Client(timeout=5) as client:
                resp = client.get(url, headers=self._headers())
                return resp.is_success
        except Exception:
            return False

    def _generate(self, prompt: str, **kwargs: Any) -> AIResult:
        try:
            import httpx
        except ImportError:
            return AIResult(success=False, error="httpx is not installed. Run: pip install httpx")

        params = self._build_kwargs(kwargs)
        model = self._model_name(kwargs)
        configured_model = settings.OLLAMA_MODEL
        if configured_model != model:
            logger.info("Model override: configured=%s, payload=%s", configured_model, model)
        else:
            logger.info("Using configured model: %s", model)
        payload = {"model": model, "prompt": prompt, "stream": False, **params}
        url = f"{self._base_url}/api/generate"
        start = time.perf_counter()

        if settings.AI_DEBUG:
            logger.debug("=== AI_DEBUG: HTTP Request ===")
            logger.debug("URL: %s", url)
            logger.debug("Configured model: %s", configured_model)
            logger.debug("Payload model: %s", model)
            logger.debug("Full payload:\n%s",
                         json.dumps({k: v for k, v in payload.items() if k != "prompt"},
                                    indent=2, default=str))
            logger.debug("Prompt length: %d chars", len(prompt))
            logger.debug("Prompt:\n%s", prompt)

        try:
            with httpx.Client(timeout=settings.LLM_TIMEOUT) as client:
                resp = client.post(url, json=payload, headers=self._headers())
                resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            body = exc.response.text[:2000]
            if status == 404:
                logger.error(
                    "Ollama API returned 404 for model '%s'. "
                    "Response: %s. Check that OLLAMA_MODEL=%s is available "
                    "on the server at %s.",
                    model, body, configured_model, self._base_url,
                )
            if settings.AI_DEBUG:
                logger.debug("=== AI_DEBUG: HTTP Error ===")
                logger.debug("Status: %d", status)
                logger.debug("Response body: %s", body)
            if status in _PERMANENT_HTTP_STATUSES:
                return _http_error_result(status, _error_detail(exc.response), model)
            raise
        except json.JSONDecodeError:
            if settings.AI_DEBUG:
                logger.debug("=== AI_DEBUG: JSON Decode FAILED ===")
            return AIResult(
                success=False,
                error="Received malformed JSON response from API",
                model_used=model,
            )

        api_elapsed = time.perf_counter() - start
        if settings.AI_DEBUG:
            logger.debug("=== AI_DEBUG: HTTP Response ===")
            logger.debug("Status: %d", resp.status_code)
            safe_headers = {k: v for k, v in resp.headers.items()
                            if k.lower() not in ("authorization", "set-cookie")}
            logger.debug("Headers: %s", safe_headers)
            logger.debug("Response size: %d bytes", len(resp.content))
            logger.debug("Raw body (first 5000 chars):\n%s", resp.text[:5000])
            if len(resp.text) > 5000:
                logger.debug("Raw body (last 2000 chars):\n%s", resp.text[-2000:])
            logger.debug("API request time: %.3fs", api_elapsed)

        try:
            data = resp.json()
        except json.JSONDecodeError:
            if settings.AI_DEBUG:
                logger.debug("=== AI_DEBUG: JSON Decode FAILED ===")
            return AIResult(
                success=False,
                error="Received malformed JSON response from API",
                model_used=model,
            )

        if settings.AI_DEBUG:
            logger.debug("=== AI_DEBUG: JSON Decode ===")
            logger.debug("Response keys: %s", list(data.keys()))
            logger.debug("'response' field length: %d", len(data.get("response", "")))
            logger.debug("Tokens in: %s, Tokens out: %s",
                         data.get("prompt_eval_count"), data.get("eval_count"))

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

        try:
            with httpx.Client(timeout=settings.LLM_TIMEOUT) as client:
                with client.stream("POST", url, json=payload, headers=self._headers()) as resp:
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
        except httpx.HTTPStatusError as exc:
            logger.error("Stream HTTP error %s: %s", exc.response.status_code, _error_detail(exc.response))
        except httpx.TimeoutException:
            logger.error("Stream timed out after %ss", settings.LLM_TIMEOUT)
        except httpx.ConnectError:
            logger.error("Stream connection error to %s", self._base_url)

    def _chat(self, messages: list[dict[str, str]], **kwargs: Any) -> AIResult:
        try:
            import httpx
        except ImportError:
            return AIResult(success=False, error="httpx is not installed. Run: pip install httpx")

        params = self._build_kwargs(kwargs)
        model = self._model_name(kwargs)
        configured_model = settings.OLLAMA_MODEL
        if configured_model != model:
            logger.info("Chat model override: configured=%s, payload=%s", configured_model, model)
        else:
            logger.info("Chat using configured model: %s", model)
        payload = {"model": model, "messages": messages, "stream": False, **params}
        url = f"{self._base_url}/api/chat"
        start = time.perf_counter()

        if settings.AI_DEBUG:
            logger.debug("=== AI_DEBUG: Chat HTTP Request ===")
            logger.debug("URL: %s", url)
            logger.debug("Configured model: %s", configured_model)
            logger.debug("Payload model: %s", model)
            logger.debug("Temperature: %s", params.get("temperature"))
            logger.debug("Max tokens: %s", params.get("max_tokens"))
            logger.debug("Timeout: %s", settings.LLM_TIMEOUT)
            logger.debug("Messages count: %d", len(messages))
            if messages:
                logger.debug("First message role=%s, content_len=%d",
                             messages[0].get("role"), len(messages[0].get("content", "")))

        try:
            with httpx.Client(timeout=settings.LLM_TIMEOUT) as client:
                resp = client.post(url, json=payload, headers=self._headers())
                resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            body = exc.response.text[:2000]
            if status == 404:
                logger.error(
                    "Ollama API returned 404 for model '%s' in chat. "
                    "Response: %s. Check that OLLAMA_MODEL=%s is available "
                    "on the server at %s.",
                    model, body, configured_model, self._base_url,
                )
            if settings.AI_DEBUG:
                logger.debug("=== AI_DEBUG: Chat HTTP Error ===")
                logger.debug("Status: %d", status)
                logger.debug("Response body: %s", body)
            if status in _PERMANENT_HTTP_STATUSES:
                return _http_error_result(status, _error_detail(exc.response), model)
            raise
        except json.JSONDecodeError:
            if settings.AI_DEBUG:
                logger.debug("=== AI_DEBUG: Chat JSON Decode FAILED ===")
            return AIResult(
                success=False,
                error="Received malformed JSON response from API",
                model_used=model,
            )

        if settings.AI_DEBUG:
            logger.debug("=== AI_DEBUG: Chat HTTP Response ===")
            logger.debug("Status: %d", resp.status_code)
            safe_headers = {k: v for k, v in resp.headers.items()
                            if k.lower() not in ("authorization", "set-cookie")}
            logger.debug("Headers: %s", safe_headers)
            logger.debug("Response size: %d bytes", len(resp.content))
            logger.debug("Raw body (first 5000 chars):\n%s", resp.text[:5000])
            if len(resp.text) > 5000:
                logger.debug("Raw body (last 2000 chars):\n%s", resp.text[-2000:])

        try:
            data = resp.json()
        except json.JSONDecodeError:
            if settings.AI_DEBUG:
                logger.debug("=== AI_DEBUG: Chat JSON Decode FAILED ===")
            return AIResult(
                success=False,
                error="Received malformed JSON response from API",
                model_used=model,
            )

        if settings.AI_DEBUG:
            logger.debug("=== AI_DEBUG: Chat JSON Decode ===")
            logger.debug("Response keys: %s", list(data.keys()))
            msg = data.get("message", {})
            logger.debug("Message role: %s, content length: %d",
                         msg.get("role"), len(msg.get("content", "")))
            logger.debug("Tokens in: %s, Tokens out: %s",
                         data.get("prompt_eval_count"), data.get("eval_count"))
            logger.debug("Chat API request time: %.3fs", time.perf_counter() - start)

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

        try:
            with httpx.Client(timeout=settings.LLM_TIMEOUT) as client:
                with client.stream("POST", url, json=payload, headers=self._headers()) as resp:
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
        except httpx.HTTPStatusError as exc:
            logger.error("Chat stream HTTP error %s: %s", exc.response.status_code, _error_detail(exc.response))
        except httpx.TimeoutException:
            logger.error("Chat stream timed out after %ss", settings.LLM_TIMEOUT)
        except httpx.ConnectError:
            logger.error("Chat stream connection error to %s", self._base_url)


def _error_detail(response: Any) -> str:
    """Extract a human-readable error detail from an HTTP response."""
    try:
        body = response.json()
        return body.get("error", body.get("message", str(response.reason_phrase)))
    except Exception:
        return str(response.reason_phrase)


def _http_error_result(status: int, detail: str, model: str) -> AIResult:
    """Build an :class:`AIResult` for an HTTP error response."""
    if status == 401:
        msg = "Invalid API key. Check OLLAMA_API_KEY."
    elif status == 403:
        msg = "Access denied. Check API permissions."
    elif status == 404:
        msg = f"Model '{model}' not found at the API endpoint."
    elif status == 429:
        msg = "Rate limit exceeded. Try again later."
    elif 500 <= status < 600:
        msg = f"Ollama API server error (HTTP {status})."
    else:
        msg = f"HTTP {status}: {detail}"

    return AIResult(success=False, error=msg, model_used=model)
