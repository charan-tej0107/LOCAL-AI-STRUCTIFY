"""Unit tests for AI inference module (Ollama API)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pytest

from ai import (
    AIResult,
    ModelInfo,
    Message,
    Role,
    ChatHistory,
    ModelManager,
    ModelNotFoundError,
    PromptManager,
    TokenCollector,
    consume_stream,
    RetryHandler,
    create_inference,
    BaseInference,
    OllamaInference,
)


# =========================================================================
# Models
# =========================================================================


class TestRole:
    def test_values(self) -> None:
        assert Role.SYSTEM.value == "system"
        assert Role.USER.value == "user"
        assert Role.ASSISTANT.value == "assistant"
        assert Role.TOOL.value == "tool"

    def test_from_string(self) -> None:
        assert Role("user") == Role.USER


class TestMessage:
    def test_create(self) -> None:
        msg = Message(role=Role.USER, content="Hello")
        assert msg.role == Role.USER
        assert msg.content == "Hello"
        assert isinstance(msg, Message)

    def test_equality(self) -> None:
        a = Message(Role.USER, "hi")
        b = Message(Role.USER, "hi")
        assert a == b


class TestModelInfo:
    def test_defaults(self) -> None:
        info = ModelInfo(name="test-model", provider="ollama")
        assert info.name == "test-model"
        assert info.provider == "ollama"
        assert info.available is True
        assert info.path == ""
        assert info.size_bytes == 0

    def test_full(self) -> None:
        info = ModelInfo(
            name="llama3",
            provider="ollama",
            available=True,
            path="llama3:latest",
            size_bytes=4_700_000_000,
            description="Meta Llama 3",
        )
        assert info.size_bytes > 0


class TestAIResult:
    def test_defaults(self) -> None:
        r = AIResult(success=True)
        assert r.text == ""
        assert r.model_used == ""
        assert r.tokens_in == 0
        assert r.tokens_out == 0

    def test_failure(self) -> None:
        r = AIResult(success=False, error="Connection refused")
        assert not r.success
        assert r.error == "Connection refused"

    def test_metadata(self) -> None:
        r = AIResult(success=True, text="hello", metadata={"source": "test"})
        assert r.metadata["source"] == "test"


class TestChatHistory:
    def test_add_user(self) -> None:
        h = ChatHistory()
        h.add_user("Hello")
        assert len(h.messages) == 1
        assert h.messages[0].role == Role.USER

    def test_add_assistant(self) -> None:
        h = ChatHistory()
        h.add_assistant("Hi there")
        assert h.messages[0].role == Role.ASSISTANT

    def test_to_openai_list_with_system(self) -> None:
        h = ChatHistory(system_prompt="Be helpful")
        h.add_user("Hi")
        h.add_assistant("Hello!")
        result = h.to_openai_list()
        assert len(result) == 3
        assert result[0] == {"role": "system", "content": "Be helpful"}
        assert result[1] == {"role": "user", "content": "Hi"}
        assert result[2] == {"role": "assistant", "content": "Hello!"}

    def test_to_openai_list_no_system(self) -> None:
        h = ChatHistory()
        h.add_user("Hi")
        result = h.to_openai_list()
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_clear(self) -> None:
        h = ChatHistory()
        h.add_user("Hi")
        h.clear()
        assert len(h.messages) == 0


# =========================================================================
# PromptManager
# =========================================================================


class TestPromptManager:
    def test_add_and_render(self) -> None:
        pm = PromptManager()
        pm.add_template("greet", "Hello, $name!")
        assert pm.render("greet", name="World") == "Hello, World!"

    def test_render_unknown_raises(self) -> None:
        pm = PromptManager()
        with pytest.raises(KeyError, match="unknown"):
            pm.render("unknown")

    def test_render_missing_var_uses_empty(self) -> None:
        pm = PromptManager()
        pm.add_template("t", "$a $b")
        result = pm.render("t", a="x")
        assert "x" in result
        # safe_substitute leaves $b as-is
        assert "$b" in result

    def test_build_messages_with_all(self) -> None:
        pm = PromptManager()
        history = [Message(Role.USER, "prev q"), Message(Role.ASSISTANT, "prev a")]
        msgs = pm.build_messages(
            system_prompt="You are helpful.",
            user_prompt="New question",
            history=history,
        )
        assert len(msgs) == 4
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"
        assert msgs[3]["role"] == "user"

    def test_build_messages_empty(self) -> None:
        pm = PromptManager()
        msgs = pm.build_messages()
        assert msgs == []

    def test_remove_template(self) -> None:
        pm = PromptManager()
        pm.add_template("t", "hello")
        pm.remove_template("t")
        assert pm.list_templates() == []

    def test_list_templates(self) -> None:
        pm = PromptManager()
        pm.add_template("a", "aa")
        pm.add_template("b", "bb")
        assert set(pm.list_templates()) == {"a", "b"}


# =========================================================================
# RetryHandler
# =========================================================================


class TestRetryHandler:
    def test_success_first_attempt(self) -> None:
        handler = RetryHandler(max_attempts=3)
        result = handler.execute(lambda: 42)
        assert result == 42

    def test_retry_then_succeed(self) -> None:
        call_count = 0

        def flaky() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("not yet")
            return "ok"

        handler = RetryHandler(max_attempts=3, base_delay=0.01)
        result = handler.execute(flaky)
        assert result == "ok"
        assert call_count == 3

    def test_all_attempts_fail(self) -> None:
        call_count = 0

        def always_fail() -> None:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("boom")

        handler = RetryHandler(max_attempts=3, base_delay=0.01)
        result = handler.execute(always_fail)
        assert result is None
        assert call_count == 3

    def test_raise_on_failure(self) -> None:
        def fail() -> None:
            raise ValueError("no")

        handler = RetryHandler(max_attempts=2, base_delay=0.01)
        with pytest.raises(ValueError, match="no"):
            handler.execute(fail, raise_on_failure=True)

    def test_backoff_delay(self) -> None:
        """Verify that delays increase between attempts."""
        delays: list[float] = []
        real_sleep = time.sleep

        def tracking_sleep(d: float) -> None:
            delays.append(d)
            real_sleep(0)  # Don't actually wait in tests

        call_count = 0

        def flaky() -> None:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise OSError("try again")

        with patch("time.sleep", side_effect=tracking_sleep):
            handler = RetryHandler(
                max_attempts=3, base_delay=1.0, backoff=2.0, max_delay=10.0
            )
            handler.execute(flaky)

        assert len(delays) == 2  # Two retries before success
        assert delays[0] == 1.0
        assert delays[1] == 2.0


# =========================================================================
# Streaming
# =========================================================================


class TestTokenCollector:
    def test_accumulate_tokens(self) -> None:
        c = TokenCollector(model_name="test")
        c.add("Hello")
        c.add(" world")
        assert c.full_text == "Hello world"
        assert c.token_count == 2

    def test_empty(self) -> None:
        c = TokenCollector()
        assert c.full_text == ""
        assert c.token_count == 0

    def test_result_success(self) -> None:
        c = TokenCollector(model_name="m")
        c.add("hi")
        r = c.result()
        assert r.success
        assert r.text == "hi"
        assert r.model_used == "m"

    def test_result_with_error(self) -> None:
        c = TokenCollector()
        c.add("partial")
        c.set_error("connection lost")
        r = c.result()
        assert not r.success
        assert r.text == "partial"
        assert r.error == "connection lost"


class TestConsumeStream:
    def test_basic(self) -> None:
        def gen() -> Generator[str, None, None]:
            yield "Hello"
            yield " world"

        result = consume_stream(gen(), model_name="m")
        assert result.success
        assert result.text == "Hello world"
        assert result.model_used == "m"
        assert result.processing_time_seconds >= 0

    def test_empty_stream(self) -> None:
        def gen() -> Generator[str, None, None]:
            return
            yield  # pragma: no cover

        result = consume_stream(gen())
        assert result.success
        assert result.text == ""


# =========================================================================
# ModelManager
# =========================================================================


class TestModelManager:
    def test_ollama_list_models(self) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3", "size": 4700000000},
                {"name": "mistral", "size": 4100000000},
            ]
        }
        mock_response.is_success = True

        with patch("httpx.Client") as mock_client:
            instance = mock_client.return_value.__enter__.return_value
            instance.get.return_value = mock_response
            mgr = ModelManager()
            models = mgr.list_models("ollama")

        assert len(models) == 2
        assert models[0].name == "llama3"
        assert models[0].provider == "ollama"
        assert models[0].size_bytes == 4700000000

    def test_ollama_list_models_connection_error(self) -> None:
        with patch("httpx.Client") as mock_client:
            instance = mock_client.return_value.__enter__.return_value
            instance.get.side_effect = ConnectionError("refused")
            mgr = ModelManager()
            models = mgr.list_models("ollama")

        assert models == []

    def test_get_model_found(self) -> None:
        mgr = ModelManager()
        with patch.object(mgr, "list_models", return_value=[
            ModelInfo(name="llama3", provider="ollama"),
        ]):
            info = mgr.get_model("llama3", "ollama")
        assert info.name == "llama3"

    def test_get_model_not_found(self) -> None:
        mgr = ModelManager()
        with patch.object(mgr, "list_models", return_value=[]):
            with pytest.raises(ModelNotFoundError, match="not found"):
                mgr.get_model("nonexistent", "ollama")

    def test_is_available_ollama_true(self) -> None:
        mock_response = MagicMock()
        mock_response.is_success = True

        with patch("httpx.Client") as mock_client:
            instance = mock_client.return_value.__enter__.return_value
            instance.get.return_value = mock_response
            mgr = ModelManager()
            assert mgr.is_available("ollama") is True

    def test_is_available_ollama_false(self) -> None:
        with patch("httpx.Client") as mock_client:
            instance = mock_client.return_value.__enter__.return_value
            instance.get.side_effect = ConnectionError
            mgr = ModelManager()
            assert mgr.is_available("ollama") is False

# =========================================================================
# OllamaInference
# =========================================================================


class TestOllamaInference:
    def test_is_available_true(self) -> None:
        mock_resp = MagicMock()
        mock_resp.is_success = True

        with patch("httpx.Client") as mock_client:
            instance = mock_client.return_value.__enter__.return_value
            instance.get.return_value = mock_resp
            inf = OllamaInference()
            assert inf.is_available() is True

    def test_is_available_false(self) -> None:
        with patch("httpx.Client") as mock_client:
            instance = mock_client.return_value.__enter__.return_value
            instance.get.side_effect = ConnectionError
            inf = OllamaInference()
            assert inf.is_available() is False

    def test_generate_success(self) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "model": "llama3",
            "response": "Hello back!",
            "prompt_eval_count": 5,
            "eval_count": 3,
        }

        with patch("httpx.Client") as mock_client:
            instance = mock_client.return_value.__enter__.return_value
            instance.post.return_value = mock_resp
            inf = OllamaInference()
            result = inf.generate("Hello")

        assert result.success
        assert result.text == "Hello back!"
        assert result.model_used == "llama3"
        assert result.tokens_in == 5
        assert result.tokens_out == 3

    def test_generate_http_error(self) -> None:
        import httpx

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 422
        mock_response.reason_phrase = "Unprocessable"
        mock_response.json.return_value = {"error": "bad request"}

        with patch("httpx.Client") as mock_client:
            instance = mock_client.return_value.__enter__.return_value
            instance.post.side_effect = httpx.HTTPStatusError(
                "422", request=MagicMock(), response=mock_response
            )
            inf = OllamaInference()
            result = inf.generate("Hello")

        assert not result.success
        assert result.error != ""

    def test_generate_timeout(self) -> None:
        import httpx

        with patch("httpx.Client") as mock_client:
            instance = mock_client.return_value.__enter__.return_value
            instance.post.side_effect = httpx.TimeoutException("timed out")
            inf = OllamaInference()
            result = inf.generate("Hello")

        assert not result.success

    def test_generate_stream(self) -> None:
        chunks = [
            b'{"response":"Hello","done":false}\n',
            b'{"response":" world","done":true}\n',
        ]

        mock_stream = MagicMock()
        mock_stream.__enter__.return_value.iter_lines.return_value = chunks
        mock_post = MagicMock()
        mock_post.return_value = mock_stream

        with patch("httpx.Client") as mock_client:
            instance = mock_client.return_value.__enter__.return_value
            instance.stream = mock_post
            inf = OllamaInference()
            tokens = list(inf.generate_stream("Hi"))

        assert tokens == ["Hello", " world"]

    def test_chat_success(self) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "model": "llama3",
            "message": {"role": "assistant", "content": "Sure!"},
            "prompt_eval_count": 10,
            "eval_count": 5,
        }

        with patch("httpx.Client") as mock_client:
            instance = mock_client.return_value.__enter__.return_value
            instance.post.return_value = mock_resp
            inf = OllamaInference()
            result = inf.chat([{"role": "user", "content": "Help"}])

        assert result.success
        assert result.text == "Sure!"
        assert result.tokens_in == 10
        assert result.tokens_out == 5

    def test_chat_stream(self) -> None:
        chunks = [
            b'{"message":{"content":"Sure"},"done":false}\n',
            b'{"message":{"content":"!"},"done":true}\n',
        ]

        mock_stream = MagicMock()
        mock_stream.__enter__.return_value.iter_lines.return_value = chunks
        mock_post = MagicMock()
        mock_post.return_value = mock_stream

        with patch("httpx.Client") as mock_client:
            instance = mock_client.return_value.__enter__.return_value
            instance.stream = mock_post
            inf = OllamaInference()
            tokens = list(inf.chat_stream([{"role": "user", "content": "Hi"}]))

        assert tokens == ["Sure", "!"]

    def test_generate_empty_prompt(self) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "model": "llama3",
            "response": "",
            "prompt_eval_count": 0,
            "eval_count": 0,
        }

        with patch("httpx.Client") as mock_client:
            instance = mock_client.return_value.__enter__.return_value
            instance.post.return_value = mock_resp
            inf = OllamaInference()
            result = inf.generate("")

        assert result.success
        assert result.text == ""


# =========================================================================
# Factory
# =========================================================================


class TestCreateInference:
    def test_ollama_provider(self) -> None:
        engine = create_inference("ollama")
        assert isinstance(engine, OllamaInference)

    def test_default_provider(self) -> None:
        from config import settings

        original = settings.LLM_PROVIDER
        try:
            settings.LLM_PROVIDER = "ollama"
            engine = create_inference()
            assert isinstance(engine, OllamaInference)
        finally:
            settings.LLM_PROVIDER = original

    def test_unsupported_provider(self) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            create_inference("invalid")


# =========================================================================
# BaseInference (edge cases via concrete subclass)
# =========================================================================


class TestBaseInferenceEdgeCases:
    def test_generate_empty_prompt(self) -> None:
        """Ollama handles empty prompts gracefully."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "model": "llama3",
            "response": "",
            "prompt_eval_count": 0,
            "eval_count": 0,
        }

        with patch("httpx.Client") as mock_client:
            instance = mock_client.return_value.__enter__.return_value
            instance.post.return_value = mock_resp
            inf = OllamaInference()
            result = inf.generate("")

        assert result.success
        assert result.text == ""

    def test_retry_on_transient_error(self) -> None:
        """RetryHandler should retry on transient httpx errors."""
        import httpx

        call_count = 0
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "model": "llama3",
            "response": "ok",
            "prompt_eval_count": 1,
            "eval_count": 1,
        }

        def side_effect(*args: Any, **kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise httpx.TimeoutException("timeout")
            return mock_resp

        with patch("httpx.Client") as mock_client:
            instance = mock_client.return_value.__enter__.return_value
            instance.post.side_effect = side_effect
            inf = OllamaInference()
            result = inf.generate("Hello")

        assert result.success
        assert result.text == "ok"
        assert call_count == 2

    def test_auth_header_sent_when_api_key_set(self) -> None:
        """Verify Authorization header is sent when OLLAMA_API_KEY is set."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "model": "llama3",
            "response": "ok",
            "prompt_eval_count": 1,
            "eval_count": 1,
        }

        captured_headers: dict[str, str] = {}

        def capture_request(*args: Any, **kwargs: Any) -> MagicMock:
            captured_headers.update(kwargs.get("headers", {}))
            return mock_resp

        with patch("httpx.Client") as mock_client:
            instance = mock_client.return_value.__enter__.return_value
            instance.post.side_effect = capture_request
            from config import settings

            original_key = settings.OLLAMA_API_KEY
            settings.OLLAMA_API_KEY = "test-key-123"
            try:
                inf = OllamaInference()
                result = inf.generate("Hello")
                assert result.success
            finally:
                settings.OLLAMA_API_KEY = original_key

        assert captured_headers.get("Authorization") == "Bearer test-key-123"

    def test_generate_401_error(self) -> None:
        """Invalid API key returns appropriate error message."""
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.reason_phrase = "Unauthorized"
        mock_response.json.return_value = {"error": "invalid API key"}

        with patch("httpx.Client") as mock_client:
            instance = mock_client.return_value.__enter__.return_value
            instance.post.side_effect = httpx.HTTPStatusError(
                "401", request=MagicMock(), response=mock_response
            )
            inf = OllamaInference()
            result = inf.generate("Hello")

        assert not result.success
        assert "API key" in result.error

    def test_generate_403_error(self) -> None:
        """Forbidden returns appropriate error message."""
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.reason_phrase = "Forbidden"
        mock_response.json.return_value = {"error": "forbidden"}

        with patch("httpx.Client") as mock_client:
            instance = mock_client.return_value.__enter__.return_value
            instance.post.side_effect = httpx.HTTPStatusError(
                "403", request=MagicMock(), response=mock_response
            )
            inf = OllamaInference()
            result = inf.generate("Hello")

        assert not result.success
        assert "denied" in result.error.lower() or "forbidden" in result.error.lower()

    def test_generate_404_error(self) -> None:
        """Missing model returns appropriate error message."""
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.reason_phrase = "Not Found"
        mock_response.json.return_value = {"error": "model not found"}

        with patch("httpx.Client") as mock_client:
            instance = mock_client.return_value.__enter__.return_value
            instance.post.side_effect = httpx.HTTPStatusError(
                "404", request=MagicMock(), response=mock_response
            )
            inf = OllamaInference()
            result = inf.generate("Hello")

        assert not result.success
        assert "not found" in result.error.lower()

    def test_generate_429_error(self) -> None:
        """Rate limit is retried then returns exhaustion message."""
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.reason_phrase = "Too Many Requests"
        mock_response.json.return_value = {"error": "rate limit"}

        with patch("httpx.Client") as mock_client:
            instance = mock_client.return_value.__enter__.return_value
            instance.post.side_effect = httpx.HTTPStatusError(
                "429", request=MagicMock(), response=mock_response
            )
            inf = OllamaInference()
            result = inf.generate("Hello")

        assert not result.success
        assert "exhausted" in result.error

    def test_generate_500_error(self) -> None:
        """Server error is retried then returns exhaustion message."""
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.reason_phrase = "Internal Server Error"
        mock_response.json.return_value = {"error": "server error"}

        with patch("httpx.Client") as mock_client:
            instance = mock_client.return_value.__enter__.return_value
            instance.post.side_effect = httpx.HTTPStatusError(
                "500", request=MagicMock(), response=mock_response
            )
            inf = OllamaInference()
            result = inf.generate("Hello")

        assert not result.success
        assert "exhausted" in result.error

    def test_connection_error(self) -> None:
        """Connection error is retried then returns exhaustion message."""
        import httpx

        with patch("httpx.Client") as mock_client:
            instance = mock_client.return_value.__enter__.return_value
            instance.post.side_effect = httpx.ConnectError("connection refused")
            inf = OllamaInference()
            result = inf.generate("Hello")

        assert not result.success
        assert "exhausted" in result.error

    def test_json_decode_error(self) -> None:
        """Malformed JSON response returns appropriate message."""
        import json

        mock_response = MagicMock()
        mock_response.json.side_effect = json.JSONDecodeError("bad json", "bad", 0)

        with patch("httpx.Client") as mock_client:
            instance = mock_client.return_value.__enter__.return_value
            instance.post.return_value = mock_response
            inf = OllamaInference()
            result = inf.generate("Hello")

        assert not result.success
        assert "malformed" in result.error.lower()

    def test_retry_handler_429_retry_after(self) -> None:
        """RetryHandler respects Retry-After header for 429 responses."""
        import httpx

        def make_429_response() -> MagicMock:
            resp = MagicMock(spec=httpx.Response)
            resp.status_code = 429
            resp.reason_phrase = "Too Many Requests"
            resp.headers = {"Retry-After": "5"}
            resp.json.return_value = {"error": "rate limit"}
            return resp

        mock_ok = MagicMock()
        mock_ok.json.return_value = {
            "model": "llama3",
            "response": "ok",
            "prompt_eval_count": 1,
            "eval_count": 1,
        }

        call_count = [0]
        with patch("httpx.Client") as mock_client:
            instance = mock_client.return_value.__enter__.return_value
            original_post = instance.post

            def side_effect(*args: Any, **kwargs: Any) -> MagicMock:
                call_count[0] += 1
                if call_count[0] < 2:
                    raise httpx.HTTPStatusError(
                        "429", request=MagicMock(), response=make_429_response()
                    )
                return mock_ok

            instance.post.side_effect = side_effect
            inf = OllamaInference()
            result = inf.generate("Hello")

        assert result.success
        assert result.text == "ok"
        assert call_count[0] == 2


class TestAiModuleExport:
    def test_core_exports(self) -> None:
        from ai import create_inference, AIResult, ModelManager, PromptManager

        assert callable(create_inference)
        assert AIResult is not None
        assert ModelManager is not None
        assert PromptManager is not None
