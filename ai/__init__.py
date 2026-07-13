"""AI / LLM inference module.

Usage::

    from ai import create_inference

    engine = create_inference("ollama")
    if engine.is_available():
        result = engine.generate("Hello!")
        print(result.text)
"""

from ai.factory import create_inference
from ai.inference import BaseInference, OllamaInference
from ai.models import (
    AIResult,
    ModelInfo,
    Message,
    Role,
    ChatHistory,
)
from ai.model_manager import ModelManager, ModelNotFoundError
from ai.prompt_manager import PromptManager
from ai.streaming import TokenCollector, consume_stream
from ai.retry import RetryHandler

# Convenience alias for the most common entry point.
InferenceEngine = BaseInference

__all__ = [
    "create_inference",
    "BaseInference",
    "OllamaInference",
    "AIResult",
    "ModelInfo",
    "Message",
    "Role",
    "ChatHistory",
    "ModelManager",
    "ModelNotFoundError",
    "PromptManager",
    "TokenCollector",
    "consume_stream",
    "RetryHandler",
    "InferenceEngine",
]
