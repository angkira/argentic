import sys
import types

import pytest

# ---------------------------------------------------------------------------
# Fixtures to stub out external dependencies that may not be installed during
# CI execution. This allows us to import OllamaProvider without errors.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _stub_langchain_ollama(monkeypatch):
    """Insert dummy langchain_ollama modules so provider can import."""

    # Base dummy class with invoke / ainvoke methods
    class _DummyLLM:
        def __init__(self, *args, **kwargs):
            pass

        # Synchronous invoke returns constant string
        def invoke(self, *args, **kwargs):
            return "dummy"

        async def ainvoke(self, *args, **kwargs):
            return "dummy"

    # Create main langchain_ollama module
    ollama_mod = types.ModuleType("langchain_ollama")
    ollama_mod.OllamaLLM = _DummyLLM  # type: ignore
    sys.modules["langchain_ollama"] = ollama_mod

    # Create sub-module langchain_ollama.chat_models with ChatOllama
    chat_models_mod = types.ModuleType("langchain_ollama.chat_models")

    class _DummyChatOllama(_DummyLLM):
        def invoke(self, *args, **kwargs):  # noqa: D401, N802 – signature matches real class
            return "dummy"

        async def ainvoke(self, *args, **kwargs):
            return "dummy"

    chat_models_mod.ChatOllama = _DummyChatOllama  # type: ignore
    sys.modules["langchain_ollama.chat_models"] = chat_models_mod

    yield

    # Cleanup after test session
    sys.modules.pop("langchain_ollama", None)
    sys.modules.pop("langchain_ollama.chat_models", None)


# ---------------------------------------------------------------------------
# Actual tests
# ---------------------------------------------------------------------------

from unittest.mock import AsyncMock, patch

import httpx

from argentic.core.llm.providers.ollama import OllamaProvider  # noqa: E402 – import after stubs


def _create_provider() -> OllamaProvider:
    config = {
        "llm": {
            "ollama_model_name": "gemma3:12b-it-qat",
            "ollama_base_url": "http://localhost:11434",
            "ollama_use_chat_model": True,
        }
    }
    return OllamaProvider(config)


@pytest.mark.asyncio
async def test_ollama_provider_invoke_and_chat():
    provider = _create_provider()

    # Prepare a dummy API response
    dummy_api = {"message": {"role": "assistant", "content": "dummy"}, "done": True}

    # Stub both sync and async HTTP calls used internally
    def _sync_post(*args, **kwargs):
        return httpx.Response(200, json=dummy_api)

    async def _async_post(*args, **kwargs):
        return httpx.Response(200, json=dummy_api)

    with (
        patch("httpx.Client.post", side_effect=_sync_post),
        patch("httpx.AsyncClient.post", new=AsyncMock(side_effect=_async_post)),
    ):
        # Test sync invoke (uses internal sync httpx fallback when asyncio.run not applicable)
        response = provider.invoke("Hello")
        assert response.message.content == "dummy"

        # Test async ainvoke
        response_text_async = await provider.ainvoke("Hello async")
        assert response_text_async.message.content == "dummy"

        # Test chat invocation
        messages = [
            {"role": "user", "content": "Hi there"},
            {"role": "assistant", "content": "Greetings"},
        ]
        chat_response = provider.chat(messages)
        assert chat_response.message.content == "dummy"

        # Async chat
        async_chat_response = await provider.achat(messages)
        assert async_chat_response.message.content == "dummy"


def test_ollama_provider_metadata():
    provider = _create_provider()
    assert provider.get_model_name() == "gemma3:12b-it-qat"
    assert provider.supports_tools() is False
    assert provider.supports_streaming() is True
    models = provider.get_available_models()
    assert isinstance(models, list)
    assert provider.get_model_name() in models
