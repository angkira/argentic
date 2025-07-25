import subprocess
import types

import pytest
from langchain_core.messages import BaseMessage

# Stubs moved to tests/conftest.py to ensure global availability.
# -----------------------------------------------------------------------------
# Now import providers
# -----------------------------------------------------------------------------
from argentic.core.llm.providers.google_gemini import GoogleGeminiProvider
from argentic.core.llm.providers.ollama import OllamaProvider
from argentic.core.llm.providers.llama_cpp_cli import LlamaCppCLIProvider
from argentic.core.llm.providers.llama_cpp_langchain import LlamaCppLangchainProvider
from argentic.core.llm.providers.llama_cpp_server import LlamaCppServerProvider
from argentic.core.llm.providers.mock import MockLLMProvider


# -----------------------------------------------------------------------------
# Helper fixtures / utilities
# -----------------------------------------------------------------------------


def _assert_base_message(msg: BaseMessage):
    """Common assertion helper."""
    assert isinstance(msg, BaseMessage)
    assert msg.content, "Message content should not be empty"


# -----------------------------------------------------------------------------
# Tests for individual providers (pure unit, no external calls)
# -----------------------------------------------------------------------------


def test_google_gemini_provider_invoke():
    provider = GoogleGeminiProvider({"google_gemini_api_key": "dummy"})
    result = provider.invoke("Hello, world!")
    _assert_base_message(result)


def test_ollama_provider_invoke():
    provider = OllamaProvider({"ollama_model_name": "dummy", "ollama_base_url": "http://x"})
    result = provider.invoke("Ping")
    _assert_base_message(result)


def test_llama_cpp_cli_provider_invoke(monkeypatch):
    # Ensure binary/model path checks pass
    monkeypatch.setattr("os.path.isfile", lambda *_: True)

    # Stub subprocess.run so no external process is spawned
    def _stub_run(cmd, capture_output, text, check, encoding):  # noqa: D401
        _cp = types.SimpleNamespace()
        _cp.stdout = "LLM completed output"
        _cp.stderr = ""
        _cp.returncode = 0
        _cp.cmd = cmd
        return _cp

    monkeypatch.setattr(subprocess, "run", _stub_run)

    provider = LlamaCppCLIProvider(
        {
            "llm": {
                "llama_cpp_cli_binary": "/bin/llama",
                "llama_cpp_cli_model_path": "/models/model.gguf",
            }
        }
    )
    result = provider.invoke("Hi")
    _assert_base_message(result)


def test_llama_cpp_langchain_provider_invoke():
    provider = LlamaCppLangchainProvider({"llm": {"llama_cpp_model_path": "/models/model.gguf"}})
    result = provider.invoke("Test prompt")
    _assert_base_message(result)


def test_llama_cpp_server_provider_invoke(monkeypatch):
    provider = LlamaCppServerProvider(
        {
            "llama_cpp_server_host": "localhost",
            "llama_cpp_server_port": 8000,
            "llama_cpp_server_auto_start": False,
            "llama_cpp_model_path": "/models/model.gguf",
        }
    )

    async def _stub_make_request(self, endpoint, payload):  # noqa: D401
        return {"content": "stub-response"}

    monkeypatch.setattr(LlamaCppServerProvider, "_make_request", _stub_make_request)

    result = provider.invoke("Hello")
    _assert_base_message(result)


def test_mock_llm_provider_basic():
    provider = MockLLMProvider({})
    result = provider.invoke("Hi there")
    _assert_base_message(result)
    assert provider.call_count == 1
