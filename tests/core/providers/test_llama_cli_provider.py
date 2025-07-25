import subprocess
import types
import sys

import pytest


# Stub to ensure any external call to subprocess.run behaves deterministically
class _DummyCompletedProcess:
    def __init__(self, stdout: str = ""):
        self.stdout = stdout
        self.returncode = 0
        self.stderr = ""
        self.cmd = []


@pytest.fixture(autouse=True)
def _patch_subprocess_run(monkeypatch):
    def _fake_run(cmd, capture_output, text, check, encoding):  # noqa: D401
        prompt_index = cmd.index("-p") + 1 if "-p" in cmd else len(cmd) - 1
        prompt = cmd[prompt_index]
        # Echo prompt followed by space then mock response – mimics llama.cpp behaviour
        return _DummyCompletedProcess(stdout=f"{prompt} mock-response")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    yield


@pytest.fixture(autouse=True)
def _patch_os_path_isfile(monkeypatch):
    """Treat any path as existing to bypass file checks in provider."""
    monkeypatch.setattr("os.path.isfile", lambda path: True)
    yield


from argentic.core.llm.providers.llama_cpp_cli import LlamaCppCLIProvider  # noqa: E402


def _create_provider() -> LlamaCppCLIProvider:
    config = {
        "llm": {
            "llama_cpp_cli_binary": "/usr/bin/llama-fake-binary",
            "llama_cpp_cli_model_path": "/models/dummy.gguf",
        }
    }
    return LlamaCppCLIProvider(config)


def test_llama_cli_provider_invoke_and_chat():
    provider = _create_provider()

    response = provider.invoke("Hello")
    assert response.content and "mock-response" in response.content

    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Yo"},
    ]
    chat_resp = provider.chat(messages)
    assert hasattr(chat_resp, "content")


def test_llama_cli_provider_metadata():
    provider = _create_provider()
    assert provider.supports_tools() is False
    assert provider.supports_streaming() is False
    assert provider.get_available_models() == [provider.get_model_name()]
