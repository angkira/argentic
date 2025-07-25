import types
import sys
import asyncio

import pytest


@pytest.fixture(autouse=True)
def _patch_os_path_isfile(monkeypatch):
    # Treat any file path as existing
    monkeypatch.setattr("os.path.isfile", lambda path: True)
    yield


# Stub Langchain LlamaCpp (optional) so import doesn't fail when use_langchain True
@pytest.fixture(autouse=True)
def _stub_langchain_llama(monkeypatch):
    dummy_mod = types.ModuleType("langchain_community.llms.llamacpp")

    class _DummyLlama:
        def __init__(self, *args, **kwargs):
            pass

        def invoke(self, *args, **kwargs):
            return "dummy"

        async def ainvoke(self, *args, **kwargs):
            return "dummy"

    dummy_mod.LlamaCpp = _DummyLlama  # type: ignore
    sys.modules["langchain_community.llms.llamacpp"] = dummy_mod
    yield
    sys.modules.pop("langchain_community.llms.llamacpp", None)


from argentic.core.llm.providers.llama_cpp_server import LlamaCppServerProvider  # noqa: E402


def _create_provider() -> LlamaCppServerProvider:
    config = {
        "llama_cpp_server_host": "localhost",
        "llama_cpp_server_port": 8000,
        "llama_cpp_server_auto_start": False,
    }
    return LlamaCppServerProvider(config)


@pytest.mark.asyncio
async def test_llama_server_provider_invoke_and_chat(monkeypatch):
    provider = _create_provider()

    # Patch _make_request to bypass actual HTTP requests
    async def _fake_make_request(endpoint: str, payload):
        if endpoint.startswith("/completion"):
            # mimic /completion response
            return {"content": "dummy"}
        else:  # /v1/chat/completions
            return {"choices": [{"message": {"content": "dummy"}}]}

    monkeypatch.setattr(provider, "_make_request", _fake_make_request)

    resp = provider.invoke("Hello")
    assert resp.content == "dummy"

    messages = [{"role": "user", "content": "Hi"}]
    chat_resp = provider.chat(messages)
    assert chat_resp.content == "dummy"

    # Async counterparts
    async_resp = await provider.ainvoke("Hi async")
    assert async_resp.content == "dummy"

    async_chat_resp = await provider.achat(messages)
    assert async_chat_resp.content == "dummy"


def test_llama_server_metadata():
    provider = _create_provider()
    assert provider.supports_tools() is False
    assert provider.get_available_models() == [provider.get_model_name()]
