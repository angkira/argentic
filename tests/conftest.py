import os
import sys
import pytest
import warnings
import types

# Add the parent directory to sys.path to ensure imports work properly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mark all tests in the tests directory as asyncio tests
pytest.importorskip("pytest_asyncio")


# Filter out specific warnings related to async mocks that we can't easily fix
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    # Suppress coroutine warnings from unittest.mock
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="coroutine '.*' was never awaited", category=RuntimeWarning
        )
        yield


# This allows imports like 'from argentic...' to work correctly

# -----------------------------------------------------------------------------
# Global stubs for heavyweight / external libraries so unit tests are offline.
# This module is imported by pytest before any test collection, ensuring that
# providers import these stubs instead of real libraries.
# -----------------------------------------------------------------------------

# --- google.api_core.exceptions ------------------------------------------------
_google_mod = types.ModuleType("google")
_api_core_mod = types.ModuleType("google.api_core")
_ex_mod = types.ModuleType("google.api_core.exceptions")
for _name in [
    "GoogleAPICallError",
    "ResourceExhausted",
    "InvalidArgument",
    "PermissionDenied",
    "InternalServerError",
    "DeadlineExceeded",
    "ServiceUnavailable",
    "BadRequest",
    "NotFound",
    "Unauthenticated",
    "Unknown",
]:
    setattr(_ex_mod, _name, type(_name, (Exception,), {}))
_api_core_mod.exceptions = _ex_mod  # type: ignore[attr-defined]
_google_mod.api_core = _api_core_mod  # type: ignore[attr-defined]

sys.modules.setdefault("google", _google_mod)
sys.modules.setdefault("google.api_core", _api_core_mod)
sys.modules.setdefault("google.api_core.exceptions", _ex_mod)

# --- langchain_google_genai ----------------------------------------------------
_lg_mod = types.ModuleType("langchain_google_genai")

from langchain_core.messages import AIMessage  # import here to avoid circular later


class _StubChatGoogleGenerativeAI:
    def __init__(self, **kwargs):
        pass

    def invoke(self, *_, **__):
        return AIMessage(content="dummy")

    async def ainvoke(self, *_, **__):
        return AIMessage(content="dummy")

    def bind_tools(self, *_):
        return self


_lg_mod.ChatGoogleGenerativeAI = _StubChatGoogleGenerativeAI

_chat_models = types.ModuleType("langchain_google_genai.chat_models")
_chat_models._response_to_result = lambda resp: resp  # type: ignore
_lg_mod.chat_models = _chat_models

sys.modules.setdefault("langchain_google_genai", _lg_mod)
sys.modules.setdefault("langchain_google_genai.chat_models", _chat_models)

# --- langchain_ollama ----------------------------------------------------------
_lo_mod = types.ModuleType("langchain_ollama")


class _StubLLM:
    def __init__(self, **kwargs):
        pass

    def invoke(self, *_, **__):
        return "dummy"

    async def ainvoke(self, *_, **__):
        return "dummy"


class _StubChatOllama(_StubLLM):
    pass


_lo_mod.OllamaLLM = _StubLLM
sys.modules.setdefault("langchain_ollama", _lo_mod)

_chat_ollama_mod = types.ModuleType("langchain_ollama.chat_models")
_chat_ollama_mod.ChatOllama = _StubChatOllama  # type: ignore
sys.modules.setdefault("langchain_ollama.chat_models", _chat_ollama_mod)

# --- langchain_community.llms.llamacpp -----------------------------------------
_lc_root = types.ModuleType("langchain_community")
_lc_llms = types.ModuleType("langchain_community.llms")
_lc_llamacpp = types.ModuleType("langchain_community.llms.llamacpp")


class _StubLlamaCpp:
    def __init__(self, **kwargs):
        pass

    def invoke(self, *_, **__):
        return "dummy"

    async def ainvoke(self, *_, **__):
        return "dummy"


_lc_llamacpp.LlamaCpp = _StubLlamaCpp
_lc_llms.llamacpp = _lc_llamacpp  # type: ignore[attr-defined]
_lc_root.llms = _lc_llms  # type: ignore[attr-defined]

sys.modules.setdefault("langchain_community", _lc_root)
sys.modules.setdefault("langchain_community.llms", _lc_llms)
sys.modules.setdefault("langchain_community.llms.llamacpp", _lc_llamacpp)

# -----------------------------------------------------------------------------
# Optional: stub llama_cpp to prevent ImportError‘s downstream
# -----------------------------------------------------------------------------
_llama_cpp_mod = types.ModuleType("llama_cpp")
sys.modules.setdefault("llama_cpp", _llama_cpp_mod)

# No pytest hooks/fixtures yet—stubs done at import time.
