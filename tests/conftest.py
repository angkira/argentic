import os
import sys
import types
import warnings

import pytest

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

# -----------------------------------------------------------------------------
# Optional: stub llama_cpp to prevent ImportError's downstream
# -----------------------------------------------------------------------------
_llama_cpp_mod = types.ModuleType("llama_cpp")
sys.modules.setdefault("llama_cpp", _llama_cpp_mod)

# No pytest hooks/fixtures yet—stubs done at import time.
