import os
import sys
import pytest
import warnings

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


# This allows imports like 'from src.core...' to work correctly
