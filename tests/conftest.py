import os
import sys
import pytest

# Add the parent directory to sys.path to ensure imports work properly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mark all tests in the tests directory as asyncio tests
pytest.importorskip("pytest_asyncio")

# This allows imports like 'from src.core...' to work correctly
