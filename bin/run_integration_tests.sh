#!/bin/bash

# Install the package in editable mode along with all test dependencies
echo "🔧 Installing package in editable mode with all test extras..."
uv pip install -e ".[dev,kafka,redis,rabbitmq]"

# Run integration tests
echo "🚀 Running integration tests..."
uv run pytest tests/core/messager/test_messager_integration.py "$@" 
uv run pytest tests/integration "$@" 