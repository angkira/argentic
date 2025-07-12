#!/bin/bash

# Source the shared virtual environment activation script
source "$(dirname "$0")/activate_venv.sh"

# Setup project environment (activate venv, change directory, set PYTHONPATH)
setup_project_env

# Install the package in editable mode along with all test dependencies
echo "🔧 Installing package in editable mode with all test extras..."
uv pip install -e ".[dev,kafka,redis,rabbitmq]"

# Run unit tests (exclude tests with e2e marker)
echo "🚀 Running unit tests..."
python -m pytest tests/core/messager/unit -m "not e2e" "$@" 