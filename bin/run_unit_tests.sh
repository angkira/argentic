#!/bin/bash

# Source the shared virtual environment activation script
source "$(dirname "$0")/activate_venv.sh"

# Setup project environment (activate venv, change directory, set PYTHONPATH)
setup_project_env

# Install test dependencies if needed
# python -m pip install -e ".[dev,kafka,redis,rabbitmq]"
# uv sync --extra dev --extra kafka --extra redis --extra rabbitmq

# Run unit tests (exclude tests with e2e marker)
python -m pytest tests/core/messager/unit -m "not e2e" "$@" 