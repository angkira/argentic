#!/bin/bash

# Install test dependencies if needed
python -m pip install -e ".[dev]"

# Set Python path to include the src directory
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run unit tests (exclude tests with e2e marker)
python -m pytest tests/core/messager/unit -m "not e2e" "$@" 