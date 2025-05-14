#!/bin/bash

# Install test dependencies if needed
# python -m pip install -e ".[dev,kafka,redis,rabbitmq]"
uv sync --extra dev --extra kafka --extra redis --extra rabbitmq

# Set Python path to include the src directory
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the tests
python -m pytest "$@" 