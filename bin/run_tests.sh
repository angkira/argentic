#!/bin/bash

# Install test dependencies if needed
python -m pip install -e ".[dev]"

# Set Python path to include the src directory
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the tests
python -m pytest "$@" 