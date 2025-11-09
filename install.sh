#!/bin/bash

# This script installs the project and its core development dependencies.
# It assumes you are running it from the root of the project
# and that your PYTHONPATH will allow importing the local 'src' directory.

# Setup Python virtual environment
echo "Setting up Python virtual environment..."
python -m venv .venv
source .venv/bin/activate

# Install package in development mode
echo "Installing Argentic package in development mode..."
# pip install -e .
uv sync --extra dev

echo "Project and dev dependencies installed successfully using uv."

echo "Installation complete!"
echo "To activate the environment, run: source .venv/bin/activate" 