#!/bin/bash

# Setup Python virtual environment
echo "Setting up Python virtual environment..."
python -m venv .venv
source .venv/bin/activate

# Install package in development mode
echo "Installing Argentic package in development mode..."
pip install -e .

echo "Installation complete!"
echo "To activate the environment, run: source .venv/bin/activate" 