#!/bin/bash
# Simple wrapper script to generate TypeScript models
# Uses uv run to handle dependencies automatically

cd "$(dirname "$0")"
echo "🔄 Generating TypeScript models from Python..."
uv run python scripts/generate_typescript_models.py
