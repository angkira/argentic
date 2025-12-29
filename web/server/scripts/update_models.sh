#!/bin/bash
# Script to update TypeScript models from Python Pydantic models

set -e

cd "$(dirname "$0")/.."

echo "🔄 Generating TypeScript models from Python..."
python scripts/generate_typescript_models.py

echo "✅ Models updated successfully!"
