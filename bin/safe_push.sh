#!/bin/bash

# Safe push script for Cursor IDE
# This script runs pre-push hooks manually and then pushes

set -e  # Exit on any error

# Source the shared virtual environment activation script
source "$(dirname "$0")/activate_venv.sh"

# Setup project environment (activate venv, change directory, set PYTHONPATH)
setup_project_env

echo "🔍 Running pre-push checks..."

# Run unit tests first
if [ -f "./bin/run_unit_tests.sh" ]; then
    echo "Running unit tests..."
    ./bin/run_unit_tests.sh
    if [ $? -ne 0 ]; then
        echo "❌ Unit tests failed. Aborting push."
        exit 1
    fi
    echo "✅ Unit tests passed!"
else
    echo "⚠️  Unit test script not found, skipping..."
fi

# Run version bump if configured
if [ -f "./bin/auto_bump_version.sh" ]; then
    echo "Running version bump..."
    ./bin/auto_bump_version.sh
    if [ $? -ne 0 ]; then
        echo "❌ Version bump failed. Aborting push."
        exit 1
    fi
    echo "✅ Version bump completed!"
else
    echo "⚠️  Version bump script not found, skipping..."
fi

# Get current branch
BRANCH=$(git rev-parse --abbrev-ref HEAD)

# Push with all arguments passed to this script
echo "🚀 Pushing to origin/$BRANCH..."
git push "$@"

echo "✅ Push completed successfully!" 