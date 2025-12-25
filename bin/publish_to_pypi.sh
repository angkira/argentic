#!/bin/bash

# PyPI Publishing Script
# Builds and publishes package to PyPI using credentials from .env
# Assumes version is already set in pyproject.toml

set -e

# Source environment variables
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Check for PyPI token
if [ -z "$PY_PI_TOKEN" ]; then
    echo "❌ PY_PI_TOKEN not found in .env"
    exit 1
fi

# Get current version
CURRENT_VERSION=$(grep 'version = ' pyproject.toml | cut -d'"' -f2)
echo "📦 Publishing version $CURRENT_VERSION to PyPI..."

# Create git tag if it doesn't exist
if git rev-parse "$CURRENT_VERSION" >/dev/null 2>&1; then
    echo "🏷️  Tag $CURRENT_VERSION already exists"
else
    echo "🏷️  Creating tag $CURRENT_VERSION..."
    git tag -a "$CURRENT_VERSION" -m "$CURRENT_VERSION"
    echo "✅ Tag created successfully"
fi

# Clean old dist files
echo "🧹 Cleaning old builds..."
rm -rf dist build *.egg-info

# Build package
echo "🔨 Building package..."
uv run python -m build

# Check if version already exists on PyPI
echo "🔍 Checking if version exists on PyPI..."
if pip index versions argentic 2>/dev/null | grep -q "$CURRENT_VERSION"; then
    echo "⚠️  Version $CURRENT_VERSION already exists on PyPI!"
    echo "Continue anyway? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Publish to PyPI
echo "🚀 Publishing to PyPI..."
uv run python -m twine upload dist/* \
    --username __token__ \
    --password "$PY_PI_TOKEN" \
    --non-interactive

echo "✅ Successfully published argentic $CURRENT_VERSION to PyPI!"
echo "🌐 Check: https://pypi.org/project/argentic/$CURRENT_VERSION/"

# Ask to push tag and commit
echo ""
echo "Push tag and commits to remote? (y/N)"
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    echo "🚀 Pushing to remote..."
    git push origin main
    git push origin "$CURRENT_VERSION"
    echo "✅ Pushed to remote successfully"
fi
