#!/bin/bash

# Release Workflow Script
# Assumes version is already bumped via post-commit hook
# This script:
# 1. Creates git tag for current version
# 2. Builds and publishes to PyPI
# 3. Pushes tag to GitHub
# 4. Creates GitHub release

set -e

# Check for .env file
if [ ! -f .env ]; then
    echo "❌ .env file not found. Please create it with PY_PI_TOKEN"
    exit 1
fi

# Source environment variables
set -a
source .env
set +a

# Check for PyPI token
if [ -z "$PY_PI_TOKEN" ]; then
    echo "❌ PY_PI_TOKEN not found in .env"
    exit 1
fi

echo "🚀 Starting release workflow..."

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo "❌ You have uncommitted changes. Please commit them first."
    exit 1
fi

# Get current version from pyproject.toml
CURRENT_VERSION=$(grep 'version = ' pyproject.toml | cut -d'"' -f2)
echo "📦 Current version: $CURRENT_VERSION"

# Check if tag already exists
if git rev-parse "$CURRENT_VERSION" >/dev/null 2>&1; then
    echo "⚠️  Tag $CURRENT_VERSION already exists locally."
    echo "Delete it? (y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        git tag -d "$CURRENT_VERSION"
    else
        echo "❌ Aborted."
        exit 1
    fi
fi

# Step 1: Create git tag
echo "🏷️  Step 1: Creating git tag..."
git tag "$CURRENT_VERSION"

# Step 2: Build package
echo "🔨 Step 2: Building package..."
rm -rf dist build *.egg-info
uv run python -m build

# Step 3: Check if version exists on PyPI
echo "🔍 Step 3: Checking PyPI..."
if pip index versions argentic 2>/dev/null | grep -q "$CURRENT_VERSION"; then
    echo "⚠️  Version $CURRENT_VERSION already exists on PyPI!"
    echo "Continue anyway? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "❌ Aborted."
        git tag -d "$CURRENT_VERSION"
        exit 1
    fi
else
    echo "✅ Version $CURRENT_VERSION not yet on PyPI"
fi

# Step 4: Publish to PyPI
echo "🚀 Step 4: Publishing to PyPI..."
uv run python -m twine upload dist/* \
    --username __token__ \
    --password "$PY_PI_TOKEN" \
    --non-interactive

echo "✅ Published to PyPI!"

# Step 5: Push tag to GitHub
echo "📤 Step 5: Pushing tag to GitHub..."
git push origin "$CURRENT_VERSION"

# Step 6: Create GitHub release
echo "🎯 Step 6: Creating GitHub release..."

# Extract release notes from CHANGELOG.md
TEMP_NOTES=$(mktemp)
if [ -f "CHANGELOG.md" ]; then
    awk "/^## $CURRENT_VERSION/{flag=1; next} /^## [0-9]/{flag=0} flag" CHANGELOG.md > "$TEMP_NOTES"
    
    if [ ! -s "$TEMP_NOTES" ]; then
        echo "Release $CURRENT_VERSION" > "$TEMP_NOTES"
    fi
else
    echo "Release $CURRENT_VERSION" > "$TEMP_NOTES"
fi

# Delete existing release if exists
gh release delete "$CURRENT_VERSION" --yes 2>/dev/null || true

# Create release
gh release create "$CURRENT_VERSION" \
    --title "Release $CURRENT_VERSION" \
    --notes-file "$TEMP_NOTES"

rm "$TEMP_NOTES"

echo ""
echo "🎉 Release $CURRENT_VERSION completed successfully!"
echo ""
echo "📋 What was done:"
echo "  ✅ Git tag created: $CURRENT_VERSION"
echo "  ✅ Published to PyPI"
echo "  ✅ Tag pushed to GitHub"
echo "  ✅ GitHub release created"
echo ""
echo "🌐 Links:"
echo "  PyPI: https://pypi.org/project/argentic/$CURRENT_VERSION/"
echo "  GitHub: https://github.com/$(gh repo view --json owner,name -q '.owner.login + "/" + .name')/releases/tag/$CURRENT_VERSION"
