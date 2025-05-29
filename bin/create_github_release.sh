#!/bin/bash

# GitHub Release Creation Script
# This script creates a GitHub release from the latest git tag using the changelog

set -e  # Exit on any error

# Source the shared virtual environment activation script
source "$(dirname "$0")/activate_venv.sh"

# Setup project environment
setup_project_env

echo "🚀 Creating GitHub Release..."

# Get the latest version tag (only semantic version tags)
LATEST_TAG=$(git tag --sort=-version:refname | grep -E '^[0-9]+\.[0-9]+\.[0-9]+' | head -n 1)

if [ -z "$LATEST_TAG" ]; then
    echo "❌ No semantic version tags found. Please create a version tag first (e.g., 1.0.0)."
    exit 1
fi

echo "📦 Latest version tag: $LATEST_TAG"

# Check if release already exists
if gh release view "$LATEST_TAG" >/dev/null 2>&1; then
    echo "⚠️  Release $LATEST_TAG already exists on GitHub."
    echo "Do you want to delete and recreate it? (y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "🗑️  Deleting existing release..."
        gh release delete "$LATEST_TAG" --yes
    else
        echo "❌ Aborted."
        exit 1
    fi
fi

# Extract release notes for this version from CHANGELOG.md
if [ -f "CHANGELOG.md" ]; then
    echo "📝 Extracting release notes from CHANGELOG.md..."
    
    # Create temporary file with release notes for this version only
    TEMP_NOTES=$(mktemp)
    
    # Extract content between this version and the next older version
    awk "/^## $LATEST_TAG/{flag=1; next} /^## [0-9]/{flag=0} flag" CHANGELOG.md > "$TEMP_NOTES"
    
    # If no specific content found, use the full changelog
    if [ ! -s "$TEMP_NOTES" ]; then
        echo "⚠️  Could not extract specific release notes for $LATEST_TAG, using full changelog"
        cp CHANGELOG.md "$TEMP_NOTES"
    fi
    
    NOTES_FILE="$TEMP_NOTES"
else
    echo "⚠️  No CHANGELOG.md found. Creating release without detailed notes."
    NOTES_FILE=""
fi

# Create the release
echo "🎯 Creating GitHub release for $LATEST_TAG..."

if [ -n "$NOTES_FILE" ]; then
    gh release create "$LATEST_TAG" \
        --title "Release $LATEST_TAG" \
        --notes-file "$NOTES_FILE"
else
    gh release create "$LATEST_TAG" \
        --title "Release $LATEST_TAG" \
        --notes "Release $LATEST_TAG - See git log for changes since previous release."
fi

# Clean up temporary file
if [ -f "$TEMP_NOTES" ]; then
    rm "$TEMP_NOTES"
fi

echo "✅ GitHub release created successfully!"
echo "🌐 URL: https://github.com/$(gh repo view --json owner,name -q '.owner.login + "/" + .name')/releases/tag/$LATEST_TAG" 