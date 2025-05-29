#!/bin/bash

# Complete Release Workflow Script
# This script handles the entire release process:
# 1. Version bump with commitizen
# 2. Commit changes
# 3. Push to GitHub
# 4. Create GitHub release

set -e  # Exit on any error

# Source the shared virtual environment activation script
source "$(dirname "$0")/activate_venv.sh"

# Setup project environment
setup_project_env

echo "🚀 Starting complete release workflow..."

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo "❌ You have uncommitted changes. Please commit or stash them first."
    exit 1
fi

# Check for untracked files
if [ -n "$(git ls-files --others --exclude-standard)" ]; then
    echo "⚠️  You have untracked files. Consider adding them to .gitignore if needed."
    git ls-files --others --exclude-standard
    echo "Continue anyway? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "❌ Aborted."
        exit 1
    fi
fi

# Step 1: Version bump
echo "📈 Step 1: Checking for version bump..."
output=$(cz bump --changelog --yes --dry-run 2>&1)
bump_exit_code=$?

case $bump_exit_code in
  0)
    echo "✅ Version bump needed. Proceeding..."
    # Actually perform the bump
    cz bump --changelog --yes
    ;;
  16|19|21)
    echo "ℹ️  No version bump needed. Current version is up to date."
    echo "Creating GitHub release for current version..."
    
    # Skip to GitHub release creation
    ./bin/create_github_release.sh
    echo "✅ Release workflow completed!"
    exit 0
    ;;
  *)
    echo "❌ Commitizen error:"
    echo "$output"
    exit 1
    ;;
esac

# Step 2: Commit the version changes
echo "💾 Step 2: Committing version changes..."
git add pyproject.toml CHANGELOG.md
git commit -m "chore: release $(grep 'version = ' pyproject.toml | cut -d'"' -f2)"

# Step 3: Push everything to GitHub
echo "📤 Step 3: Pushing to GitHub..."
git push --follow-tags

# Step 4: Create GitHub release
echo "🎯 Step 4: Creating GitHub release..."
./bin/create_github_release.sh

echo "🎉 Complete release workflow finished successfully!"
echo ""
echo "📋 What was done:"
echo "  ✅ Version bumped based on conventional commits"
echo "  ✅ CHANGELOG.md updated"
echo "  ✅ Changes committed and pushed to GitHub"
echo "  ✅ GitHub release created"
echo ""
echo "🌐 Check your release at: https://github.com/$(gh repo view --json owner,name -q '.owner.login + "/" + .name')/releases" 