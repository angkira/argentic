#!/bin/bash
set -eo pipefail

# This script automates the version bumping and push process.
# 1. It checks if there are any new commits to push.
# 2. It runs commitizen to bump the version and create a bump commit if needed.
#    - If no conventional keywords (feat, fix) are found, it forces a PATCH bump.
#    - If there are no new commits since the last tag, it does nothing.
# 3. It pushes all changes and tags to the remote repository.

# --- VENV SETUP ---
# Source the shared virtual environment activation script
source "$(dirname "$0")/activate_venv.sh"
# Setup project environment (activate venv, change directory, set PYTHONPATH)
setup_project_env
# --- END VENV SETUP ---


echo "Checking for new commits to push..."
# Fetch the latest state from the remote to ensure we're comparing against the latest version
git fetch

LOCAL_SHA=$(git rev-parse @)
REMOTE_SHA=$(git rev-parse @{u} 2>/dev/null || echo "")

# If remote SHA is empty, it's a new branch, so we definitely need to push.
if [ -n "$REMOTE_SHA" ] && [ "$LOCAL_SHA" == "$REMOTE_SHA" ]; then
    echo "✅ Your branch is up to date. Nothing to push."
    exit 0
fi

echo "🚀 Attempting to bump version..."
# Run cz bump and capture its output and exit code.
# Commitizen will check for commits since the last tag.
output=$(cz bump --changelog --yes 2>&1)
bump_exit_code=$?

case $bump_exit_code in
  0)
    # Bump successful based on keywords like feat: or fix:
    echo "✅ Version bump successful."
    echo "$output"
    ;;
  16) 
    # Exit code 16: NO_COMMITS_FOUND (since last tag)
    # This is not an error; it just means there's nothing to bump.
    echo "ℹ️ No new commits found since the last tag. Nothing to bump."
    ;;
  19|21)
    # Exit code 19: NO_BUMP (keywords found, but they don't trigger a bump, e.g. 'chore')
    # Exit code 21: NO_COMMITS_TO_BUMP (the commits are not eligible)
    echo "ℹ️ No bump-worthy keywords found. Forcing a PATCH bump by default..."
    patch_output=$(cz bump --increment PATCH --yes 2>&1)
    patch_exit_code=$?
    if [ $patch_exit_code -ne 0 ]; then
        echo "❌ Default PATCH bump failed:" >&2
        echo "$patch_output" >&2
        exit 1
    fi
    echo "✅ Default PATCH bump successful."
    echo "$patch_output"
    ;;
  *)
    # Any other error is a real failure
    echo "❌ An unexpected error occurred with commitizen (code: $bump_exit_code):" >&2
    echo "$output" >&2
    exit 1
    ;;
esac

echo "📡 Pushing all changes and tags to the remote..."
git push --follow-tags

echo "🎉 Shipment successful!" 