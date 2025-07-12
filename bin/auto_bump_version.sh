#!/usr/bin/env bash

# Source the shared virtual environment activation script
# Note: Using bash instead of sh for better compatibility with sourcing
source "$(dirname "$0")/activate_venv.sh"

# Setup project environment (activate venv, change directory, set PYTHONPATH)
setup_project_env

# This script is run by pre-commit in a pre-push hook.
# The pre-commit framework provides the range of commits being pushed
# in the PRE_COMMIT_FROM_REF and PRE_COMMIT_TO_REF environment variables.

# Get all commit messages in the range of commits being pushed.
# We use '|| true' to avoid an error if PRE_COMMIT_FROM_REF is a new branch root (000000...).
COMMIT_MSGS=$(git log --pretty=%B "$PRE_COMMIT_FROM_REF..$PRE_COMMIT_TO_REF" || true)

# Check if there are any non-bump commits in the push range.
# We use grep -v to filter out the bump commits and then check if anything remains.
NON_BUMP_COMMITS=$(echo "$COMMIT_MSGS" | grep -v -E '^bump:' | grep -v -E '^\s*$' || true)

# If NON_BUMP_COMMITS is empty, it means all commits being pushed are version bumps.
# In this case, we should allow the push to proceed without trying to bump again.
if [ -z "$NON_BUMP_COMMITS" ]; then
  echo "No new non-bump commits to process. Allowing push to proceed."
  exit 0
fi

echo "Attempting to bump version with commitizen..."
# Run cz bump non-interactively and capture its output and exit code.
# We expect cz bump to create a tag automatically if a bump occurs.
output=$(cz bump --changelog --yes 2>&1)
bump_exit_code=$?

case $bump_exit_code in
  0)
    # Bump successful
    echo "Version bump successful."
    echo "$output" # Show what cz bump did (new version, etc.)
    echo ""
    echo "IMPORTANT: Files (like pyproject.toml, CHANGELOG.md) have been modified and a new tag created."
    echo "These changes are NOT included in the current push."
    echo "Please:"
    echo "  1. Stage the modified files (e.g., 'git add pyproject.toml CHANGELOG.md')."
    echo "  2. Commit these changes (e.g., 'git commit -m \"chore: bump version\"')."
    echo "  3. Push your commits again."
    echo "  4. Push the new tags (e.g., 'git push --tags' or 'git push --follow-tags')."
    echo ""
    echo "💡 TIP: After pushing, you can create a GitHub release with:"
    echo "   ./bin/create_github_release.sh"
    echo ""
    echo "Aborting current push to allow you to commit version changes."
    exit 1 # Abort the current push
    ;;
  16|19|21)
    # Exit code 16: NO_COMMITS_FOUND (No commits found since last release)
    # Exit code 19: NO_BUMP (No commits meet the criteria to bump the version)
    # Exit code 21: NO_COMMITS_TO_BUMP (The commits found are not eligible to be bumped)
    echo "No bump-worthy keywords found. Forcing a PATCH bump by default..."
    
    # Re-run the bump, but force a patch increment without generating a changelog
    output_patch=$(cz bump --increment PATCH --yes 2>&1)
    patch_bump_exit_code=$?

    if [ $patch_bump_exit_code -ne 0 ]; then
        echo "Default PATCH bump failed with exit code $patch_bump_exit_code:"
        echo "$output_patch"
        exit 1 # Abort push on failure
    fi

    # If patch bump was successful, show message and abort push
    echo "Default PATCH bump successful."
    echo "$output_patch"
    echo ""
    echo "IMPORTANT: Files (like pyproject.toml, CHANGELOG.md) have been modified and a new tag created."
    echo "These changes are NOT included in the current push."
    echo "Please:"
    echo "  1. Stage the modified files (e.g., 'git add pyproject.toml CHANGELOG.md')."
    echo "  2. Commit these changes (e.g., 'git commit -m \"chore: bump version\"')."
    echo "  3. Push your commits again."
    echo "  4. Push the new tags (e.g., 'git push --tags' or 'git push --follow-tags')."
    echo ""
    echo "Aborting current push to allow you to commit version changes."
    exit 1 # Abort the current push
    ;;
  17)
    # Exit code 17: NO_VERSION_SPECIFIED (The project has no version specified)
    # This might happen on a brand new project before the first version is set.
    # Or if pyproject.toml version is somehow missing.
    echo "Commitizen error: No version specified in the project."
    echo "$output"
    echo "Aborting push. Please ensure your project has an initial version set in pyproject.toml."
    exit 1 # Abort push
    ;;
  *)
    # Other error
    echo "cz bump command failed with an unexpected error (exit code $bump_exit_code):"
    echo "$output"
    echo "Aborting push."
    exit 1 # Abort push
    ;;
esac 