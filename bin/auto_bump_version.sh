#!/bin/sh

# Source the shared virtual environment activation script
# Note: Using bash instead of sh for better compatibility with sourcing
. "$(dirname "$0")/activate_venv.sh"

# Setup project environment (activate venv, change directory, set PYTHONPATH)
setup_project_env

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
  16|19)
    # Exit code 16: NO_COMMITS_FOUND (No commits found since last release)
    # Exit code 19: NO_BUMP (No commits meet the criteria to bump the version)
    echo "No relevant commits found for a version bump, or no bump needed based on commits."
    echo "$output"
    echo "Proceeding with push."
    exit 0 # Allow push
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