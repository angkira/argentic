#!/usr/bin/env bash
set -eo pipefail

# This hook runs at commit-msg stage. Argument 1 is the path to the commit
# message file provided by Git (and pre-commit framework).
MSG_FILE="$1"
if [[ -z "$MSG_FILE" ]]; then
  echo "No commit message file path provided to commit-msg hook." >&2
  exit 1
fi

# --- Activate project environment (venv + PROJECT_ROOT) ---
source "$(dirname "$0")/activate_venv.sh"
setup_project_env
# ---------------------------------------------------------

COMMIT_MSG="$(cat "$MSG_FILE")"

# Skip processing for merge commits or already-bumped commits
if [[ "$COMMIT_MSG" =~ ^Merge\  ]]; then
  exit 0
fi
if [[ "$COMMIT_MSG" =~ ^bump: ]]; then
  echo "Commit message already indicates a version bump – skipping bump logic." >&2
  exit 0
fi

# Decide the increment type based on Conventional Commits keywords
INCREMENT="PATCH" # default
if echo "$COMMIT_MSG" | grep -qE "BREAKING CHANGE|!:"; then
  INCREMENT="MAJOR"
elif [[ "$COMMIT_MSG" =~ ^feat ]]; then
  INCREMENT="MINOR"
elif [[ "$COMMIT_MSG" =~ ^fix ]]; then
  INCREMENT="PATCH"
else
  INCREMENT="PATCH"
fi

echo "📦 Detected increment: $INCREMENT"

# Run Commitizen to update version files only (no commit / no tag)
if ! cz bump --files-only --increment "$INCREMENT" --yes; then
  echo "❌ Commitizen bump failed" >&2
  exit 1
fi

# Stage any files changed by the bump so they become part of this commit
# Usually this is at least pyproject.toml and possibly CHANGELOG.md
# `git add -u` stages all modified tracked files.
git add -u

echo "✅ Version bumped and changes staged."

exit 0 