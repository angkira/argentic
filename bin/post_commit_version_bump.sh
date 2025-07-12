#!/usr/bin/env bash
set -eo pipefail

# --- Activate project environment (venv + PROJECT_ROOT) ---
source "$(dirname "$0")/activate_venv.sh"
setup_project_env
# ---------------------------------------------------------

# Get last commit message (the commit that just finished)
COMMIT_MSG="$(git log -1 --pretty=%B)"

# Skip if last commit already a bump commit
if [[ "$COMMIT_MSG" =~ ^bump: ]]; then
    exit 0
fi

# Skip if this commit already contains changes to version file to avoid infinite loop
if git diff-tree --no-commit-id --name-only -r HEAD | grep -q '^pyproject.toml$'; then
    echo "pyproject.toml already changed in this commit – skipping bump to avoid loop."
    exit 0
fi

# Determine increment type from commit message
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

echo "📦 Post-commit bump: $INCREMENT"

# Perform bump (files only, no tag)
# Exit codes: 0 (success with changes), 16/19/21 etc -> we ignore and force patch if needed
if ! cz bump --files-only --increment "$INCREMENT" --yes >/dev/null 2>&1; then
    echo "cz bump reported no changes, forcing PATCH bump"
    cz bump --files-only --increment PATCH --yes
fi

# Stage modified files
git add -u

# If nothing staged (already up-to-date) – skip amend
if git diff --cached --quiet; then
    echo "No version change detected – skipping amend"
    exit 0
fi

# Amend the previous commit to include version bump
GIT_COMMITTER_DATE="$(git log -1 --pretty=%cI)" git commit --amend --no-edit --no-verify --date "$(git log -1 --pretty=%aI)"

echo "✅ Version bumped and commit amended"

exit 0 