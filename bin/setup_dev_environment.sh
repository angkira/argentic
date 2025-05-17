#!/bin/sh

# This script sets up the development environment for this project.

echo "🚀 Starting development environment setup..."

# --- Check for uv ---
echo "\n🔎 Checking for uv (Python package manager)..."
if ! command -v uv &> /dev/null
then
    echo "❌ Error: uv command not found."
    echo "Please install uv first. See: https://github.com/astral-sh/uv"
    exit 1
fi
echo "✅ uv found."

# --- Python Dependencies ---
echo "\n🐍 Installing Python dependencies (including pre-commit and commitizen)..."
echo "(This may take a moment)"
# Assuming you want to install the package as editable (-e) with all dev dependencies.
# If you have different groups or no extras, adjust accordingly.
if uv pip install -e ".[dev]"
_PY_DEPS_INSTALLED=true
then
    echo "✅ Python dependencies installed successfully."
else
    echo "❌ Error: Failed to install Python dependencies with uv."
    echo "Please check the output above for errors."
    _PY_DEPS_INSTALLED=false
    # exit 1 # Optionally exit here if essential
fi

# --- Pre-commit Hooks ---
if [ "$_PY_DEPS_INSTALLED" = true ] && command -v pre-commit &> /dev/null
then
    echo "\n🔨 Setting up pre-commit hooks..."
    if pre-commit install --hook-type pre-push
    then
        echo "✅ Pre-push hooks installed successfully."
    else
        echo "⚠️ Warning: Failed to install pre-commit hooks automatically."
        echo "You might need to run 'pre-commit install --hook-type pre-push' manually after ensuring pre-commit is correctly installed and in your PATH."
    fi
else
    echo "\n⏭️  Skipping pre-commit hook setup because Python dependencies (including pre-commit) did not install successfully or pre-commit command is not available."
fi

# --- Final Instructions ---
echo "\n🎉 Setup script finished!"
echo "---------------------------------------------------"
echo "Key development workflows:"
echo "  ➡️  For making commits: Use 'cz commit' or 'git commit' (if you set up a commit-msg hook for commitizen later)."
echo "       'cz commit' will guide you through conventional commit messages."
echo "  ➡️  Before pushing:"
echo "       1. Unit tests will run automatically."
echo "       2. If tests pass, a version bump will be attempted via commitizen."
echo "          - If a bump occurs, the push will be aborted. Follow the instructions to commit the version change and push again (including tags)."
echo "          - If no bump is needed, your push will proceed."
echo "  ➡️  Remember to activate your Python virtual environment if you haven't already."
echo "---------------------------------------------------"

if [ "$_PY_DEPS_INSTALLED" = false ]
then
    echo "🔴 Some steps failed. Please review the output above and address any errors."
    exit 1
fi

exit 0 