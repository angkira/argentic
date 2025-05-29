#!/bin/bash

# Reusable virtual environment activation script
# This script should be sourced by other scripts: source ./bin/activate_venv.sh

# Get the project root directory (relative to where this script is located)
if [ -z "$PROJECT_ROOT" ]; then
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

# Function to activate virtual environment
activate_venv() {
    local venv_path="$PROJECT_ROOT/.venv/bin/activate"
    
    if [ -f "$venv_path" ]; then
        echo "🔧 Activating virtual environment..."
        source "$venv_path"
        return 0
    else
        echo "⚠️  Virtual environment not found at $venv_path"
        echo "Trying to use system Python..."
        return 1
    fi
}

# Function to setup project environment (activate venv + change to project root)
setup_project_env() {
    activate_venv
    
    # Change to project root
    cd "$PROJECT_ROOT" || {
        echo "❌ Failed to change to project root: $PROJECT_ROOT"
        exit 1
    }
    
    # Set Python path to include the src directory
    export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$PROJECT_ROOT"
}

# If this script is run directly (not sourced), just activate the environment
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    setup_project_env
    echo "✅ Project environment ready!"
    echo "Project root: $PROJECT_ROOT"
    echo "Python path: $PYTHONPATH"
    if command -v python >/dev/null 2>&1; then
        echo "Python executable: $(which python)"
    fi
fi 