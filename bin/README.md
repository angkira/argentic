# Scripts Directory

This directory contains various utility scripts for the project.

## Core Scripts

### `activate_venv.sh`

**Shared virtual environment activation utility**

- Reusable script that handles virtual environment activation
- Can be sourced by other scripts or run directly
- Provides functions: `activate_venv()` and `setup_project_env()`
- Sets up project root, PYTHONPATH, and activates virtual environment

### `run_unit_tests.sh`

**Unit test runner**

- Runs pytest with virtual environment activated
- Excludes e2e tests by default
- Uses shared `activate_venv.sh` for environment setup

### `auto_bump_version.sh`

**Automatic version bumping with Commitizen**

- Analyzes conventional commits and bumps version accordingly
- Creates changelog and git tags
- Handles different exit codes appropriately
- Uses shared `activate_venv.sh` for environment setup

### `safe_push.sh`

**Safe git push with pre-push checks**

- Alternative to using Cursor's git GUI
- Runs unit tests and version bumping before pushing
- Provides clear feedback for each step
- Uses shared `activate_venv.sh` for environment setup

## GitHub Release Scripts

### `create_github_release.sh`

**GitHub release creation**

- Creates GitHub releases from the latest git tag
- Extracts release notes from CHANGELOG.md
- Handles existing releases (with confirmation)
- Requires GitHub CLI authentication

### `release_workflow.sh`

**Complete release workflow**

- End-to-end release automation
- Performs version bump, commit, push, and GitHub release creation
- Checks for uncommitted changes
- Provides comprehensive feedback

## Usage

### Direct execution:

```bash
# Activate virtual environment
./bin/activate_venv.sh

# Run unit tests
./bin/run_unit_tests.sh

# Safe push (runs all pre-push checks)
./bin/safe_push.sh

# Create GitHub release for latest tag
./bin/create_github_release.sh

# Complete release workflow
./bin/release_workflow.sh
```

### Release Workflow:

For a complete release (recommended):

```bash
# Make your commits with conventional commit messages
git commit -m "feat: add new feature"
git commit -m "fix: resolve bug"

# Run complete release workflow
./bin/release_workflow.sh
```

This will:

1. Analyze commits and bump version
2. Update CHANGELOG.md
3. Commit changes
4. Push to GitHub
5. Create GitHub release

### As part of git hooks:

These scripts are automatically called by git pre-push hooks configured in `.pre-commit-config.yaml`.

## Dependencies

- Virtual environment at `.venv/`
- Python packages: pytest, commitizen
- Git repository with conventional commits
- GitHub CLI (gh) for release creation
- Authenticated GitHub account

# Testing Scripts for Argentic Messaging System

This directory contains scripts to help with testing the Argentic messaging system.

## Running Tests

### E2E Tests with Docker Containers

The `run_e2e_tests.sh` script helps manage Docker containers for end-to-end testing.

```bash
# Basic usage - use existing containers if running, or start new ones
./bin/run_e2e_tests.sh --start-docker

# Force restart of containers even if they're running
./bin/run_e2e_tests.sh --force-restart-docker

# Only manage containers, don't run tests
./bin/run_e2e_tests.sh --docker-only --start-docker

# Start containers, run tests, then stop containers
./bin/run_e2e_tests.sh --start-docker --stop-docker

# Pass arguments to pytest
./bin/run_e2e_tests.sh --start-docker -- -v
```

#### Container Reuse

The script now robustly detects if the required containers (mosquitto, redis, rabbitmq, zookeeper, kafka) are already running and reuses them by default. The detection works with:

- Containers started by our docker compose file
- Containers started by other means but with matching names
- Custom-built images like our mosquitto container

To force a fresh start of containers:

```bash
./bin/run_e2e_tests.sh --force-restart-docker
```

## Serialization Helper

The `fix_serialization.py` script provides a universal JSON encoder that properly handles:

- UUID objects
- Datetime objects
- Custom BaseMessage objects
- TestMessage objects

This is automatically applied by the test scripts.

## Environment Setup

All test scripts ensure:

1. The necessary Python dependencies are installed
2. The proper PYTHONPATH is set
3. Docker containers are properly configured

## RabbitMQ Configuration

When RabbitMQ containers are running, the scripts automatically:

1. Create a test vhost if it doesn't exist
2. Set proper permissions for the guest user
3. Verify the RabbitMQ management interface is accessible

## Port Configuration

To avoid conflicts with locally running services:

- MQTT (Mosquitto): Port 1884 (instead of default 1883)
- Redis: Port 6380 (instead of default 6379)
- RabbitMQ: Standard ports (5672, 15672)
- Kafka: Standard port (9092)
- Zookeeper: Standard port (2181)
