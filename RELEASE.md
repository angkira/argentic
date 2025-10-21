# Release Process

This document describes the release process for Argentic.

## Overview

The project uses **automated version bumping** via git hooks:
- **post-commit hook**: Automatically bumps version after each commit based on conventional commit message
- **pre-push hook**: Runs unit tests before push

Version is automatically synchronized between `pyproject.toml`, GitHub tags, and PyPI.

## Prerequisites

1. **Environment Setup**:
   ```bash
   # Install dev dependencies
   uv pip install -e ".[dev]"
   
   # Install pre-commit hooks
   pre-commit install --hook-type post-commit --hook-type pre-push
   ```

2. **PyPI Token**:
   Create a `.env` file in the project root with:
   ```bash
   PY_PI_TOKEN=your_pypi_token_here
   ```

3. **GitHub CLI**:
   Make sure you have `gh` CLI installed and authenticated:
   ```bash
   gh auth login
   ```

## Automatic Version Bumping (Already Configured)

Version is automatically bumped after each commit based on conventional commit message:
- `feat:` → Minor version bump (0.x.0)
- `fix:` → Patch version bump (0.0.x)
- `BREAKING CHANGE:` or `!:` → Major version bump (x.0.0)
- Other types → Patch version bump (0.0.x)

The version bump happens via **post-commit hook** and amends the commit automatically.

## Release to PyPI and GitHub

### Quick Release (Recommended)

Run the automated release workflow:

```bash
./bin/release_workflow.sh
```

This script will:
1. ✅ Create git tag for current version
2. ✅ Build Python package
3. ✅ Publish to PyPI
4. ✅ Push tag to GitHub
5. ✅ Create GitHub release with changelog notes

### Manual Steps

If you need more control, you can run individual steps:

#### 1. Publish to PyPI Only

```bash
./bin/publish_to_pypi.sh
```

#### 2. Create GitHub Release Only

```bash
./bin/create_github_release.sh
```

#### 3. Push with Tests

Use the safe push script that runs tests before pushing:

```bash
./bin/safe_push.sh
```

Or use the shipit script that also handles version bumping:

```bash
./bin/shipit.sh
```

## Automated Publishing via GitHub Actions

The project includes GitHub workflows:

### 1. Publish to PyPI on Release

**File**: `.github/workflows/publish-pypi.yml`

Automatically publishes to PyPI when a GitHub release is created.

**Setup**:
1. Go to your GitHub repository settings
2. Navigate to Secrets and Variables → Actions
3. Add a new secret: `PYPI_TOKEN` with your PyPI token value

### 2. Manual Release

**File**: `.github/workflows/manual-release.yml`

Allows manual release creation from GitHub Actions UI:
1. Go to Actions → Manual Release
2. Select bump type (patch/minor/major)
3. Run workflow

## Workflow Examples

### Standard Development Flow

```bash
# 1. Make changes
git add .
git commit -m "feat: add new feature"
# → Version automatically bumped (e.g., 0.11.2 → 0.12.0)

# 2. Push changes (runs tests automatically)
git push
# → Pre-push hook runs unit tests

# 3. Create release
./bin/release_workflow.sh
# → Publishes to PyPI and creates GitHub release
```

### Quick Ship (Tests + Push)

```bash
git add .
git commit -m "fix: resolve bug"
./bin/shipit.sh
# → Bumps version, pushes with tags
```

## Conventional Commits

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
feat: add new feature          → Minor bump (0.x.0)
fix: fix a bug                 → Patch bump (0.0.x)
docs: documentation changes    → Patch bump (0.0.x)
style: code style changes      → Patch bump (0.0.x)
refactor: code refactoring     → Patch bump (0.0.x)
test: add tests                → Patch bump (0.0.x)
chore: maintenance tasks       → Patch bump (0.0.x)

feat!: breaking change         → Major bump (x.0.0)
BREAKING CHANGE: in body       → Major bump (x.0.0)
```

## Troubleshooting

### Version Already Exists on PyPI

The script will ask if you want to continue. If you need a new version, make a new commit first to trigger version bump.

### GitHub Release Already Exists

The script will ask if you want to delete and recreate it.

### .env File Missing

Create `.env` file with:
```bash
PY_PI_TOKEN=your_token_here
```

### Pre-commit Hooks Not Working

Reinstall hooks:
```bash
pre-commit install --hook-type post-commit --hook-type pre-push
```

### Version Not Bumping

Check that:
1. Pre-commit hooks are installed
2. Commit message follows conventional commits format
3. `pyproject.toml` is not already in the commit (to avoid infinite loop)

## Current Version

Check current version:
```bash
grep 'version = ' pyproject.toml | cut -d'"' -f2
```

## File Structure

- `bin/post_commit_version_bump.sh` - Automatic version bump hook
- `bin/release_workflow.sh` - Full release to PyPI + GitHub
- `bin/publish_to_pypi.sh` - Publish to PyPI only
- `bin/create_github_release.sh` - Create GitHub release only
- `bin/shipit.sh` - Bump + push with tags
- `bin/safe_push.sh` - Run tests + push
- `.pre-commit-config.yaml` - Pre-commit hooks configuration
- `pyproject.toml` - Version source of truth

## Links

- **PyPI**: https://pypi.org/project/argentic/
- **GitHub Releases**: https://github.com/angkira/argentic/releases
- **Commitizen Docs**: https://commitizen-tools.github.io/commitizen/
