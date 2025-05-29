"""Argentic - AI Agent Framework"""


def _get_version():
    """Get version from package metadata or pyproject.toml"""
    # Try to get version from installed package metadata
    try:
        from importlib.metadata import version

        return version("argentic")
    except ImportError:
        # Fallback for Python < 3.8
        try:
            from importlib_metadata import version

            return version("argentic")
        except ImportError:
            pass  # Fall through to file reading
        except Exception:
            pass  # Fall through to file reading
    except Exception:
        pass  # Fall through to file reading

    # Package not installed or metadata not available, read from pyproject.toml
    try:
        from pathlib import Path

        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        if pyproject_path.exists():
            # Try using tomllib (Python 3.11+)
            try:
                import tomllib

                with open(pyproject_path, "rb") as f:
                    pyproject = tomllib.load(f)
                return pyproject["project"]["version"]
            except ImportError:
                # Fallback: simple regex parsing for version
                import re

                with open(pyproject_path, "r") as f:
                    content = f.read()
                match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
                return match.group(1) if match else "unknown"
    except Exception:
        pass

    return "unknown"


__version__ = _get_version()

# This package serves as the entry point for the argentic module
# The actual implementation is in the src directory
