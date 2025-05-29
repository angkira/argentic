#!/usr/bin/env python3
"""
Entry point for running argentic as a Python module.
Usage: python -m argentic [subcommand] [options]
"""

import sys
from pathlib import Path

# Add src to path so we can import the actual implementation
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


def main():
    """Main entry point for the argentic console script."""
    # Import the main function from the src/__main__.py module
    import importlib.util

    spec = importlib.util.spec_from_file_location("src_main", src_path / "__main__.py")
    if spec is None or spec.loader is None:
        raise ImportError("Could not load src/__main__.py")
    src_main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(src_main)
    src_main.main()


# Import and run the main function from the src implementation
if __name__ == "__main__":
    main()
