#!/usr/bin/env python3
"""
Test runner script for Argentic multi-agent system.

This script provides convenient commands to run different types of tests
and generate coverage reports.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and handle errors."""
    if description:
        print(f"\n{'='*60}")
        print(f"Running: {description}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        return False
    except FileNotFoundError as e:
        print(f"Command not found: {e}")
        print("Make sure pytest is installed: pip install pytest pytest-asyncio")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run tests for Argentic multi-agent system")
    parser.add_argument(
        "test_type",
        choices=["unit", "integration", "e2e", "all", "fast", "slow", "coverage"],
        help="Type of tests to run",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--parallel",
        "-p",
        action="store_true",
        help="Run tests in parallel (requires pytest-xdist)",
    )
    parser.add_argument(
        "--coverage",
        "-c",
        action="store_true",
        help="Generate coverage report (requires pytest-cov)",
    )
    parser.add_argument("--html-report", action="store_true", help="Generate HTML coverage report")
    parser.add_argument(
        "--markers", "-m", help="Run tests with specific markers (e.g., 'not slow')"
    )
    parser.add_argument("--file", "-f", help="Run specific test file")
    parser.add_argument("--function", "-k", help="Run tests matching pattern")
    parser.add_argument(
        "--lf",
        "--last-failed",
        action="store_true",
        help="Run only tests that failed in the last run",
    )
    parser.add_argument(
        "--tb",
        choices=["short", "long", "line", "native", "no"],
        default="short",
        help="Traceback format",
    )

    args = parser.parse_args()

    # Base command
    cmd = ["python", "-m", "pytest"]

    # Add verbosity
    if args.verbose:
        cmd.append("-vv")
    else:
        cmd.append("-v")

    # Add traceback format
    cmd.extend(["--tb", args.tb])

    # Add parallel execution
    if args.parallel:
        cmd.extend(["-n", "auto"])

    # Add coverage
    if args.coverage or args.test_type == "coverage":
        cmd.extend(["--cov=src/argentic", "--cov-report=term-missing"])
        if args.html_report:
            cmd.extend(["--cov-report=html"])

    # Add last failed
    if args.lf:
        cmd.append("--lf")

    # Add specific markers
    if args.markers:
        cmd.extend(["-m", args.markers])

    # Add pattern matching
    if args.function:
        cmd.extend(["-k", args.function])

    # Add specific file
    if args.file:
        cmd.append(args.file)
    else:
        # Add test selection based on type
        if args.test_type == "unit":
            cmd.extend(["-m", "unit", "tests/unit/"])
        elif args.test_type == "integration":
            cmd.extend(["-m", "integration", "tests/integration/"])
        elif args.test_type == "e2e":
            cmd.extend(["-m", "e2e", "tests/integration/"])
        elif args.test_type == "fast":
            cmd.extend(["-m", "not slow"])
        elif args.test_type == "slow":
            cmd.extend(["-m", "slow"])
        elif args.test_type == "coverage":
            cmd.append("tests/")
        elif args.test_type == "all":
            cmd.append("tests/")

    # Ensure we're in the right directory
    project_root = Path(__file__).parent
    os.chdir(project_root)

    # Run the tests
    description = f"Running {args.test_type} tests"
    success = run_command(cmd, description)

    if success:
        print(f"\n✅ {args.test_type.capitalize()} tests completed successfully!")

        if args.coverage or args.test_type == "coverage":
            print("\n📊 Coverage report generated.")
            if args.html_report:
                html_path = project_root / "htmlcov" / "index.html"
                print(f"📄 HTML report available at: {html_path}")
    else:
        print(f"\n❌ {args.test_type.capitalize()} tests failed!")
        sys.exit(1)


def run_quick_tests():
    """Run quick unit tests for development."""
    cmd = ["python", "-m", "pytest", "-v", "--tb=short", "-m", "not slow", "tests/unit/"]
    return run_command(cmd, "Quick unit tests")


def run_full_suite():
    """Run the full test suite with coverage."""
    cmd = [
        "python",
        "-m",
        "pytest",
        "-v",
        "--tb=short",
        "--cov=src/argentic",
        "--cov-report=term-missing",
        "--cov-report=html",
        "tests/",
    ]
    return run_command(cmd, "Full test suite with coverage")


def check_dependencies():
    """Check if required testing dependencies are installed."""
    required_packages = [
        "pytest",
        "pytest-asyncio",
    ]

    optional_packages = [
        "pytest-cov",
        "pytest-xdist",
        "pytest-html",
    ]

    print("Checking test dependencies...")

    missing_required = []
    missing_optional = []

    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            missing_required.append(package)
            print(f"❌ {package} (required)")

    for package in optional_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            missing_optional.append(package)
            print(f"⚠️  {package} (optional)")

    if missing_required:
        print(f"\n❌ Missing required packages: {', '.join(missing_required)}")
        print("Install with: pip install " + " ".join(missing_required))
        return False

    if missing_optional:
        print(f"\n⚠️  Missing optional packages: {', '.join(missing_optional)}")
        print("Install with: pip install " + " ".join(missing_optional))

    print("\n✅ All required dependencies are available!")
    return True


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Argentic Test Runner")
        print("===================")
        print()
        print("Usage examples:")
        print("  python run_tests.py unit              # Run unit tests")
        print("  python run_tests.py integration       # Run integration tests")
        print("  python run_tests.py all --coverage    # Run all tests with coverage")
        print("  python run_tests.py fast              # Run fast tests only")
        print("  python run_tests.py unit -k test_agent # Run specific test pattern")
        print()
        print("For more options, run: python run_tests.py --help")
        print()

        # Check dependencies
        if not check_dependencies():
            sys.exit(1)

        # Run quick tests by default
        print("Running quick unit tests...")
        if not run_quick_tests():
            sys.exit(1)
    else:
        main()
