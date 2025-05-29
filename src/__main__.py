#!/usr/bin/env python3
"""
Entry point for running argentic as a Python module.
Usage: python -m argentic [subcommand] [options]
"""

import argparse
import asyncio
import sys
import os
from pathlib import Path

# Add src to path if not already there (for development mode)
src_path = Path(__file__).parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


def main():
    """Main entry point for the argentic module."""
    parser = argparse.ArgumentParser(
        prog="argentic",
        description="Argentic AI Agent Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
subcommands:
  agent       Start the AI agent service
  rag         Start the RAG tool service
  environment Start the environment tool service
  cli         Start the CLI client

examples:
  python -m argentic agent --config-path config.yaml --log-level INFO
  python -m argentic rag --config-path config.yaml
  python -m argentic cli
        """,
    )

    # Add global arguments
    parser.add_argument(
        "--config-path",
        type=str,
        default=os.getenv("CONFIG_PATH", "config.yaml"),
        help="Path to the configuration file. Defaults to 'config.yaml' or ENV VAR CONFIG_PATH.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=os.getenv("LOG_LEVEL", "INFO"),
        help="Logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL). Defaults to 'INFO' or ENV VAR LOG_LEVEL.",
    )

    # Add subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Create subcommand parsers (they inherit global arguments)
    subparsers.add_parser("agent", help="Start the AI agent service")
    subparsers.add_parser("rag", help="Start the RAG tool service")
    subparsers.add_parser("environment", help="Start the environment tool service")
    subparsers.add_parser("cli", help="Start the CLI client")

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Set environment variables for the modules to use
    os.environ["CONFIG_PATH"] = args.config_path
    os.environ["LOG_LEVEL"] = args.log_level

    # Route to appropriate module
    if args.command == "agent":
        from main import main as agent_main

        asyncio.run(agent_main())

    elif args.command == "rag":
        from services.rag_tool_service import main as rag_main

        asyncio.run(rag_main())

    elif args.command == "environment":
        from services.environment_tool_service import main as env_main

        asyncio.run(env_main())

    elif args.command == "cli":
        from cli_client import CliClient

        cli_client = CliClient()
        exit_code = 0 if cli_client.start() else 1
        sys.exit(exit_code)

    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
