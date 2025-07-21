#!/usr/bin/env python3

import asyncio
import os
import signal
import yaml
from dotenv import load_dotenv

from argentic.core.messager.messager import Messager
from argentic.core.logger import get_logger, LogLevel

from .email_tool import EmailTool
from .note_creator_tool import NoteCreatorTool


async def main():
    """Run the secretary tools service."""
    load_dotenv()

    logger = get_logger("secretary_tools", LogLevel.INFO)

    # Load messaging configuration
    messaging_config_path = os.path.join(os.path.dirname(__file__), "config_messaging.yaml")
    with open(messaging_config_path, "r") as f:
        messaging_full_config = yaml.safe_load(f)
    if messaging_full_config is None:
        messaging_full_config = {}
    messaging_config_data = messaging_full_config.get("messaging", {})
    if not isinstance(messaging_config_data, dict):
        logger.warning("Warning: 'messaging' configuration not found. Using defaults.")
        messaging_config_data = {}

    # Initialize Messager
    broker_address = messaging_config_data.get("broker_address", "localhost")
    port = messaging_config_data.get("port", 1883)
    client_id = messaging_config_data.get("client_id", "secretary_tools_service")
    username = messaging_config_data.get("username")
    password = messaging_config_data.get("password")
    keepalive = messaging_config_data.get("keepalive", 60)

    messager = Messager(
        broker_address=broker_address,
        port=port,
        client_id=client_id,
        username=username,
        password=password,
        keepalive=keepalive,
    )

    stop_event = asyncio.Event()

    def signal_handler(sig):
        logger.info(f"Received signal {sig}. Shutting down...")
        stop_event.set()

    # Set up signal handlers
    for sig in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(sig, lambda s, f: signal_handler(s))

    try:
        await messager.connect()
        logger.info("Connected to messaging system")

        # Initialize tools
        email_tool = EmailTool(messager=messager)
        note_tool = NoteCreatorTool(messager=messager)

        logger.info("Secretary tools initialized")

        # Register tools with the agent
        # Using default topic configuration
        register_topic = "agent/tools/register"
        call_topic_base = "agent/tools/call"
        response_topic_base = "agent/tools/response"
        status_topic = "agent/status/info"

        await email_tool.register(
            register_topic, status_topic, call_topic_base, response_topic_base
        )
        await note_tool.register(register_topic, status_topic, call_topic_base, response_topic_base)

        logger.info("Secretary Tools Service running... Press Ctrl+C to exit.")
        await stop_event.wait()

    except asyncio.CancelledError:
        logger.info("Service cancelled")
    except Exception as e:
        logger.critical(f"Unhandled error: {e}", exc_info=True)
    finally:
        if messager and messager.is_connected():
            await messager.disconnect()
        logger.info("Secretary Tools Service stopped")


if __name__ == "__main__":
    asyncio.run(main())
