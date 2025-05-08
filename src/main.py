import asyncio
import signal
from typing import Optional
import yaml
import os
from dotenv import load_dotenv

# Core components (assuming they are updated)
from core.messager.messager import Messager
from core.agent.agent import Agent
from core.tools.tool_manager import ToolManager
from core.logger import get_logger, LogLevel, parse_log_level
from core.protocol.message import (
    from_payload,
    AnyMessage,
    StatusMessage,
)  # Import necessary message types
from pydantic import ValidationError

# LLM Integration (Example using Ollama)
from langchain_community.chat_models import ChatOllama

# --- Global Variables ---
stop_event = asyncio.Event()
messager: Optional[Messager] = None
agent: Optional[Agent] = None
tool_manager: Optional[ToolManager] = None

logger = get_logger("main")  # Global logger for the main module
load_dotenv()  # Load environment variables from .env file


# --- Example Handlers (ensure signatures match aiomqtt.Message) ---
async def handle_agent_status(msg: aiomqtt.Message):
    """Example handler for agent status messages."""
    topic = msg.topic.value
    payload = msg.payload
    logger.info(f"Received status message on topic: {topic}")
    try:
        # Use from_payload or specific model validation
        status_msg: AnyMessage = from_payload(topic, payload)
        if isinstance(status_msg, StatusMessage):
            logger.info(f"Agent Status Update: {status_msg.status} - {status_msg.detail}")
            # Update internal state or UI based on status
        else:
            logger.warning(
                f"Received non-StatusMessage on status topic: {type(status_msg).__name__}"
            )

    except (ValidationError, ValueError) as e:
        payload_preview = payload[:100].decode("utf-8", errors="replace")
        logger.error(
            f"Failed to parse status message on {topic}: {e}. Payload: '{payload_preview}...'"
        )
    except Exception as e:
        logger.error(f"Error handling status message on {topic}: {e}", exc_info=True)


# Add other handlers like handle_command if needed, ensuring they take aiomqtt.Message


async def shutdown_handler(sig):
    """Graceful shutdown handler."""
    logger.info(f"Received exit signal {sig.name}...")
    stop_event.set()
    if messager and messager.is_connected():
        logger.info("Stopping messager...")
        try:
            await messager.stop()
            logger.info("Messager stopped.")
        except Exception as e:
            logger.error(f"Error stopping messager: {e}")
    # Cancel remaining tasks (excluding self) - This might be handled by asyncio.run cleanup
    # tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    # ... task cancellation logic ...


async def main():
    global messager, agent, tool_manager  # Allow modification

    # --- Load Config ---
    config = yaml.safe_load(open("config.yaml"))
    mqtt_cfg = config["mqtt"]
    llm_cfg = config["llm"]
    log_cfg = config.get("logging", {})
    log_level = parse_log_level(log_cfg.get("level", "info"))
    logger.setLevel(log_level.value)

    # --- Initialize Components ---
    logger.info("Initializing components...")

    # Initialize Messager
    messager = Messager(
        broker_address=mqtt_cfg["broker_address"],
        port=mqtt_cfg["port"],
        client_id=mqtt_cfg.get("agent_client_id", "ai_agent_client"),
        username=mqtt_cfg.get("username"),
        password=mqtt_cfg.get("password"),
        keepalive=mqtt_cfg.get("keepalive", 60),
        pub_log_topic=config["topics"]["log"],
        log_level=log_level,
    )

    # Initialize LLM (Example: Ollama)
    llm = ChatOllama(model=llm_cfg["model_name"])
    logger.info(f"LLM initialized: {llm_cfg['model_name']}")

    # Initialize Agent (passes messager)
    agent = Agent(llm=llm, messager=messager, log_level=log_level)
    agent.answer_topic = mqtt_cfg["topics"]["responses"]["answer"]

    # ToolManager is initialized inside Agent now, access it via agent.tool_manager
    tool_manager = agent.tool_manager
    logger.info("Agent and ToolManager initialized.")

    # --- Setup Signal Handling ---
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(shutdown_handler(s)))

    # --- Connect and Subscribe ---
    try:
        logger.info("Connecting Messager...")
        if not await messager.connect():
            logger.critical("Messager connection failed. Exiting.")
            return
        await messager.log("AI Agent: Messager connected.")
        logger.info("Messager connected.")

        # --- Subscribe to ask_question topic from config ---
        ask_topic = mqtt_cfg["topics"]["commands"]["ask_question"]
        await messager.subscribe(ask_topic, agent.handle_ask_question)
        logger.info(f"Subscribed to ask topic: {ask_topic}")

        # Define handlers and topics
        topic_handlers = {
            # mqtt_cfg["subscriptions"]["agent_status"]: handle_agent_status,
            tool_manager.register_topic: tool_manager._handle_tool_message,
            tool_manager.response_topic_base + "/#": tool_manager._handle_result_message,
        }

        # Subscribe to other topics individually
        logger.info("Subscribing to topics...")
        for topic, handler in topic_handlers.items():
            try:
                await messager.subscribe(topic, handler)
                logger.info(f"Subscribed to topic: {topic}")
            except Exception as e:
                logger.error(f"Failed to subscribe to topic {topic}: {e}")

        # --- Main Loop / Wait ---
        logger.info("AI Agent running... Press Ctrl+C to exit.")
        await stop_event.wait()

    except asyncio.CancelledError:
        logger.info("Main task cancelled.")
    except Exception as e:
        logger.critical(f"AI Agent encountered an unhandled error in main: {e}", exc_info=True)
        if messager:
            try:
                await messager.log(f"AI Agent: Critical error: {e}", level="critical")
            except Exception as log_e:
                logger.error(f"Failed to log critical error via MQTT: {log_e}")
    finally:
        logger.info("Main function finished or errored. Cleaning up...")
        if messager and messager.is_connected():
            logger.info("Ensuring messager is stopped in finally block...")
            try:
                await messager.stop()
            except Exception as e:
                logger.error(f"Error stopping messager in finally block: {e}")
        logger.info("AI Agent cleanup complete.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
        logger.info("Application finished normally.")
        exit_code = 0
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt caught in __main__. Service shutting down.")
        exit_code = 0
    except Exception as e:
        logger.critical(f"Unhandled exception in top-level execution: {e}", exc_info=True)
        exit_code = 1
    finally:
        logger.info(f"Application exiting with code {exit_code}.")
        exit(exit_code)
