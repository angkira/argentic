import asyncio
import argparse # For command-line arguments
import os # For environment variables
import signal
import sys
from typing import Optional
import yaml
from dotenv import load_dotenv
from pathlib import Path

# Core components
from core.messager.messager import Messager
from core.agent.agent import Agent
from core.logger import get_logger, parse_log_level, LogLevel
from core.protocol.message import AskQuestionMessage
from core.llm.llm_factory import LLMFactory  # Import LLMFactory
from core.llm.providers.base import ModelProvider  # Import ModelProvider for type hinting
from core.messager.protocols import MessagerProtocol # Ensure this is imported


logger = get_logger(__name__, level=LogLevel.DEBUG)

# --- Configuration Loading ---
def load_config():
    parser = argparse.ArgumentParser(description="Argentic AI Agent Service")
    parser.add_argument(
        "--config",
        type=str,
        default=None, # Make default None to allow fallback
        help="Path to the configuration YAML file."
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=os.getenv("LOG_LEVEL", "INFO"), # Default from ENV or INFO
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Overrides config file setting."
    )
    # Use parse_known_args to ignore extra arguments from scripts like start.sh
    args, unknown = parser.parse_known_args()
    if unknown:
        logger.warning(f"Ignoring unrecognized arguments: {unknown}")

    # Config path resolution remains the same
    config_path_str = args.config or os.getenv("CONFIG_PATH") or "./config.yaml"
    config_path = Path(config_path_str).resolve()

    config_data = {}
    if not config_path.exists():
        logger.warning(f"Configuration file not found at {config_path}. Proceeding with defaults/CLI args.")
    else:
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            logger.info(f"Successfully loaded configuration from: {config_path}")
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file {config_path}: {e}", exc_info=True)
            # Decide if we should exit or proceed with defaults
            logger.warning("Proceeding with defaults due to config parsing error.")
        except Exception as e:
            logger.error(f"Error loading configuration file {config_path}: {e}", exc_info=True)
            logger.warning("Proceeding with defaults due to config loading error.")
    
    # Return loaded config data AND the parsed log_level argument
    return config_data, args.log_level 

# Load config and get log level from args
config, cli_log_level = load_config()

# --- Logger Setup ---
# Use log level from CLI arg if provided, otherwise from config, else default INFO
log_level_str = cli_log_level or config.get("agent", {}).get("log_level", "INFO")
log_level = parse_log_level(log_level_str)
# Set up the logger *after* determining the level
logger = get_logger(__name__, level=log_level)
logger.info(f"Setting log level to: {log_level.name}") # Log the determined level

# --- Global Variables ---
stop_event = asyncio.Event()
messager: Optional[Messager] = None
agent: Optional[Agent] = None
llm_provider: Optional[ModelProvider] = None  # Add global for llm_provider
load_dotenv()


async def shutdown_handler(sig):
    """Graceful shutdown handler."""
    global llm_provider  # Allow modification
    logger.info(f"Received exit signal {sig.name}...")
    stop_event.set()

    if llm_provider:  # Stop LLM provider first
        logger.info("Stopping LLM provider...")
        try:
            await llm_provider.stop()
            logger.info("LLM provider stopped.")
        except Exception as e:
            logger.error(f"Error stopping LLM provider: {e}")

    if messager and messager.is_connected():
        logger.info("Stopping messager...")
        try:
            await messager.stop()
            logger.info("Messager stopped.")
        except Exception as e:
            logger.error(f"Error stopping messager: {e}")

    remaining = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in remaining:
        task.cancel()
    await asyncio.gather(*remaining, return_exceptions=True)
    logger.info("Shutdown handler complete.")


async def initialize_services():
    global agent, messager
    logger.info("Initializing services...")

    # Messaging Configuration
    messaging_config = config.get("messaging", {})
    broker_address = messaging_config.get("broker_address", "localhost")
    port = messaging_config.get("port", 1883)
    client_id = messaging_config.get("client_id", "ai_agent_main_client")
    username = messaging_config.get("username")
    password = messaging_config.get("password")
    keepalive = messaging_config.get("keepalive", 60)
    tls_params = messaging_config.get("tls_params")
    
    # Determine protocol (default to MQTT if not specified)
    # This assumes your MessagerProtocol enum has a MQTT member
    protocol_str = messaging_config.get("driver", "mqtt").upper()
    try:
        protocol = MessagerProtocol[protocol_str]
    except KeyError:
        logger.error(f"Unsupported messaging driver/protocol: {protocol_str}. Defaulting to MQTT.")
        protocol = MessagerProtocol.MQTT

    # Get topics from the messaging config
    topics_config = messaging_config.get("topics", {})
    messager_log_topic = topics_config.get("log", "agent/log") # For Messager's own logging

    # Initialize Messager with individual parameters
    messager = Messager(
        broker_address=broker_address,
        port=port,
        protocol=protocol,
        client_id=client_id,
        username=username,
        password=password,
        keepalive=keepalive,
        pub_log_topic=messager_log_topic, # Pass the specific log topic for messager
        log_level=log_level,
        tls_params=tls_params
    )

    try:
        if await messager.connect():
            logger.info("Messager connected successfully.")
        else:
            logger.error("Failed to connect messager. Exiting.")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error connecting messager: {e}", exc_info=True)
        sys.exit(1)

    # LLM and Agent Initialization
    try:
        llm_provider = LLMFactory.create(config, messager) # Pass messager here if LLM needs it
        logger.info(f"LLM Provider created: {type(llm_provider).__name__}")

        agent_config = config.get("agent", {})
        agent_register_topic = topics_config.get("tools", {}).get("register", "agent/tools/register")
        agent_answer_topic = topics_config.get("responses", {}).get("answer", "agent/response/answer")

        agent = Agent(
            llm=llm_provider,
            messager=messager,
            log_level=log_level,
            register_topic=agent_register_topic,
            answer_topic=agent_answer_topic
        )
        await agent.async_init() # Initialize agent subscriptions via ToolManager
        logger.info("Agent initialized successfully.")

        # Subscribe agent handlers to topics defined in subscription_map
        # The subscription_map should now be read from the messaging_config
        subscription_map = topics_config.get("subscriptions", {})
        for topic_pattern, handler_name_str in subscription_map.items():
            handler_method = getattr(agent, handler_name_str, None)
            if handler_method and callable(handler_method):
                # Assuming Messager.subscribe can route to Agent methods
                # This part might need adjustment based on how Messager.subscribe works
                # with specific message types if your handlers expect them.
                # For now, assuming a generic BaseMessage or similar default.
                await messager.subscribe(topic_pattern, handler_method)
                logger.info(f"Agent subscribed to {topic_pattern} with handler {handler_name_str}")
            else:
                logger.error(f"Handler method '{handler_name_str}' not found or not callable in Agent for topic {topic_pattern}")

    except ValueError as e:
        logger.error(f"Error initializing LLM or Agent: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during service initialization: {e}", exc_info=True)
        sys.exit(1)

async def run():
    global stop_event
    logger.info("Starting main application loop...")
    await initialize_services()
    logger.info("Application initialized. Waiting for stop signal...")
    await stop_event.wait() # Keep running until stop_event is set
    logger.info("Stop signal received.")

async def shutdown_services(sig=None):
    global stop_event, agent, messager
    if stop_event.is_set():
        return
    logger.info(f"Shutdown initiated by signal {sig.name if sig else 'programmatic call'}...")
    stop_event.set() # Signal other loops to stop

    if agent:
        logger.info("Stopping Agent...")
        # await agent.stop() # Assuming agent might have a stop method
    if messager:
        logger.info("Disconnecting Messager...")
        await messager.disconnect()
        logger.info("Messager disconnected.")
    logger.info("Services shut down.")

async def main():
    loop = asyncio.get_event_loop()

    # Register signal handlers for graceful shutdown
    for sig_name in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(
            sig_name, lambda s=sig_name: asyncio.create_task(shutdown_services(s))
        )
    
    main_task = None
    try:
        main_task = asyncio.create_task(run())
        await main_task
    except asyncio.CancelledError:
        logger.info("Main task cancelled.")
    except Exception as e:
        logger.error(f"Critical error in main execution: {e}", exc_info=True)
    finally:
        if main_task and not main_task.done():
            main_task.cancel()
            await asyncio.wait([main_task], timeout=5) # Allow time for cancellation
        # Ensure shutdown runs even if run() fails partway through
        if not stop_event.is_set(): # If not already shut down by signal
            await shutdown_services()
        logger.info("Application terminated.")

if __name__ == "__main__":
    asyncio.run(main())
