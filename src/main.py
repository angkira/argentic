import asyncio
import signal
from typing import Optional
import yaml
from dotenv import load_dotenv
import argparse
import os

# Core components
from core.messager.messager import Messager
from core.agent.agent import Agent
from core.logger import get_logger, parse_log_level
from core.protocol.message import AskQuestionMessage
from core.llm.llm_factory import LLMFactory  # Import LLMFactory
from core.llm.providers.base import ModelProvider  # Import ModelProvider for type hinting

# --- Global Variables ---
stop_event = asyncio.Event()
messager: Optional[Messager] = None
agent: Optional[Agent] = None
llm_provider: Optional[ModelProvider] = None  # Add global for llm_provider
cleanup_started = False  # Add cleanup flag
logger = get_logger("main")
load_dotenv()


async def shutdown_handler(sig):
    """Graceful shutdown handler."""
    global cleanup_started
    logger.info(f"Received exit signal {sig.name}...")
    stop_event.set()

    if cleanup_started:
        logger.info("Cleanup already in progress, skipping signal handler cleanup")
        return

    cleanup_started = True

    if llm_provider:  # Stop LLM provider first
        logger.info("Stopping LLM provider...")
        try:
            await asyncio.wait_for(llm_provider.stop(), timeout=5.0)
            logger.info("LLM provider stopped.")
        except asyncio.TimeoutError:
            logger.warning("LLM provider stop timed out")
        except Exception as e:
            logger.error(f"Error stopping LLM provider: {e}")

    if messager and messager.is_connected():
        logger.info("Stopping messager...")
        try:
            await asyncio.wait_for(messager.stop(), timeout=10.0)
            logger.info("Messager stopped.")
        except asyncio.TimeoutError:
            logger.warning("Messager stop timed out")
        except Exception as e:
            logger.error(f"Error stopping messager: {e}")

    # Cancel remaining tasks
    remaining = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if remaining:
        logger.info(f"Cancelling {len(remaining)} remaining tasks...")
        for task in remaining:
            task.cancel()
        try:
            await asyncio.wait_for(asyncio.gather(*remaining, return_exceptions=True), timeout=3.0)
        except asyncio.TimeoutError:
            logger.warning("Some tasks did not cancel within timeout")

    logger.info("Shutdown handler complete.")


async def main():
    global messager, agent, llm_provider  # Allow modification

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="AI Agent Main Application")
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
    args = parser.parse_args()

    # --- Setup Logging Early ---
    # Use parsed log level directly
    parsed_log_level_str = args.log_level
    log_level_enum = parse_log_level(parsed_log_level_str)
    logger.setLevel(log_level_enum.value)
    logger.info(f"Log level set to: {parsed_log_level_str.upper()} (from CLI/ENV/Default)")

    # --- Load Config ---
    logger.info(f"Loading configuration from: {args.config_path}")
    try:
        with open(args.config_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.critical(f"Configuration file not found: {args.config_path}")
        return
    except Exception as e:
        logger.critical(f"Error loading configuration file {args.config_path}: {e}")
        return

    messaging_cfg = config["messaging"]
    # llm_cfg is now part of the main config, accessed by LLMFactory
    topic_cfg = config["topics"]
    # Log level from config is now superseded by args.log_level
    # log_cfg = config.get("logging", {}) # No longer needed for level
    # log_level = parse_log_level(log_cfg.get("level", "debug")) # No longer needed
    # logger.setLevel(log_level.value) # Moved up and uses args.log_level

    # --- Initialize Components ---
    logger.info("Initializing components...")

    # Initialize Messager
    messager = Messager(
        protocol=messaging_cfg["protocol"],
        broker_address=messaging_cfg["broker_address"],
        port=messaging_cfg["port"],
        client_id=messaging_cfg.get("client_id", "ai_agent_client"),  # Use client_id from config
        username=messaging_cfg.get("username"),
        password=messaging_cfg.get("password"),
        keepalive=messaging_cfg.get("keepalive", 60),
        pub_log_topic=topic_cfg.get("log", "agent/log"),  # Get log topic from config
        log_level=log_level_enum,  # Use the enum parsed from args
    )

    # Initialize LLM Provider using the factory
    try:
        llm_provider = LLMFactory.create(config, messager)  # Pass full config and messager
        logger.info(f"LLM Provider initialized: {type(llm_provider).__name__}")
    except Exception as e:
        logger.critical(f"Failed to initialize LLM Provider: {e}", exc_info=True)
        return  # Cannot continue without LLM

    # Initialize Agent with topics from config
    tools_cfg = topic_cfg.get("tools", {})
    register_topic = tools_cfg.get("register", "agent/tools/register")
    answer_topic = topic_cfg.get("responses", {}).get("answer", "agent/response/answer")

    # Extract system prompt from config (optional)
    agent_cfg = config.get("agent", {})
    system_prompt = agent_cfg.get("system_prompt")

    if system_prompt:
        logger.info("Using custom system prompt from config")
    else:
        logger.info("Using default system prompt")

    agent = Agent(
        llm=llm_provider,  # Pass the provider instance
        messager=messager,
        log_level=log_level_enum,  # Use the enum parsed from args
        register_topic=register_topic,
        answer_topic=answer_topic,  # Pass answer_topic directly
        system_prompt=system_prompt,  # Pass system prompt from config
    )
    logger.info("Agent initialized.")

    # --- Setup Signal Handling ---
    loop = asyncio.get_running_loop()
    for sig_name in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(
            sig_name, lambda s=sig_name: asyncio.create_task(shutdown_handler(s))
        )

    # --- Connect and Subscribe ---
    try:
        logger.info("Connecting Messager...")
        if not await messager.connect():
            logger.critical("Messager connection failed. Exiting.")
            if llm_provider:  # Attempt to stop provider even if messager fails
                await llm_provider.stop()
            return

        logger.info("Messager connected.")
        await messager.log("AI Agent: Messager connected.")

        # Start LLM Provider (if it has a start method, e.g., for auto-starting servers)
        if hasattr(llm_provider, "start"):
            logger.info(f"Starting LLM Provider ({type(llm_provider).__name__})...")
            await llm_provider.start()
            logger.info("LLM Provider started.")

        # Now that Messager is connected and LLM provider started, initialize Agent's async parts
        await agent.async_init()
        logger.info("Agent async_init complete after connection and LLM start.")

        # --- Subscribe to ask_question topic from config ---
        ask_topic = topic_cfg.get("commands", {}).get("ask_question", "agent/command/ask_question")
        await messager.subscribe(
            ask_topic,
            agent.handle_ask_question,
            message_cls=AskQuestionMessage,
        )
        logger.info(f"Subscribed to ask topic: {ask_topic}")

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
                logger.error(f"Failed to log critical error via Messager: {log_e}")
    finally:
        global cleanup_started
        logger.info("Main function finished or errored. Cleaning up...")

        if cleanup_started:
            logger.info("Cleanup already completed by signal handler")
            return

        cleanup_started = True

        if llm_provider:  # Ensure LLM provider is stopped
            logger.info("Ensuring LLM provider is stopped in finally block...")
            try:
                await asyncio.wait_for(llm_provider.stop(), timeout=5.0)
                logger.info("LLM provider stopped in finally block.")
            except asyncio.TimeoutError:
                logger.warning("LLM provider stop timed out in finally block")
            except Exception as e:
                logger.error(f"Error stopping LLM provider in finally block: {e}")

        if messager and messager.is_connected():
            logger.info("Ensuring messager is stopped in finally block...")
            try:
                await asyncio.wait_for(messager.stop(), timeout=10.0)
                logger.info("Messager stopped in finally block.")
            except asyncio.TimeoutError:
                logger.warning("Messager stop timed out in finally block")
            except Exception as e:
                logger.error(f"Error stopping messager in finally block: {e}")
        logger.info("AI Agent cleanup complete.")


if __name__ == "__main__":
    exit_code = 0
    try:
        asyncio.run(main())
        logger.info("Application finished normally.")
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt caught in __main__. Service shutting down.")
    except Exception as e:
        logger.critical(f"Unhandled exception in top-level execution: {e}", exc_info=True)
        exit_code = 1
    finally:
        logger.info(f"Application exiting with code {exit_code}.")
        exit(exit_code)
