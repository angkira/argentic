import asyncio
import signal
from typing import Optional
import yaml
from dotenv import load_dotenv

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
logger = get_logger("main")
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


async def main():
    global messager, agent, llm_provider  # Allow modification

    # --- Load Config ---
    config = yaml.safe_load(open("config.yaml"))
    mqtt_cfg = config["mqtt"]
    # llm_cfg is now part of the main config, accessed by LLMFactory
    topic_cfg = config["topics"]
    log_cfg = config.get("logging", {})
    log_level = parse_log_level(log_cfg.get("level", "debug"))
    logger.setLevel(log_level.value)

    # --- Initialize Components ---
    logger.info("Initializing components...")

    # Initialize Messager
    messager = Messager(
        broker_address=mqtt_cfg["broker_address"],
        port=mqtt_cfg["port"],
        client_id=mqtt_cfg.get("client_id", "ai_agent_client"),  # Use client_id from config
        username=mqtt_cfg.get("username"),
        password=mqtt_cfg.get("password"),
        keepalive=mqtt_cfg.get("keepalive", 60),
        pub_log_topic=topic_cfg.get("log", "agent/log"),  # Get log topic from config
        log_level=log_level,
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

    agent = Agent(
        llm=llm_provider,  # Pass the provider instance
        messager=messager,
        log_level=log_level,
        register_topic=register_topic,
        answer_topic=answer_topic,  # Pass answer_topic directly
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
                logger.error(f"Failed to log critical error via MQTT: {log_e}")
    finally:
        logger.info("Main function finished or errored. Cleaning up...")
        if llm_provider:  # Ensure LLM provider is stopped
            logger.info("Ensuring LLM provider is stopped in finally block...")
            try:
                await llm_provider.stop()
                logger.info("LLM provider stopped in finally block.")
            except Exception as e:
                logger.error(f"Error stopping LLM provider in finally block: {e}")

        if messager and messager.is_connected():
            logger.info("Ensuring messager is stopped in finally block...")
            try:
                await messager.stop()
                logger.info("Messager stopped in finally block.")
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
