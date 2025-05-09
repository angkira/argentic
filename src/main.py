import asyncio
import signal
from typing import Optional
import yaml
from dotenv import load_dotenv

# Core components (assuming they are updated)
from core.messager.messager import Messager
from core.agent.agent import Agent
from core.logger import get_logger, parse_log_level
from core.protocol.message import AskQuestionMessage  # For parsing incoming questions

# LLM Integration (Example using Ollama)
from langchain_community.chat_models import ChatOllama

# --- Global Variables ---
stop_event = asyncio.Event()
messager: Optional[Messager] = None
agent: Optional[Agent] = None
logger = get_logger("main")  # Global logger for the main module
load_dotenv()  # Load environment variables from .env file


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
    # Cancel all remaining tasks
    remaining = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in remaining:
        task.cancel()
    await asyncio.gather(*remaining, return_exceptions=True)


async def main():
    global messager, agent, tool_manager  # Allow modification

    # --- Load Config ---
    config = yaml.safe_load(open("config.yaml"))
    mqtt_cfg = config["mqtt"]
    llm_cfg = config["llm"]
    topic_cfg = config["topics"]
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

    # Initialize Agent with topics from config (passes messager)
    tools_cfg = topic_cfg.get("tools", {})
    register_topic = tools_cfg.get("register", "agent/tools/register")
    agent = Agent(
        llm=llm,
        messager=messager,
        log_level=log_level,
        register_topic=register_topic,
    )

    agent.answer_topic = topic_cfg["responses"]["answer"]
    logger.info("Agent initialized; will initialize ToolManager after connecting.")

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

        # Now that Messager is connected, initialize ToolManager subscriptions
        await agent.async_init()
        logger.info("Agent and ToolManager async_init complete after connection.")

        await messager.log("AI Agent: Messager connected.")
        logger.info("Messager connected.")

        # --- Subscribe to ask_question topic from config ---
        ask_topic = topic_cfg["commands"]["ask_question"]
        await messager.subscribe(
            ask_topic,
            agent.handle_ask_question,
            message_cls=AskQuestionMessage,
        )
        logger.info(f"Subscribed to ask topic: {ask_topic}")

        # Other tool registrations and result subscriptions are handled internally by ToolManager

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
