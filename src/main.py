import os
import time
import argparse
import sys
import requests
import json
import threading
import signal
import traceback
from typing import Callable, Dict
import datetime

import yaml

from core.llm import LLMFactory
from core.messager import MQTTMessage, Messager
from core.agent import Agent
from core.decorators import mqtt_handler_decorator
from core.logger import get_logger, LogLevel, set_global_log_level, parse_log_level

from handlers.ask_question_handler import handle_ask_question as raw_handle_ask_question
from handlers.status_request_handler import (
    handle_status_request as raw_handle_status_request,
)

# Get script start time for uptime tracking
start_time = time.time()

# Initialize root logger
logger = get_logger("main", LogLevel.INFO)

CONFIG_PATH = "config.yaml"
try:
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Configuration loaded successfully from '{CONFIG_PATH}'")
except FileNotFoundError:
    logger.critical(f"Configuration file '{CONFIG_PATH}' not found")
    exit(1)
except yaml.YAMLError as e:
    logger.critical(f"Error parsing configuration file '{CONFIG_PATH}': {e}")
    logger.error(traceback.format_exc())
    exit(1)

# Get log level from config or default to INFO
log_level = parse_log_level(config.get("logging", {}).get("level", "debug"))
logger.info(f"Using log level: {log_level.name}")
set_global_log_level(log_level)

llm_config = config["llm"]
backend: str = llm_config.get("backend", "ollama")
model_name: str = llm_config.get("model_name")
use_chat: bool = llm_config.get("use_chat", False)

MQTT_BROKER: str = config["mqtt"]["broker_address"]
MQTT_PORT: int = config["mqtt"]["port"]
MQTT_CLIENT_ID: str = config["mqtt"]["client_id"]
MQTT_KEEPALIVE: int = config["mqtt"]["keepalive"]
MQTT_SUBSCRIPTIONS: Dict[str, str] = config["mqtt"]["subscriptions"]
MQTT_PUB_RESPONSE: str = config["mqtt"]["publish_topics"]["response"]
MQTT_PUB_STATUS: str = config["mqtt"]["publish_topics"]["status"]
MQTT_PUB_LOG: str = config["mqtt"]["publish_topics"]["log"]
MQTT_PUB_ERROR: str = config["mqtt"]["publish_topics"].get("error", "agent/status/error")

OLLAMA_BASE_URL = llm_config.get("base_url", "http://localhost:11434")

ollama_monitor_stop_event = threading.Event()


def monitor_ollama_status(base_url: str, stop_event: threading.Event, messager=None):
    """Monitor Ollama server status at regular intervals"""
    monitor_logger = get_logger("ollama_monitor", log_level)
    monitor_logger.info(f"Starting Ollama status checks at {base_url}")

    if messager:
        messager.mqtt_log(f"Ollama status monitoring started, checking {base_url}")

    check_interval = 30

    while not stop_event.is_set():
        status_message = "Unknown"
        status_level = "info"
        try:
            response = requests.get(base_url, timeout=5)
            response.raise_for_status()
            if "Ollama is running" in response.text:
                status_message = "Running"
            else:
                try:
                    ps_response = requests.get(f"{base_url}/api/ps", timeout=5)
                    ps_response.raise_for_status()
                    running_models = ps_response.json().get("models", [])
                    if running_models:
                        model_names = [m.get("name") for m in running_models]
                        status_message = f"Running (Models: {', '.join(model_names)})"
                    else:
                        status_message = "Running (No models loaded)"
                except requests.exceptions.RequestException as api_err:
                    status_message = f"API endpoint error ({api_err.__class__.__name__})"
                    status_level = "warning"
                except json.JSONDecodeError:
                    status_message = "Running (API response not valid JSON)"
                    status_level = "warning"
        except requests.exceptions.ConnectionError:
            status_message = "Connection Error (Server down?)"
            status_level = "error"
            monitor_logger.warning("Ollama connection error - server might be down")
        except requests.exceptions.Timeout:
            status_message = "Timeout (Server unresponsive)"
            status_level = "error"
            monitor_logger.warning("Ollama request timeout - server unresponsive")
        except requests.exceptions.RequestException as e:
            status_message = f"Error ({e.__class__.__name__})"
            status_level = "error"
            monitor_logger.warning(f"Ollama request error: {e.__class__.__name__}")
        except Exception as e:
            status_message = f"Unexpected error: {e}"
            status_level = "critical"
            monitor_logger.error(f"Unexpected error checking Ollama status: {e}")
            monitor_logger.error(traceback.format_exc())

        monitor_logger.info(f"Ollama status: {status_message}")
        if messager:
            messager.mqtt_log(f"Ollama status: {status_message}", level=status_level)

            # If we're in a critical state, publish to the error topic
            if status_level == "critical" or status_level == "error":
                try:
                    error_data = {
                        "source": messager.client_id,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "component": "ollama_monitor",
                        "status": status_message,
                        "level": status_level,
                    }
                    messager.publish(MQTT_PUB_ERROR, json.dumps(error_data))
                except:
                    # Don't let error publishing cause issues
                    pass

        stop_event.wait(check_interval)

    monitor_logger.info("Ollama status monitoring stopped")
    if messager:
        messager.mqtt_log("Ollama status monitoring stopped")


def run_rag_agent():
    """Run the main RAG agent service"""
    logger.info("AI Agent Starting")

    monitor_thread = None

    try:
        logger.info(f"Connecting to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}")
        messager: Messager = Messager(
            broker_address=MQTT_BROKER,
            port=MQTT_PORT,
            client_id=MQTT_CLIENT_ID,
            keepalive=MQTT_KEEPALIVE,
            pub_log_topic=MQTT_PUB_LOG,
            log_level=log_level,  # Pass log level to Messager
        )

        # Start Ollama monitoring after MQTT connection so we can use the messager
        if backend == "ollama":
            monitor_thread = threading.Thread(
                target=monitor_ollama_status,
                args=(OLLAMA_BASE_URL, ollama_monitor_stop_event, messager),
                daemon=True,
            )
            monitor_thread.start()
            logger.info("Started Ollama monitoring thread")
            messager.mqtt_log("Started Ollama monitoring thread")

        logger.info(f"Initializing LLM backend='{backend}', model='{model_name}'")
        messager.mqtt_log(f"Initializing LLM backend='{backend}', model='{model_name}'")
        llm = LLMFactory.create(llm_config, messager=messager)
        logger.info(f"LLM ({backend}:{model_name}) initialized successfully")
        messager.mqtt_log(f"LLM ({backend}:{model_name}) initialized successfully")

        logger.info("Initializing Agent")
        messager.mqtt_log("Initializing Agent")
        agent: Agent = Agent(
            llm=llm,
            messager=messager,
            log_level=log_level,  # Pass log level to Agent
        )

        # Update handlers to use the new logging approach
        handle_ask_question = mqtt_handler_decorator(
            messager=messager,
            agent=agent,
            pub_response_topic=MQTT_PUB_RESPONSE,
        )(raw_handle_ask_question)

        handle_status_request = mqtt_handler_decorator(
            messager=messager,
            agent=agent,
            pub_status_topic=MQTT_PUB_STATUS,
            llm_model=model_name,
            mqtt_broker=MQTT_BROKER,
            subscribed_topics=list(MQTT_SUBSCRIPTIONS.keys()),
            start_time=start_time,  # Pass start time for uptime calculation
        )(raw_handle_status_request)

        topic_handlers: Dict[str, Callable[[MQTTMessage], None]] = {
            "handle_ask_question": handle_ask_question,
            "handle_status_request": handle_status_request,
        }

        logger.info("Starting MQTT client loop")
        messager.mqtt_log("Starting MQTT client loop")
        messager.start(MQTT_SUBSCRIPTIONS, topic_handlers, wait_for_connection=True)

        logger.info("Initializing tools")
        messager.mqtt_log("Initializing tools")
        agent.tool_manager.initialize_tools()

        # Publish initial status message
        startup_status = {
            "source": messager.client_id,
            "timestamp": time.time(),
            "status": "online",
            "service": "rag_agent",
            "llm_backend": backend,
            "model": model_name,
            "log_level": log_level.name,
        }
        messager.publish(MQTT_PUB_STATUS, json.dumps(startup_status))

        # Block here to keep the agent running until interrupted by Ctrl+C
        logger.info("AI Agent is running. Press Ctrl+C to exit")
        messager.mqtt_log("AI Agent is running. Press Ctrl+C to exit")
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("AI Agent Shutdown Initiated (Ctrl+C)")
        if "messager" in locals() and messager:
            messager.mqtt_log("AI Agent Shutdown Initiated (Ctrl+C)")
        ollama_monitor_stop_event.set()
        if "messager" in locals() and messager and messager.is_connected():
            logger.info("Stopping MQTT client")
            messager.stop()
    except Exception as e:
        logger.critical(f"AI Agent encountered an unhandled error: {e}")
        logger.error(traceback.format_exc())
        if "messager" in locals() and messager:
            messager.mqtt_log(f"AI Agent encountered an unhandled error: {e}", level="critical")
            messager.mqtt_log(f"Traceback: {traceback.format_exc()}", level="error")
        ollama_monitor_stop_event.set()
        if "messager" in locals() and messager and messager.is_connected():
            logger.info("Attempting to stop MQTT client due to error")
            messager.stop()
    finally:
        if monitor_thread and monitor_thread.is_alive():
            logger.info("Waiting for Ollama monitor thread to exit")
            monitor_thread.join(timeout=2)
        logger.info("AI Agent Shutdown Complete")
        if "messager" in locals() and messager:
            try:
                # Try to send final shutdown message
                shutdown_status = {
                    "source": messager.client_id,
                    "timestamp": time.time(),
                    "status": "offline",
                    "service": "rag_agent",
                }
                messager.publish(MQTT_PUB_STATUS, json.dumps(shutdown_status))
                messager.mqtt_log("AI Agent Shutdown Complete")
            except:
                pass


def signal_handler(sig, frame):
    """Handle termination signals"""
    logger.info(f"Signal {sig} received, initiating graceful shutdown")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Agent Service with MQTT Interface")
    parser.add_argument(
        "--start-llm",
        action="store_true",
        help="Start LLM server only (if configured with auto_start=True)",
    )
    parser.add_argument(
        "--start-agent",
        action="store_true",
        help="Start RAG agent (includes background Ollama monitoring if configured)",
    )
    parser.add_argument(
        "--monitor-ollama-only",
        action="store_true",
        help="Run only the Ollama status monitor in the foreground",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["debug", "info", "warning", "error", "critical"],
        help="Set the logging level (overrides config file setting)",
    )
    args = parser.parse_args()

    # Override log level from command line if specified
    if args.log_level:
        new_level = parse_log_level(args.log_level)
        logger.info(f"Setting log level from command line: {new_level.name}")
        set_global_log_level(new_level)
        log_level = new_level  # Update the module-level log_level

    if args.monitor_ollama_only:
        logger.info("Starting Ollama monitor in standalone mode")
        try:
            # Create messager for standalone monitor mode
            monitor_messager = Messager(
                broker_address=MQTT_BROKER,
                port=MQTT_PORT,
                client_id=f"{MQTT_CLIENT_ID}_monitor",
                keepalive=MQTT_KEEPALIVE,
                pub_log_topic=MQTT_PUB_LOG,
                log_level=log_level,
            )
            monitor_messager.connect(start_loop=True)
            monitor_messager.start_background_loop()
            monitor_messager.mqtt_log("Ollama monitor started in standalone mode")

            monitor_ollama_status(OLLAMA_BASE_URL, ollama_monitor_stop_event, monitor_messager)
        except KeyboardInterrupt:
            logger.info("Ollama monitor stopped by user")
            if "monitor_messager" in locals():
                monitor_messager.mqtt_log("Ollama monitor stopped by user")
                monitor_messager.stop()
            sys.exit(0)
    elif args.start_llm:
        logger.info("Starting LLM server mode")
        # Also make sure start_llm_server function exists
        if "start_llm_server" not in globals():
            from core.llm import start_llm_server

        if llm_config.get("backend") == "llamaserver" and llm_config.get("server_binary"):
            logger.info(f"Starting llama.cpp server with binary: {llm_config.get('server_binary')}")
            start_llm_server(llm_config)
        # Add support for Ollama backend with server_binary
        elif llm_config.get("backend") == "ollama" and llm_config.get("server_binary"):
            logger.info(f"Starting Ollama server with binary: {llm_config.get('server_binary')}")
            start_llm_server(llm_config)
        else:
            logger.error(
                "LLM server start requested, but backend is not supported or 'server_binary' is not configured"
            )
            logger.error(
                "Please configure config.yaml with backend='llamaserver' or backend='ollama' and set server_binary path"
            )
            sys.exit(1)
    elif args.start_agent or not (args.start_llm or args.monitor_ollama_only):
        run_rag_agent()
    else:
        parser.print_help()
