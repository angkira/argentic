import os
import time
import argparse
import sys
import requests
import json
import threading
import signal
from typing import Callable, Dict

import yaml

from core.llm import LLMFactory
from core.messager import MQTTMessage, Messager
from core.agent import Agent
from core.decorators import mqtt_handler_decorator

from handlers.ask_question_handler import handle_ask_question as raw_handle_ask_question
from handlers.status_request_handler import (
    handle_status_request as raw_handle_status_request,
)

CONFIG_PATH = "config.yaml"
try:
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    print(f"Configuration loaded successfully from '{CONFIG_PATH}'.")
except FileNotFoundError:
    print(f"Error: Configuration file '{CONFIG_PATH}' not found.")
    exit()
except yaml.YAMLError as e:
    print(f"Error parsing configuration file '{CONFIG_PATH}': {e}")
    exit()

llm_config = config["llm"]
backend: str = llm_config.get("backend", "ollama")
model_name: str = llm_config.get("model_name")
use_chat: bool = llm_config.get("use_chat", False)

EMBEDDING_MODEL_NAME: str = config["embedding"]["model_name"]
EMBEDDING_DEVICE: str = config["embedding"]["device"]
EMBEDDING_NORMALIZE: bool = config["embedding"]["normalize"]

VECTORSTORE_DIR: str = config["vector_store"]["directory"]
COLLECTION_NAME: str = config["vector_store"]["collection_name"]

RETRIEVER_K: int = config["retriever"]["k"]

MQTT_BROKER: str = config["mqtt"]["broker_address"]
MQTT_PORT: int = config["mqtt"]["port"]
MQTT_CLIENT_ID: str = config["mqtt"]["client_id"]
MQTT_KEEPALIVE: int = config["mqtt"]["keepalive"]
MQTT_SUBSCRIPTIONS: Dict[str, str] = config["mqtt"]["subscriptions"]
MQTT_PUB_RESPONSE: str = config["mqtt"]["publish_topics"]["response"]
MQTT_PUB_STATUS: str = config["mqtt"]["publish_topics"]["status"]
MQTT_PUB_LOG: str = config["mqtt"]["publish_topics"]["log"]

OLLAMA_BASE_URL = llm_config.get("base_url", "http://localhost:11434")

os.makedirs(VECTORSTORE_DIR, exist_ok=True)
print(f"Vector store persistence directory: '{os.path.abspath(VECTORSTORE_DIR)}'")

ollama_monitor_stop_event = threading.Event()


def monitor_ollama_status(base_url: str, stop_event: threading.Event):
    print(f"[Monitor] Starting Ollama status checks at {base_url}...")
    check_interval = 30
    while not stop_event.is_set():
        status_message = "Unknown"
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
                except json.JSONDecodeError:
                    status_message = "Running (API response not valid JSON)"
        except requests.exceptions.ConnectionError:
            status_message = "Connection Error (Server down?)"
        except requests.exceptions.Timeout:
            status_message = "Timeout (Server unresponsive)"
        except requests.exceptions.RequestException as e:
            status_message = f"Error ({e.__class__.__name__})"
        except Exception as e:
            status_message = f"Unexpected error: {e}"
            print(f"[Monitor] Unexpected error: {e}")

        print(f"[Monitor] {time.strftime('%Y-%m-%d %H:%M:%S')} - Ollama status: {status_message}")
        stop_event.wait(check_interval)
    print("[Monitor] Ollama status monitoring stopped.")


def run_rag_agent():
    print("\n--- AI Agent Starting ---")

    monitor_thread = None
    if backend == "ollama":
        monitor_thread = threading.Thread(
            target=monitor_ollama_status,
            args=(OLLAMA_BASE_URL, ollama_monitor_stop_event),
            daemon=True,
        )
        monitor_thread.start()

    try:
        messager: Messager = Messager(
            broker_address=MQTT_BROKER,
            port=MQTT_PORT,
            client_id=MQTT_CLIENT_ID,
            keepalive=MQTT_KEEPALIVE,
            pub_log_topic=MQTT_PUB_LOG,
        )

        print(f"Initializing LLM backend='{backend}', model='{model_name}'...")
        llm = LLMFactory.create(llm_config, messager=messager)
        print(f"LLM ({backend}:{model_name}) initialized successfully.")

        agent: Agent = Agent(
            llm=llm,
            messager=messager,
        )

        handle_ask_question = mqtt_handler_decorator(
            messager=messager,
            agent=agent,
            pub_response_topic=MQTT_PUB_RESPONSE,
        )(raw_handle_ask_question)

        handle_status_request = mqtt_handler_decorator(
            messager=messager,
            pub_status_topic=MQTT_PUB_STATUS,
            llm_model=model_name,
            embedding_model=EMBEDDING_MODEL_NAME,
            default_collection_name=COLLECTION_NAME,
            mqtt_broker=MQTT_BROKER,
            subscribed_topics=list(MQTT_SUBSCRIPTIONS.keys()),
        )(raw_handle_status_request)

        topic_handlers: Dict[str, Callable[[MQTTMessage], None]] = {
            "handle_ask_question": handle_ask_question,
            "handle_status_request": handle_status_request,
        }

        print("Starting MQTT client loop...")
        messager.start(MQTT_SUBSCRIPTIONS, topic_handlers, wait_for_connection=True)

        agent.tool_manager.initialize_tools()

        # Block here to keep the agent running until interrupted by Ctrl+C
        print("\n--- AI Agent is running. Press Ctrl+C to exit ---")
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\n--- AI Agent Shutdown Initiated (Ctrl+C) ---")
        ollama_monitor_stop_event.set()
        if "messager" in locals() and messager.is_connected():
            print("Stopping MQTT client...")
            messager.stop()
    except Exception as e:
        print(f"\n--- AI Agent encountered an unhandled error during setup or runtime: {e} ---")
        ollama_monitor_stop_event.set()
        if "messager" in locals() and messager.is_connected():
            print("Attempting to stop MQTT client due to error...")
            messager.stop()
    finally:
        if monitor_thread and monitor_thread.is_alive():
            print("Waiting for Ollama monitor thread to exit...")
            monitor_thread.join(timeout=2)
        print("\n--- AI Agent Shutdown Complete ---")


def signal_handler(sig, frame):
    print("\nSignal received, initiating graceful shutdown...")
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
    args = parser.parse_args()

    if args.monitor_ollama_only:
        try:
            monitor_ollama_status(OLLAMA_BASE_URL, ollama_monitor_stop_event)
        except KeyboardInterrupt:
            print("\nOllama monitor stopped by user.")
            sys.exit(0)
    elif args.start_llm:
        # Also make sure start_llm_server function exists
        if "start_llm_server" not in globals():
            from core.llm import start_llm_server

        if llm_config.get("backend") == "llamaserver" and llm_config.get("server_binary"):
            start_llm_server(llm_config)
        # Add support for Ollama backend with server_binary
        elif llm_config.get("backend") == "ollama" and llm_config.get("server_binary"):
            print(f"Starting Ollama server with binary: {llm_config.get('server_binary')}")
            start_llm_server(llm_config)
        else:
            print(
                "LLM server start requested, but backend is not supported or 'server_binary' is not configured."
            )
            print(
                "Please configure config.yaml with backend='llamaserver' or backend='ollama' and set server_binary path."
            )
            sys.exit(1)
    elif args.start_agent or not (args.start_llm or args.monitor_ollama_only):
        run_rag_agent()
    else:
        parser.print_help()
