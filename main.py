# Standard library imports
import os
import time
import warnings
import argparse
import sys
import requests
import json
import threading
from typing import Any, Callable, Dict, Optional

# Third-party imports
import chromadb
import yaml
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Local application imports
from core.llm import LLMFactory
from core.messager import MQTTMessage, Messager
from core.rag import RAGController
from core.decorators import mqtt_handler_decorator  # Import the decorator factory

# Import RAW handler functions from the new location
from handlers.add_info_handler import handle_add_info as raw_handle_add_info
from handlers.ask_question_handler import handle_ask_question as raw_handle_ask_question
from handlers.forget_info_handler import handle_forget_info as raw_handle_forget_info
from handlers.status_request_handler import (
    handle_status_request as raw_handle_status_request,
)

# --- Load Configuration ---
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

# --- Configuration Values --- (Extracted from loaded config)
# LLM Config
llm_config = config["llm"]
backend: str = llm_config.get("backend", "ollama")
model_name: str = llm_config.get("model_name")
use_chat: bool = llm_config.get("use_chat", False)

# Embedding Config
EMBEDDING_MODEL_NAME: str = config["embedding"]["model_name"]
EMBEDDING_DEVICE: str = config["embedding"]["device"]
EMBEDDING_NORMALIZE: bool = config["embedding"]["normalize"]

# Vector Store Config
VECTORSTORE_DIR: str = config["vector_store"]["directory"]
COLLECTION_NAME: str = config["vector_store"]["collection_name"]

# Retriever Config
RETRIEVER_K: int = config["retriever"]["k"]

# MQTT Config
MQTT_BROKER: str = config["mqtt"]["broker_address"]
MQTT_PORT: int = config["mqtt"]["port"]
MQTT_CLIENT_ID: str = config["mqtt"]["client_id"]
MQTT_KEEPALIVE: int = config["mqtt"]["keepalive"]
MQTT_SUBSCRIPTIONS: Dict[str, str] = config["mqtt"]["subscriptions"]
MQTT_PUB_RESPONSE: str = config["mqtt"]["publish_topics"]["response"]
MQTT_PUB_STATUS: str = config["mqtt"]["publish_topics"]["status"]
MQTT_PUB_LOG: str = config["mqtt"]["publish_topics"]["log"]

# Ollama Config
OLLAMA_BASE_URL = llm_config.get("base_url", "http://localhost:11434")

# Ensure the persistence directory exists
os.makedirs(VECTORSTORE_DIR, exist_ok=True)
print(f"Vector store persistence directory: '{os.path.abspath(VECTORSTORE_DIR)}'")

# --- Command Line Argument Parsing & Execution Modes ---


def start_llm_server(llm_config_dict):
    """Start the LLM server in its own process without running the agent."""
    print("Starting LLM server...")
    # Force auto_start so the server process is launched
    llm_cfg = dict(llm_config_dict, auto_start=True)
    # Create LLM instance which starts the server if configured
    try:
        _ = LLMFactory.create(llm_cfg)
        print("LLM server started (or connection attempted). Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down LLM server process watcher.")
        # Note: Actual server process might need separate termination depending on how it runs
        sys.exit(0)
    except Exception as e:
        print(f"Error initializing LLM for server start: {e}")
        sys.exit(1)


# Flag to control the background monitor thread
ollama_monitor_stop_event = threading.Event()


def monitor_ollama_status(base_url: str, stop_event: threading.Event):
    """Periodically checks the status of the Ollama server API in a background thread."""
    print(f"[Monitor] Starting Ollama status checks at {base_url}...")
    check_interval = 30  # seconds (increased interval for background task)
    while not stop_event.is_set():
        status_message = "Unknown"
        try:
            # Simple check: Can we connect to the base URL?
            response = requests.get(base_url, timeout=5)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            # Check if the response indicates Ollama is running
            if "Ollama is running" in response.text:
                status_message = "Running"
            else:
                # Check specific API endpoint like /api/ps for more detail
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
                    status_message = (
                        f"API endpoint error ({api_err.__class__.__name__})"
                    )
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
            print(f"[Monitor] Unexpected error: {e}")  # Log unexpected errors

        print(
            f"[Monitor] {time.strftime('%Y-%m-%d %H:%M:%S')} - Ollama status: {status_message}"
        )

        # Wait for the interval or until the stop event is set
        stop_event.wait(check_interval)
    print("[Monitor] Ollama status monitoring stopped.")


def run_rag_agent():
    """Initialize components and launch the RAG agent loop."""
    print("\n--- RAG Agent Starting ---")

    monitor_thread = None
    # Start Ollama monitor in background if backend is ollama
    if backend == "ollama":
        monitor_thread = threading.Thread(
            target=monitor_ollama_status,
            args=(OLLAMA_BASE_URL, ollama_monitor_stop_event),
            daemon=True,  # Set as daemon so it exits when main thread exits
        )
        monitor_thread.start()

    # --- Initialize Components ---
    try:
        # 1. Initialize LLM backend
        print(f"Initializing LLM backend='{backend}', model='{model_name}'...")
        llm = LLMFactory.create(llm_config)
        print(f"LLM ({backend}:{model_name}) initialized successfully.")

        # 2. Initialize Embedding Model
        embedding_function: HuggingFaceEmbeddings
        print(f"Initializing Embedding Model: {EMBEDDING_MODEL_NAME}...")
        embedding_function = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": EMBEDDING_NORMALIZE},
        )
        print("Embedding Model initialized.")

        # 3. Initialize Vector Store (ChromaDB)
        db_client: chromadb.PersistentClient
        vectorstore: Chroma
        print("Initializing ChromaDB Vector Store...")
        db_client = chromadb.PersistentClient(path=VECTORSTORE_DIR)
        vectorstore = Chroma(
            client=db_client,
            collection_name=COLLECTION_NAME,
            embedding_function=embedding_function,
            persist_directory=VECTORSTORE_DIR,
        )
        print(f"ChromaDB collection '{COLLECTION_NAME}' loaded/created.")

        # 4. Initialize Messager
        messager: Messager = Messager(
            broker_address=MQTT_BROKER,
            port=MQTT_PORT,
            client_id=MQTT_CLIENT_ID,
            keepalive=MQTT_KEEPALIVE,
            pub_log_topic=MQTT_PUB_LOG,
        )

        # 5. Initialize RAG Controller
        rag_controller: RAGController = RAGController(
            llm=llm,
            vectorstore=vectorstore,
            db_client=db_client,
            retriever_k=RETRIEVER_K,
            messager=messager,
            use_chat=use_chat,
        )

        # --- Dynamically Apply Decorators to Handlers ---
        handle_add_info = mqtt_handler_decorator(
            messager=messager,
            rag_controller=rag_controller,
            pub_status_topic=MQTT_PUB_STATUS,
        )(raw_handle_add_info)

        handle_ask_question = mqtt_handler_decorator(
            messager=messager,
            rag_controller=rag_controller,
            pub_response_topic=MQTT_PUB_RESPONSE,
        )(raw_handle_ask_question)

        handle_forget_info = mqtt_handler_decorator(
            messager=messager,
            rag_controller=rag_controller,
            pub_status_topic=MQTT_PUB_STATUS,
        )(raw_handle_forget_info)

        handle_status_request = mqtt_handler_decorator(
            messager=messager,
            pub_status_topic=MQTT_PUB_STATUS,
            llm_model=model_name,
            embedding_model=EMBEDDING_MODEL_NAME,
            collection_name=COLLECTION_NAME,
            mqtt_broker=MQTT_BROKER,
            subscribed_topics=list(MQTT_SUBSCRIPTIONS.keys()),
        )(raw_handle_status_request)

        topic_handlers: Dict[str, Callable[[MQTTMessage], None]] = {
            "handle_add_info": handle_add_info,
            "handle_ask_question": handle_ask_question,
            "handle_forget_info": handle_forget_info,
            "handle_status_request": handle_status_request,
        }

        # --- Start MQTT Client Loop ---
        print("Starting MQTT client loop...")
        messager.start(MQTT_SUBSCRIPTIONS, topic_handlers, wait_for_connection=True)
        # The start method now blocks until KeyboardInterrupt or error

    except KeyboardInterrupt:
        print("\n\n--- RAG Agent Shutdown Initiated (Ctrl+C) ---")
        # Signal the monitor thread to stop
        ollama_monitor_stop_event.set()
        if "messager" in locals() and messager.is_connected():
            print("Stopping MQTT client...")
            messager.stop()
    except Exception as e:
        print(
            f"\n--- RAG Agent encountered an unhandled error during setup or runtime: {e} ---"
        )
        # Signal the monitor thread to stop
        ollama_monitor_stop_event.set()
        if "messager" in locals() and messager.is_connected():
            print("Attempting to stop MQTT client due to error...")
            messager.stop()
    finally:
        # Wait briefly for the monitor thread to finish if it was started
        if monitor_thread and monitor_thread.is_alive():
            print("Waiting for Ollama monitor thread to exit...")
            monitor_thread.join(timeout=2)  # Wait max 2 seconds
        print("\n--- RAG Agent Shutdown Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAG Agent Service with MQTT Interface"
    )
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

    # Determine execution mode
    if args.monitor_ollama_only:
        # Run monitor in foreground, blocking until Ctrl+C
        try:
            monitor_ollama_status(OLLAMA_BASE_URL, ollama_monitor_stop_event)
        except KeyboardInterrupt:
            print("\nOllama monitor stopped by user.")
            sys.exit(0)
    elif args.start_llm:
        # Check if backend is llamaserver and auto_start is intended
        if llm_config.get("backend") == "llamaserver" and llm_config.get(
            "server_binary"
        ):
            start_llm_server(llm_config)
        else:
            print(
                "LLM server start requested, but backend is not 'llamaserver' or 'server_binary' is not configured."
            )
            print(
                "Please configure config.yaml appropriately if you intend to auto-start the server."
            )
            sys.exit(1)
    elif args.start_agent or not (args.start_llm or args.monitor_ollama_only):
        run_rag_agent()
    else:
        parser.print_help()
