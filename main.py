# Standard library imports
import os
import time
import warnings
from typing import Any, Callable, Dict, Optional

# Third-party imports
import chromadb
import yaml
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM

# Local application imports
from core.decorators import mqtt_handler_decorator
from core.messager import MQTTMessage, Messager
from core.rag import RAGController

# Ignore specific warnings if they are noisy (optional)
warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
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
OLLAMA_MODEL: str = config["llm"]["model_name"]
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

# Ensure the persistence directory exists
os.makedirs(VECTORSTORE_DIR, exist_ok=True)
print(f"Vector store persistence directory: '{os.path.abspath(VECTORSTORE_DIR)}'")

# --- Initialize Components ---

# 1. Initialize LLM via LangChain
llm: OllamaLLM
try:
    print(f"Initializing LLM: {OLLAMA_MODEL} via Ollama...")
    llm = OllamaLLM(model=OLLAMA_MODEL)
    print(f"LLM ({OLLAMA_MODEL}) initialized successfully.")
    print("Ollama should automatically use your NVIDIA GPU if drivers are correct.")
except Exception as e:
    print(f"Error initializing LLM with Ollama: {e}")
    print(
        f"Ensure the Ollama server/application is running and the model '{OLLAMA_MODEL}' is pulled ('ollama pull {OLLAMA_MODEL}')."
    )
    exit()

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
)

# --- MQTT Topic Handlers (Simplified with Decorator) ---

# Create the decorator instance with the messager
handler_decorator = mqtt_handler_decorator(messager)


@handler_decorator
def handle_add_info(data: Dict[str, Any], msg: MQTTMessage) -> None:
    """Handles messages on the 'add_info' topic (core logic)."""
    topic = msg.topic
    text: Optional[str] = data.get("text")
    source: str = data.get("source", "mqtt_input")
    timestamp: Optional[float] = data.get("timestamp")

    if text:
        success: bool = rag_controller.remember(text, source, timestamp)
        status = "processed" if success else "error_remembering"
        messager.publish(
            MQTT_PUB_STATUS,
            {"status": status, "topic": topic, "source": source},
        )
    else:
        messager.publish_log(
            "Received add_info command with missing 'text'.", level="warning"
        )
        messager.publish(
            MQTT_PUB_STATUS,
            {
                "status": "error",
                "topic": topic,
                "error": "Missing 'text' in payload",
            },
        )


@handler_decorator
def handle_ask_question(data: Dict[str, Any], msg: MQTTMessage) -> None:
    """Handles messages on the 'ask_question' topic (core logic)."""
    topic = msg.topic
    question: Optional[str] = data.get("question")
    if question:
        print(f"\n🤔 Processing Question via MQTT: {question}")
        start_time = time.time()
        response: str = rag_controller.query(question)
        end_time = time.time()
        print(f"\n💡 Answer: {response}")
        if "Sorry, an error occurred" in response:
            messager.publish(
                MQTT_PUB_RESPONSE, {"question": question, "error": response}
            )
        else:
            messager.publish(
                MQTT_PUB_RESPONSE, {"question": question, "answer": response}
            )
        print(f"(Response time: {end_time - start_time:.2f} seconds)")
        messager.publish_log(
            f"Processed question '{question}' in {end_time - start_time:.2f}s."
        )
    else:
        messager.publish_log(
            "Received ask_question command with missing 'question'.",
            level="warning",
        )
        messager.publish(MQTT_PUB_RESPONSE, {"error": "Missing 'question' in payload"})


@handler_decorator
def handle_forget_info(data: Dict[str, Any], msg: MQTTMessage) -> None:
    """Handles messages on the 'forget_info' topic (core logic)."""
    topic = msg.topic
    where_filter: Optional[Dict[str, Any]] = data.get("where_filter")

    if where_filter and isinstance(where_filter, dict):
        result: Dict[str, Any] = rag_controller.forget(where_filter)
        messager.publish(
            MQTT_PUB_STATUS,
            {
                "status": result["status"],
                "topic": topic,
                "filter": where_filter,
                "deleted_count": result["deleted_count"],
                "message": result.get("message"),
            },
        )
    else:
        err_msg = "Received forget_info command requires a valid JSON object in 'where_filter'."
        messager.publish_log(err_msg, level="warning")
        messager.publish(
            MQTT_PUB_STATUS,
            {"status": "error", "topic": topic, "error": err_msg},
        )


# Note: Status request might not need JSON parsing, adjust decorator or handler if needed
# For simplicity, keeping it decorated, assuming payload might be empty JSON {} or ignored
@handler_decorator
def handle_status_request(data: Dict[str, Any], msg: MQTTMessage) -> None:
    """Handles messages on the 'status_request' topic (core logic)."""
    topic = msg.topic
    # Payload (data) is likely ignored for status requests, but decorator provides it
    messager.publish_log(
        f"Processing status request for topic {topic}"
    )  # Log moved from wrapper
    status_info: Dict[str, Any] = {
        "status": "running",
        "llm_model": OLLAMA_MODEL,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "vector_store_collection": COLLECTION_NAME,
        "mqtt_broker": MQTT_BROKER,
        "subscribed_topics": list(MQTT_SUBSCRIPTIONS.keys()),
        "timestamp": time.time(),
    }
    messager.publish(MQTT_PUB_STATUS, status_info)


# Map handler names from config to the decorated functions
# Note the change in function signature expected by Messager.start
topic_handlers: Dict[str, Callable[[MQTTMessage], None]] = {
    "handle_add_info": handle_add_info,
    "handle_ask_question": handle_ask_question,
    "handle_forget_info": handle_forget_info,
    "handle_status_request": handle_status_request,
}

# --- Main Execution ---
if __name__ == "__main__":
    print("\n--- RAG Agent Starting ---")

    try:
        # Pass the mapping of *handler names* to *decorated functions*
        messager.start(MQTT_SUBSCRIPTIONS, topic_handlers)
        print("RAG Agent is running. Press Ctrl+C to exit.")
    except KeyboardInterrupt:
        print("\n\n--- RAG Agent Shutdown Initiated ---")
        messager.stop()
        print("RAG Agent has been stopped.")

    print("\n--- RAG Agent Shutdown Complete ---")
