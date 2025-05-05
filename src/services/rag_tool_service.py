import yaml
import time
import signal
import json
import traceback
import threading
import os
from core.messager import Messager, MQTTMessage
from tools.RAG.rag import RAGManager
from tools.RAG.knowledge_base_tool import KnowledgeBaseTool, KnowledgeBaseInput
from core.logger import get_logger, LogLevel, parse_log_level
from core.protocol.message import (
    RegisterToolMessage,
    TaskMessage,
    TaskResultMessage,
    TaskStatus,
    ToolRegisteredMessage,
    MessageType,
)
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Setup logger
logger = get_logger("rag_tool_service", LogLevel.INFO)

HG_TOKEN = os.getenv("HG_TOKEN")

# Load main configuration
config = yaml.safe_load(open("config.yaml"))
mqtt_cfg = config["mqtt"]

# Load dedicated RAG configuration
rag_config_path = os.path.join("src", "tools", "RAG", "rag_config.yaml")
rag_config = yaml.safe_load(open(rag_config_path))
embed_cfg = rag_config["embedding"]
vec_cfg = rag_config["vector_store"]
default_retriever_cfg = rag_config["default_retriever"]
collections_cfg = rag_config.get("collections", {})

# Get log level from config or default to INFO
log_level_str = config.get("logging", {}).get("level", "debug")
log_level = parse_log_level(log_level_str)
logger.setLevel(log_level.value)

logger.info("Configuration loaded successfully.")
logger.info(f"MQTT Broker: {mqtt_cfg['broker_address']}, Port: {mqtt_cfg['port']}")
logger.info(f"Vector Store Directory: {vec_cfg['base_directory']}")
logger.info(f"Default Collection: {vec_cfg['default_collection']}")
logger.info(f"Available Collections: {', '.join(collections_cfg.keys())}")
logger.info(f"Embedding Model: {embed_cfg['model_name']}, Device: {embed_cfg['device']}")
logger.info(f"Log level: {log_level.name}")
logger.info("Initializing messager and vector store...")

# Initialize Messager
messager = Messager(
    broker_address=mqtt_cfg["broker_address"],
    port=mqtt_cfg["port"],
    client_id=mqtt_cfg.get("tool_client_id", "rag_tool_service"),
    keepalive=mqtt_cfg["keepalive"],
    pub_log_topic=mqtt_cfg["publish_topics"]["log"],
)
# Connect and start loop
messager.connect(start_loop=True)
logger.info("Messager connected and background loop started.")
messager.mqtt_log("RAG Tool Service: Messager connected and background loop started.")

# Initialize vectorstore and embedding
try:
    db_client = chromadb.PersistentClient(path=vec_cfg["base_directory"])
    logger.info("Chroma client initialized.")
    messager.mqtt_log("RAG Tool Service: Chroma client initialized.")

    logger.info("Initializing embedding model...")
    messager.mqtt_log("RAG Tool Service: Initializing embedding model...")

    # Use model_name from config instead of pre-loading the model
    embedding = HuggingFaceEmbeddings(
        model_name=embed_cfg["model_name"],  # Use the model_name from config
        model_kwargs={
            "device": embed_cfg["device"],
            "token": HG_TOKEN,
        },
        encode_kwargs={"normalize_embeddings": embed_cfg["normalize"]},
    )
    logger.info(f"Embedding model initialized: {embed_cfg['model_name']}")
    messager.mqtt_log(f"RAG Tool Service: Embedding model initialized: {embed_cfg['model_name']}")

    logger.info("Initializing RAGManager...")
    messager.mqtt_log("RAG Tool Service: Initializing RAGManager...")
    # Initialize RAGManager with the default collection from config
    rag_manager = RAGManager(
        db_client=db_client,
        retriever_k=default_retriever_cfg["k"],
        messager=messager,
        embedding_function=embedding,
        default_collection_name=vec_cfg["default_collection"],
    )

    # Initialize all collections defined in the config
    for collection_name, collection_config in collections_cfg.items():
        logger.info(f"Initializing collection: {collection_name}")
        messager.mqtt_log(f"RAG Tool Service: Initializing collection: {collection_name}")

        # Get collection-specific retriever settings or use defaults
        retriever_cfg = collection_config.get("retriever", default_retriever_cfg)
        retriever_k = retriever_cfg.get("k", default_retriever_cfg["k"])

        # Get or create the collection
        vectorstore = rag_manager.get_or_create_collection(collection_name)
        logger.info(f"Collection {collection_name} initialized")
        messager.mqtt_log(f"RAG Tool Service: Collection {collection_name} initialized")

    logger.info("All RAG collections initialized.")
    messager.mqtt_log("RAG Tool Service: All RAG collections initialized.")
except Exception as e:
    logger.error(f"Error initializing components: {e}")
    logger.error(traceback.format_exc())
    messager.mqtt_log(f"RAG Tool Service: Error initializing components: {e}", level="error")
    raise

# Create the Knowledge Base Tool instance (but don't use ToolManager)
kb_tool = KnowledgeBaseTool(rag_manager=rag_manager, messager=messager)

# We'll store the assigned tool ID once we get it from the registration confirmation
assigned_tool_id = None
registration_complete = threading.Event()


# Handle the registration confirmation message
def handle_registration_confirmation(message: MQTTMessage):
    global assigned_tool_id
    try:
        payload = json.loads(message.payload)

        # Check the message type
        msg_type = payload.get("type")

        if msg_type == MessageType.TOOL_REGISTERED:
            # Handle registration confirmation
            confirm_msg = ToolRegisteredMessage.model_validate(payload)

            if confirm_msg.tool_name == kb_tool.name:
                assigned_tool_id = confirm_msg.tool_id
                logger.info(f"Received tool registration confirmation with ID: {assigned_tool_id}")
                messager.mqtt_log(
                    f"RAG Tool Service: Received tool registration confirmation with ID: {assigned_tool_id}"
                )
                registration_complete.set()
        elif msg_type == "TOOL_UNREGISTERED":
            # Handle unregistration confirmation - no validation needed
            tool_name = payload.get("tool_name")
            logger.info(f"Tool '{tool_name}' successfully unregistered")
            messager.mqtt_log(f"RAG Tool Service: Tool '{tool_name}' successfully unregistered")
            if tool_name == kb_tool.name:
                assigned_tool_id = None
        else:
            logger.debug(f"Ignoring message with type: {msg_type}")

    except Exception as e:
        logger.error(f"Error processing registration confirmation: {e}")
        logger.error(traceback.format_exc())
        messager.mqtt_log(
            f"RAG Tool Service: Error processing registration confirmation: {e}", level="error"
        )


# Register to receive registration confirmations
messager.subscribe("agent/status/info", handle_registration_confirmation)


# Register the tool directly using the protocol messages
def register_rag_tool():
    # Create the RegisterToolMessage
    schema_str = json.dumps(kb_tool.argument_schema.model_json_schema())
    reg_msg = RegisterToolMessage(
        source=messager.client_id,
        tool_name=kb_tool.name,
        tool_manual=kb_tool.description,
        tool_api=schema_str,
    )

    # Publish to the registration topic
    messager.publish("rag/command/register_tool", reg_msg.model_dump_json())
    logger.info(f"Sent direct registration for tool '{kb_tool.name}'")
    messager.mqtt_log(f"RAG Tool Service: Sent direct registration for tool '{kb_tool.name}'")
    logger.info("Waiting for registration confirmation...")
    messager.mqtt_log("RAG Tool Service: Waiting for registration confirmation...")

    # Wait for the registration confirmation (with timeout)
    if not registration_complete.wait(timeout=10.0):
        logger.warning("Didn't receive registration confirmation within timeout")
        messager.mqtt_log(
            "RAG Tool Service: Didn't receive registration confirmation within timeout",
            level="warning",
        )
        return None

    return assigned_tool_id


def handle_task_message(message: MQTTMessage):
    """Handle incoming task messages directly"""
    logger.debug(f"Received raw message on task topic {message.topic}: {message.payload[:200]}...")
    try:
        # Parse incoming message
        payload = json.loads(message.payload)
        task = TaskMessage.model_validate(payload)

        logger.info(f"Received task: {task.task_id} for tool: {task.tool_id}")
        messager.mqtt_log(
            f"RAG Tool Service: Received task: {task.task_id} for tool: {task.tool_id}"
        )

        # Log task details at debug level
        logger.debug(f"Task payload: {task.payload}")

        # Parse arguments using the KnowledgeBaseInput schema
        args = KnowledgeBaseInput.model_validate(task.payload)

        try:
            logger.info(f"Executing action '{args.action}' with query: '{args.query}'")
            messager.mqtt_log(
                f"RAG Tool Service: Executing action '{args.action}' with query: '{args.query}'"
            )

            result = kb_tool._execute(
                action=args.action,
                query=args.query,
                collection_name=args.collection_name,
                content_to_add=args.content_to_add,
                where_filter=args.where_filter,
            )

            # Create success response
            response = TaskResultMessage(
                source=messager.client_id,
                task_id=task.task_id,
                tool_id=task.tool_id,
                status=TaskStatus.SUCCESS,
                result=result,
            )
            logger.info(f"Task {task.task_id} completed successfully")
            messager.mqtt_log(f"RAG Tool Service: Task {task.task_id} completed successfully")
        except Exception as e:
            # Create error response
            logger.error(f"Error executing task {task.task_id}: {e}")
            logger.error(traceback.format_exc())
            messager.mqtt_log(
                f"RAG Tool Service: Error executing task {task.task_id}: {e}", level="error"
            )
            response = TaskResultMessage(
                source=messager.client_id,
                task_id=task.task_id,
                tool_id=task.tool_id,
                status=TaskStatus.ERROR,
                error=str(e),
            )

        # Publish result to the result topic
        result_topic = f"tool/{task.tool_id}/result"
        messager.publish(result_topic, response.model_dump_json())
        logger.info(f"Published result for task {task.task_id} to {result_topic}")
        messager.mqtt_log(
            f"RAG Tool Service: Published result for task {task.task_id} to {result_topic}"
        )

    except Exception as e:
        logger.error(f"Error handling task message: {e}")
        logger.error(traceback.format_exc())
        messager.mqtt_log(f"RAG Tool Service: Error handling task message: {e}", level="error")


# Create a function to unregister the tool
def unregister_tool(tool_id, tool_name):
    """Sends an unregistration message for the tool to the Agent"""
    if not tool_id:
        logger.warning("No tool ID to unregister")
        return False

    try:
        # Import here to avoid circular imports
        from core.protocol.message import UnregisterToolMessage

        # Create the unregistration message
        unreg_msg = UnregisterToolMessage(
            source=messager.client_id, tool_id=tool_id, tool_name=tool_name
        )

        # Publish to the registration topic (same topic as registration)
        messager.publish("rag/command/register_tool", unreg_msg.model_dump_json())
        logger.info(f"Sent unregistration request for tool '{tool_name}' (ID: {tool_id})")
        messager.mqtt_log(
            f"RAG Tool Service: Sent unregistration request for tool '{tool_name}' (ID: {tool_id})"
        )
        return True
    except Exception as e:
        logger.error(f"Error sending tool unregistration: {e}")
        logger.error(traceback.format_exc())
        messager.mqtt_log(
            f"RAG Tool Service: Error sending tool unregistration: {e}", level="error"
        )
        return False


# Register the tool and wait for the assigned ID
assigned_tool_id = register_rag_tool()

if assigned_tool_id:
    # Subscribe to the task topic using the assigned ID from the ToolManager
    task_topic = f"tool/{assigned_tool_id}/task"
    messager.subscribe(task_topic, handle_task_message)
    logger.info(f"Subscribed to task topic: {task_topic}")
    messager.mqtt_log(f"RAG Tool Service: Subscribed to task topic: {task_topic}")
else:
    logger.error("Failed to get assigned tool ID, cannot subscribe to task topic")
    messager.mqtt_log(
        "RAG Tool Service: Failed to get assigned tool ID, cannot subscribe to task topic",
        level="error",
    )

logger.info("RAG Tool Service is running. Press Ctrl+C to exit.")
messager.mqtt_log("RAG Tool Service is running.")


def shutdown(signum, frame):
    logger.info("Shutting down RAG Tool Service...")
    messager.mqtt_log("RAG Tool Service: Shutting down...")

    # Unregister the tool before disconnecting
    if assigned_tool_id:
        logger.info(f"Unregistering tool {kb_tool.name} (ID: {assigned_tool_id})...")
        messager.mqtt_log(
            f"RAG Tool Service: Unregistering tool {kb_tool.name} (ID: {assigned_tool_id})..."
        )
        unregister_tool(assigned_tool_id, kb_tool.name)
        # Give a moment for the message to be sent
        time.sleep(0.5)

    messager.stop()
    logger.info("RAG Tool Service stopped.")
    messager.mqtt_log("RAG Tool Service stopped.")
    exit(0)


signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

# Keep the service alive
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    shutdown(None, None)
