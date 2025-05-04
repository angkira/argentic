import yaml
import time
import signal
import uuid
import json
import threading
from core.messager import Messager, MQTTMessage
from core.rag import RAGManager
from tools.knowledge_base_tool import KnowledgeBaseTool, KnowledgeBaseInput
from core.protocol.message import (
    RegisterToolMessage,
    TaskMessage,
    TaskResultMessage,
    TaskStatus,
    MessageType,
)
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables
import os
from dotenv import load_dotenv

load_dotenv()

HG_TOKEN = os.getenv("HG_TOKEN")

# Load configuration
config = yaml.safe_load(open("config.yaml"))
mqtt_cfg = config["mqtt"]
vec_cfg = config["vector_store"]
embed_cfg = config["embedding"]
retr_cfg = config["retriever"]

print("Configuration loaded successfully.")
print(f"MQTT Broker: {mqtt_cfg['broker_address']}, Port: {mqtt_cfg['port']}")
print(f"Vector Store Directory: {vec_cfg['directory']}")
print(f"Embedding Model: {embed_cfg['model_name']}, Device: {embed_cfg['device']}")
print("Initializing messager and vector store...")
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
messager.start_background_loop()
print("Messager connected and background loop started.")

# Initialize vectorstore and embedding
db_client = chromadb.PersistentClient(path=vec_cfg["directory"])
print("Chroma client initialized.")

print("Initializing embedding model...")

# Use model_name from config instead of pre-loading the model
embedding = HuggingFaceEmbeddings(
    model_name=embed_cfg["model_name"],  # Use the model_name from config
    model_kwargs={
        "device": embed_cfg["device"],
        "token": HG_TOKEN,
    },
    encode_kwargs={"normalize_embeddings": embed_cfg["normalize"]},
)
print(f"Embedding model initialized: {embed_cfg['model_name']}")

print("Initializing vector store...")
vectorstore = Chroma(
    client=db_client,
    collection_name=vec_cfg["collection_name"],
    embedding_function=embedding,
    persist_directory=vec_cfg["directory"],
)
print("Vector store initialized.")

print("Initializing retriever...")
# Use simpler retriever initialization to avoid potential validation errors
retriever = vectorstore.as_retriever(
    search_type=retr_cfg.get("search_type", "mmr"),
    search_kwargs={
        "k": retr_cfg["k"],
        "fetch_k": retr_cfg.get("fetch_k", 20),
    },
)
print("Retriever initialized.")

print("Initializing RAGManager...")
# Initialize RAGManager
rag_manager = RAGManager(
    db_client=db_client,
    retriever_k=retr_cfg["k"],
    messager=messager,
)
print("RAGManager initialized.")

# Create the Knowledge Base Tool instance (but don't use ToolManager)
kb_tool = KnowledgeBaseTool(rag_manager=rag_manager, messager=messager)

# We'll store the assigned tool ID once we get it from the registration confirmation
assigned_tool_id = None
registration_complete = threading.Event()


# Handle the registration confirmation message
def handle_registration_confirmation(message: MQTTMessage):
    global assigned_tool_id
    try:
        from core.protocol.message import ToolRegisteredMessage

        payload = json.loads(message.payload)
        confirm_msg = ToolRegisteredMessage.model_validate(payload)

        if confirm_msg.tool_name == kb_tool.name:
            assigned_tool_id = confirm_msg.tool_id
            messager.log(f"Received tool registration confirmation with ID: {assigned_tool_id}")
            registration_complete.set()
    except Exception as e:
        messager.log(f"Error processing registration confirmation: {str(e)}", level="error")


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
    print(f"Sent direct registration for tool '{kb_tool.name}'")
    print("Waiting for registration confirmation...")

    # Wait for the registration confirmation (with timeout)
    if not registration_complete.wait(timeout=10.0):
        print("Warning: Didn't receive registration confirmation within timeout")
        return None

    return assigned_tool_id


def handle_task_message(message: MQTTMessage):
    """Handle incoming task messages directly"""
    try:
        # Parse incoming message
        payload = json.loads(message.payload)
        task = TaskMessage.model_validate(payload)

        messager.log(f"Received task: {task.task_id} for tool: {task.tool_id}")

        # Parse arguments using the KnowledgeBaseInput schema
        args = KnowledgeBaseInput.model_validate(task.arguments)

        # Execute the tool
        try:
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
        except Exception as e:
            # Create error response
            messager.log(f"Error executing task {task.task_id}: {str(e)}", level="error")
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
        messager.log(f"Published result for task {task.task_id} to {result_topic}")

    except Exception as e:
        messager.log(f"Error handling task message: {str(e)}", level="error")


# Create a function to unregister the tool
def unregister_tool(tool_id, tool_name):
    """Sends an unregistration message for the tool to the Agent"""
    if not tool_id:
        print("No tool ID to unregister")
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
        print(f"Sent unregistration request for tool '{tool_name}' (ID: {tool_id})")
        return True
    except Exception as e:
        print(f"Error sending tool unregistration: {e}")
        return False


# Register the tool and wait for the assigned ID
assigned_tool_id = register_rag_tool()

if assigned_tool_id:
    # Subscribe to the task topic using the assigned ID from the ToolManager
    task_topic = f"tool/{assigned_tool_id}/task"
    messager.subscribe(task_topic, handle_task_message)
    print(f"Subscribed to task topic: {task_topic}")
else:
    print("Error: Failed to get assigned tool ID, cannot subscribe to task topic")

print("RAG Tool Service is running. Press Ctrl+C to exit.")


def shutdown(signum, frame):
    print("Shutting down RAG Tool Service...")

    # Unregister the tool before disconnecting
    if assigned_tool_id:
        print(f"Unregistering tool {kb_tool.name} (ID: {assigned_tool_id})...")
        unregister_tool(assigned_tool_id, kb_tool.name)
        # Give a moment for the message to be sent
        time.sleep(0.5)

    messager.stop()
    print("RAG Tool Service stopped.")
    exit(0)


signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

# Keep the service alive
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    shutdown(None, None)
