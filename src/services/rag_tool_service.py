import asyncio
import yaml
import signal
import json
import os
import chromadb
from typing import Optional

from core.messager import Messager
from tools.RAG.rag import RAGManager
from tools.RAG.knowledge_base_tool import KnowledgeBaseTool
from core.logger import get_logger, LogLevel, parse_log_level
from core.protocol.message import (
    RegisterToolMessage,
    TaskMessage,
    TaskResultMessage,
    TaskStatus,
    ToolRegisteredMessage,
)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings

from dotenv import load_dotenv

load_dotenv()

logger = get_logger("rag_tool_service", LogLevel.INFO)

HG_TOKEN = os.getenv("HG_TOKEN")

# --- Configuration Loading ---
config = yaml.safe_load(open("config.yaml"))
mqtt_cfg = config["mqtt"]

rag_config_path = os.path.join("src", "tools", "RAG", "rag_config.yaml")
rag_config = yaml.safe_load(open(rag_config_path))
embed_cfg = rag_config["embedding"]
vec_cfg = rag_config["vector_store"]
default_retriever_cfg = rag_config["default_retriever"]
collections_cfg = rag_config.get("collections", {})

log_level_str = config.get("logging", {}).get("level", "debug")
log_level = parse_log_level(log_level_str)
logger.setLevel(log_level.value)

logger.info("Configuration loaded successfully.")
logger.info(f"MQTT Broker: {mqtt_cfg['broker_address']}, Port: {mqtt_cfg['port']}")
logger.info(f"Vector Store Directory: {vec_cfg['base_directory']}")
logger.info(f"Default Collection: {vec_cfg['default_collection']}")
logger.info(f"Embedding Model: {embed_cfg['model_name']}, Device: {embed_cfg['device']}")
logger.info(f"Log level: {log_level.name}")
logger.info("Initializing messager and RAG components...")

# --- Global Variables ---
messager: Optional[Messager] = None
kb_tool: Optional[KnowledgeBaseTool] = None
rag_manager: Optional[RAGManager] = None
assigned_tool_id: Optional[str] = None
stop_event = asyncio.Event()


# --- Async Handlers ---
async def handle_registration_confirmation(message: ToolRegisteredMessage):
    """Handles confirmation that the tool was registered by the agent."""
    global assigned_tool_id
    try:
        logger.info(f"Received registration confirmation: {message}")

        if kb_tool and assigned_tool_id is None and message.tool_name == kb_tool.name:
            assigned_tool_id = message.tool_id
            logger.info(
                f"Tool '{kb_tool.name}' successfully registered with ID: {assigned_tool_id}"
            )
            task_topic = f"tool/{assigned_tool_id}/task"
            await messager.subscribe(task_topic, handle_task_message)
            logger.info(f"Subscribed to task topic: {task_topic}")
        else:
            logger.debug(
                f"Received confirmation for unrelated tool, uninitialized tool, or already assigned: {message.tool_name}"
            )

    except Exception as e:
        logger.error(f"Error handling registration confirmation: {e}", exc_info=True)


async def handle_task_message(message: TaskMessage):
    """Handles incoming task execution requests for the RAG tool."""
    if not kb_tool:
        logger.error("KnowledgeBaseTool not initialized, cannot handle task.")
        return

    try:
        logger.info(f"Received task message: {message}")

        status = TaskStatus.ERROR
        result_payload = {}

        if message.tool_id != assigned_tool_id:
            logger.warning(
                f"Received task for incorrect tool ID {message.tool_id}, expected {assigned_tool_id}. Ignoring."
            )
            return

        # message.payload should be the input for the tool
        result = await kb_tool.run(message.payload)
        result_payload = result
        status = TaskStatus.SUCCESS
        logger.info(f"Task {message.task_id} completed successfully.")

    except Exception as e:
        logger.error(
            f"Error executing RAG tool task {getattr(message, 'task_id', 'N/A')}: {e}",
            exc_info=True,
        )
        result_payload = {"error": f"Internal server error: {e}"}
        status = TaskStatus.ERROR

    # Always send a result if we have a valid message
    result_topic = f"tool/{message.tool_id}/result"
    result_message = TaskResultMessage(
        source=messager.client_id,
        task_id=message.task_id,
        status=status,
        result=result_payload,
    )
    try:
        await messager.publish(result_topic, result_message.model_dump_json())
        logger.debug(f"Sent result for task {message.task_id} to {result_topic}")
    except Exception as e:
        logger.error(f"Failed to publish result for task {message.task_id}: {e}")


async def register_rag_tool():
    """Registers the RAG tool with the agent/tool manager."""
    if not kb_tool or not messager:
        logger.error("Cannot register tool: Messager or KB Tool not initialized.")
        return

    logger.info(f"Attempting to register tool: {kb_tool.name}")
    try:
        # Check attribute name for schema in KnowledgeBaseTool
        schema_attr = getattr(kb_tool, "args_schema", getattr(kb_tool, "argument_schema", None))
        if schema_attr is None:
            logger.error(
                "Could not find schema attribute (args_schema or argument_schema) on KnowledgeBaseTool."
            )
            return
        # Get the schema as a dictionary
        api_schema_dict = schema_attr.model_json_schema()
        # Convert the dictionary to a JSON string
        api_schema_str = json.dumps(api_schema_dict)

        registration_msg = RegisterToolMessage(
            source=messager.client_id,
            tool_name=kb_tool.name,
            tool_manual=kb_tool.description,
            tool_api=api_schema_str,  # Pass the JSON string
        )
        # Use correct topic from config for registration
        register_topic = mqtt_cfg["subscriptions"].get("tool_register", "agent/tools/register")
        await messager.publish(register_topic, registration_msg.model_dump_json())
        logger.info(f"Sent registration message for '{kb_tool.name}' to topic '{register_topic}'")
    except AttributeError as ae:
        logger.error(f"Error getting tool attributes for registration: {ae}", exc_info=True)
    except Exception as e:
        logger.error(f"Failed to send registration message: {e}", exc_info=True)


async def shutdown_handler():
    """Graceful shutdown handler."""
    logger.info("Shutdown initiated...")
    stop_event.set()
    if messager and messager.is_connected():
        logger.info("Stopping messager...")
        try:
            await messager.stop()
            logger.info("Messager stopped.")
        except Exception as e:
            logger.error(f"Error stopping messager: {e}")

    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if tasks:
        logger.info(f"Cancelling {len(tasks)} outstanding tasks...")
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("Outstanding tasks cancelled.")
    else:
        logger.info("No outstanding tasks to cancel.")


async def main():
    """Main async function for the RAG Tool Service."""
    global messager, kb_tool, rag_manager

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown_handler()))

    messager = Messager(
        broker_address=mqtt_cfg["broker_address"],
        port=mqtt_cfg["port"],
        client_id=mqtt_cfg.get("tool_client_id", "rag_tool_service"),
        keepalive=mqtt_cfg["keepalive"],
        pub_log_topic=mqtt_cfg["topics"]["log"],
        log_level=log_level,
    )

    logger.info("Connecting messager...")
    try:
        if not await messager.connect():
            logger.critical("Messager connection failed. Exiting.")
            return
        await messager.log("RAG Tool Service: Messager connected.")
        logger.info("Messager connected.")

        logger.info("Initializing embeddings...")
        embeddings: Embeddings = HuggingFaceEmbeddings(
            model_name=embed_cfg["model_name"],
            model_kwargs={"device": embed_cfg["device"]},
            encode_kwargs={"normalize_embeddings": embed_cfg["normalize"]},
        )
        logger.info("Embeddings initialized.")

        logger.info("Initializing ChromaDB client...")
        db_client = chromadb.PersistentClient(path=vec_cfg["base_directory"])
        logger.info(f"ChromaDB client initialized with path: {vec_cfg['base_directory']}")

        retriever_k = default_retriever_cfg.get("k", 3)
        default_collection = vec_cfg.get("default_collection", "default_rag_collection")

        logger.info("Initializing RAGManager...")
        rag_manager = RAGManager(
            db_client=db_client,
            retriever_k=retriever_k,
            messager=messager,
            embedding_function=embeddings,
            default_collection_name=default_collection,
        )
        await rag_manager.async_init()
        logger.info("RAGManager initialized asynchronously.")

        kb_tool = KnowledgeBaseTool(rag_manager=rag_manager, messager=messager)
        logger.info(f"KnowledgeBaseTool instance created: {kb_tool.name}")

        status_topic = "agent/status/info"
        await messager.subscribe(status_topic, handle_registration_confirmation)
        logger.info(f"Subscribed to registration confirmation topic: {status_topic}")

        await register_rag_tool()

        logger.info("RAG Tool Service running... Press Ctrl+C to exit.")
        await stop_event.wait()

    except asyncio.CancelledError:
        logger.info("Main task cancelled during execution.")
    except Exception as e:
        logger.critical(f"Unhandled error in main: {e}", exc_info=True)
        if messager and messager.is_connected():
            try:
                await messager.log(f"RAG Tool Service: Critical error: {e}", level="critical")
            except Exception as log_e:
                logger.error(f"Failed to log critical error via MQTT: {log_e}")
    finally:
        logger.info("Main function finished or errored. Cleaning up...")
        if messager and messager.is_connected():
            logger.info("Ensuring messager is stopped in finally block...")
            try:
                await messager.stop()
            except asyncio.CancelledError:
                logger.info("Messager stop cancelled during shutdown.")
            except Exception as e:
                logger.error(f"Error stopping messager in finally block: {e}")
        logger.info("RAG Tool Service cleanup complete.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt caught in __main__. Service shutting down.")
    except Exception as e:
        logger.critical(f"Critical error outside asyncio.run: {e}", exc_info=True)
    finally:
        logger.info("RAG Tool Service process exiting.")
