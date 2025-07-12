import asyncio
import yaml
import signal
import os
import chromadb
from typing import Optional

from argentic.core.messager.messager import Messager
from argentic.tools.RAG.rag import RAGManager
from argentic.tools.RAG.knowledge_base_tool import KnowledgeBaseTool
from argentic.core.logger import get_logger, parse_log_level

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings

from dotenv import load_dotenv

load_dotenv()

# --- Configuration Loading ---
config = yaml.safe_load(open("config.yaml"))
messaging_cfg = config["messaging"]
topic_cfg = config.get("topics", {})

rag_config_path = os.path.join("src", "tools", "RAG", "rag_config.yaml")
rag_config = yaml.safe_load(open(rag_config_path))
embed_cfg = rag_config["embedding"]
vec_cfg = rag_config["vector_store"]
default_retriever_cfg = rag_config["default_retriever"]
collections_cfg = rag_config.get("collections", {})

log_level_str = config.get("logging", {}).get("level", "debug")
log_level = parse_log_level(log_level_str)
logger = get_logger("rag_tool_service", log_level)

logger.info("Configuration loaded successfully.")
logger.info(f"MQTT Broker: {messaging_cfg['broker_address']}, Port: {messaging_cfg['port']}")
logger.info(f"Vector Store Directory: {vec_cfg['base_directory']}")
logger.info(f"Default Collection: {vec_cfg['default_collection']}")
logger.info(f"Embedding Model: {embed_cfg['model_name']}, Device: {embed_cfg['device']}")
logger.info(f"Log level: {log_level.name}")
logger.info("Initializing messager and RAG components...")

# --- Global Variables ---
messager: Optional[Messager] = None
kb_tool: Optional[KnowledgeBaseTool] = None
rag_manager: Optional[RAGManager] = None
stop_event = asyncio.Event()


async def shutdown_handler():
    """Graceful shutdown handler."""
    logger.info("Shutdown initiated...")

    await kb_tool.unregister()
    logger.info("KnowledgeBaseTool unregistered.")
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
        protocol=messaging_cfg["protocol"],
        broker_address=messaging_cfg["broker_address"],
        port=messaging_cfg["port"],
        client_id=messaging_cfg.get("tool_client_id", "rag_tool_service"),
        keepalive=messaging_cfg["keepalive"],
        pub_log_topic=topic_cfg["log"],
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
        try:
            embeddings: Embeddings = HuggingFaceEmbeddings(
                model_name=embed_cfg["model_name"],
                model_kwargs={"device": embed_cfg["device"]},
                encode_kwargs={"normalize_embeddings": embed_cfg["normalize"]},
            )

            logger.info("Embeddings initialized.")
        except Exception as e:
            logger.critical(f"Failed to initialize embeddings: {e}", exc_info=True)
            await messager.log(
                f"RAG Tool Service: Embeddings initialization failed: {e}", level="critical"
            )
            return

        logger.info("Initializing ChromaDB client...")
        try:
            db_client = chromadb.PersistentClient(path=vec_cfg["base_directory"])

            logger.info(f"ChromaDB client initialized with path: {vec_cfg['base_directory']}")
        except Exception as e:
            logger.critical(f"Failed to initialize ChromaDB client: {e}", exc_info=True)
            await messager.log(
                f"RAG Tool Service: ChromaDB client initialization failed: {e}", level="critical"
            )
            return

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

        kb_tool = KnowledgeBaseTool(messager=messager, rag_manager=rag_manager)

        logger.info(f"KnowledgeBaseTool instance created: {kb_tool.name}")

        # Register the tool: publish on register_topic and listen on Agent's status topic
        tools_topics = topic_cfg.get("tools", {})
        register_topic = tools_topics.get("register", "agent/tools/register")
        call_topic_base = tools_topics.get("call", "agent/tools/call")
        response_topic_base = tools_topics.get("response_base", "agent/tools/response")
        # Agent publishes confirmations on responses.status (e.g. 'agent/status/info')
        status_topic = topic_cfg.get("responses", {}).get("status", "agent/status/info")
        await kb_tool.register(register_topic, status_topic, call_topic_base, response_topic_base)

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
