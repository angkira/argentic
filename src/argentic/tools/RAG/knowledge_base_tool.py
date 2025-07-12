import time
import json
from typing import List, Dict, Any, Optional, Union
from enum import Enum

from pydantic import BaseModel, Field
from langchain.docstore.document import Document

# Assuming RAGManager and Messager are accessible or passed during initialization
from argentic.tools.RAG.rag import RAGManager
from argentic.core.messager.messager import Messager
from argentic.core.tools.tool_base import BaseTool  # Import BaseTool
from argentic.core.logger import get_logger, LogLevel, parse_log_level


# --- Argument Schema --- Define actions
class KBAction(str, Enum):
    REMIND = "remind"  # Retrieve information
    REMEMBER = "remember"  # Add information to the knowledge base
    FORGET = "forget"  # Remove information from the knowledge base
    LIST_COLLECTIONS = "list_collections"  # List available collections


class KnowledgeBaseInput(BaseModel):
    action: KBAction = Field(
        description="The action to perform: remind (retrieve info), remember (add info), forget (remove info), list_collections (get available collections)."
    )
    query: Optional[str] = Field(
        None, description="The specific question or topic to search for when action is 'remind'."
    )
    collection_name: Optional[str] = Field(
        None, description="Optional name of a specific collection to search within."
    )
    content_to_add: Optional[str] = Field(
        None, description="Content to add when action is 'remember'."
    )
    where_filter: Optional[Dict[str, Any]] = Field(
        None, description="Metadata filter dict when action is 'forget'."
    )

    # Add validator to ensure required fields are present based on action
    def model_post_init(self, __context):
        if self.action == KBAction.REMIND and not self.query:
            raise ValueError("'query' field is required when action is 'remind'")
        elif self.action == KBAction.REMEMBER and not self.content_to_add:
            raise ValueError("'content_to_add' field is required when action is 'remember'")
        elif self.action == KBAction.FORGET and not self.where_filter:
            raise ValueError("'where_filter' field is required when action is 'forget'")
        # LIST_COLLECTIONS action doesn't require any additional parameters


# --- Helper Function --- (Moved from old implementation)
def format_docs_for_tool_output(docs: List[Document]) -> str:
    """Formats retrieved documents for the LLM, including metadata."""
    if not docs:
        return "No relevant information found in the knowledge base for the query."

    formatted_docs = []
    for _, doc in enumerate(docs):
        ts_unix = doc.metadata.get("timestamp", 0)
        ts_str = "N/A"
        if ts_unix:
            try:
                ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(ts_unix)))
            except (ValueError, TypeError):
                ts_str = f"Invalid timestamp ({ts_unix})"
        source = doc.metadata.get("source", "unknown")
        collection = doc.metadata.get("collection", "unknown")
        formatted_docs.append(
            f"Source: {source}, Collection: {collection}, Time: {ts_str}\nContent: {doc.page_content}"
        )
    return "\n---\n".join(formatted_docs)


# --- Tool Implementation --- Inherit from BaseTool
class KnowledgeBaseTool(BaseTool):
    id: str = None

    def __init__(
        self,
        messager: Messager,
        rag_manager: Optional[RAGManager] = None,
        log_level: Union[LogLevel, str] = LogLevel.INFO,
    ):
        """rag_manager is optional when instantiating in Agent for prompt-only tools."""
        # Build JSON API schema for tool registration
        api_schema = KnowledgeBaseInput.model_json_schema()
        super().__init__(
            name="knowledge_base_tool",
            manual=(
                "Manages the knowledge base. Use it only when you need some specific information, local context, user preferences, etc."
                "Use 'remind' to search for information relevant to a query. Specify the query and optionally a collection name. "
                "Use 'remember' to add new information to the knowledge base. Provide content_to_add parameter with the text to store. "
                "Use 'forget' to remove information from the knowledge base with a where_filter. "
                "Use 'list_collections' to get a list of all available collections."
            ),
            api=json.dumps(api_schema),
            argument_schema=KnowledgeBaseInput,
            messager=messager,
        )
        self.rag_manager = rag_manager

        # Set up logger
        if isinstance(log_level, str):
            self.log_level = parse_log_level(log_level)
        else:
            self.log_level = log_level

        self.logger = get_logger("kb_tool", self.log_level)
        self.logger.info("KnowledgeBaseTool instance created")

    def set_log_level(self, level: Union[LogLevel, str]) -> None:
        """Set the log level for the tool"""
        if isinstance(level, str):
            self.log_level = parse_log_level(level)
        else:
            self.log_level = level

        self.logger.setLevel(self.log_level.value)
        self.logger.info(f"Log level changed to {self.log_level.name}")

        # Update handlers
        for handler in self.logger.handlers:
            handler.setLevel(self.log_level.value)

    async def _execute(
        self,
        action: KBAction,
        query: Optional[str] = None,
        collection_name: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """Executes the requested action on the knowledge base."""
        self.logger.info(f"Executing action: {action.value}")

        if action == KBAction.REMIND:
            if not query:
                error_msg = "'query' argument is required for the 'remind' action."
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            self.logger.info(
                f"Retrieving from collection '{collection_name or 'default'}' with query: '{query[:50]}...'"
            )
            docs = await self.rag_manager.retrieve(query=query, collection_name=collection_name)
            formatted_result = format_docs_for_tool_output(docs)
            self.logger.info(f"Found {len(docs)} documents for query")

            return formatted_result

        elif action == KBAction.REMEMBER:
            content = kwargs.get("content_to_add")
            if not content:
                error_msg = "'content_to_add' argument is required for the 'remember' action."
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            self.logger.info(
                f"Adding content to collection '{collection_name or 'default'}': '{content[:50]}...'"
            )
            success = await self.rag_manager.remember(
                text=content,
                collection_name=collection_name,
                source=kwargs.get("source", "tool_remember"),
                timestamp=kwargs.get("timestamp"),
                metadata=kwargs.get("metadata"),
            )
            msg = f"Remember action {'succeeded' if success else 'failed'} for collection '{collection_name or 'default'}'."
            self.logger.info(msg)

            return msg

        elif action == KBAction.FORGET:
            where = kwargs.get("where_filter")
            if not where or not isinstance(where, dict):
                error_msg = "'where_filter' argument (dict) is required for the 'forget' action."
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            self.logger.info(
                f"Forgetting entries from collection '{collection_name or 'default'}' with filter: {where}"
            )
            result = await self.rag_manager.forget(
                where_filter=where, collection_name=collection_name
            )
            self.logger.info(f"Forget action result: {result}")

            return result

        elif action == KBAction.LIST_COLLECTIONS:
            self.logger.info("Listing available collections")
            try:
                # Get available collections from RAGManager
                collections = list(self.rag_manager.vectorstores.keys())
                default_collection = self.rag_manager.default_collection_name

                result = {
                    "collections": collections,
                    "default_collection": default_collection,
                    "count": len(collections),
                }

                self.logger.info(f"Found {len(collections)} collections")

                return result
            except Exception as e:
                error_msg = f"Error listing collections: {str(e)}"
                self.logger.error(error_msg)

                return {
                    "collections": [],
                    "default_collection": None,
                    "error": error_msg,
                    "count": 0,
                }
        else:
            error_msg = f"Unsupported action: {action}"
            self.logger.error(error_msg)

            raise ValueError(error_msg)
