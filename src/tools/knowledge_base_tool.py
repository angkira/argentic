import time
from typing import List, Dict, Any, Optional, Type
from enum import Enum

from pydantic import BaseModel, Field
from langchain.docstore.document import Document

# Assuming RAGManager and Messager are accessible or passed during initialization
from core.rag import RAGManager
from core.messager import Messager
from core.tool_base import BaseTool  # Import BaseTool


# --- Argument Schema --- Define actions
class KBAction(str, Enum):
    REMIND = "remind"  # Retrieve information
    REMEMBER = "remember"  # Add information to the knowledge base
    FORGET = "forget"  # Remove information from the knowledge base


class KnowledgeBaseInput(BaseModel):
    action: KBAction = Field(
        description="The action to perform: remind (retrieve info), remember (add info), forget (remove info)."
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


# --- Helper Function --- (Moved from old implementation)
def format_docs_for_tool_output(docs: List[Document]) -> str:
    """Formats retrieved documents for the LLM, including metadata."""
    if not docs:
        return "No relevant information found in the knowledge base for the query."

    formatted_docs = []
    for i, doc in enumerate(docs):
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
    TOOL_ID = "knowledge_base_tool"  # Class attribute for easy access

    def __init__(self, messager: Messager, rag_manager: Optional[RAGManager] = None):
        """rag_manager is optional when instantiating in Agent for prompt-only tools."""
        super().__init__(
            tool_id=self.TOOL_ID,
            name="Knowledge Base Tool",
            description=(
                "Manages the knowledge base. "
                "Use 'remind' to search for information relevant to a query. Specify the query and optionally a collection name. "
                "Use 'remember' to add new information to the knowledge base. Provide content_to_add parameter with the text to store. "
                "Use 'forget' to remove information from the knowledge base with a where_filter."
            ),
            argument_schema=KnowledgeBaseInput,
            messager=messager,
        )
        self.rag_manager = rag_manager
        self.messager.log(f"KnowledgeBaseTool instance created.")

    def _execute(
        self,
        action: KBAction,
        query: Optional[str] = None,
        collection_name: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """Executes the requested action on the knowledge base."""
        self.messager.log(f"KB Tool executing action: {action.value}")
        if action == KBAction.REMIND:
            if not query:
                raise ValueError("'query' argument is required for the 'remind' action.")
            docs = self.rag_manager.retrieve(query=query, collection_name=collection_name)
            formatted_result = format_docs_for_tool_output(docs)
            self.messager.log(f"KB Tool 'remind' found {len(docs)} documents.")
            return formatted_result
        elif action == KBAction.REMEMBER:
            content = kwargs.get("content_to_add")
            if not content:
                raise ValueError("'content_to_add' argument is required for the 'remember' action.")
            success = self.rag_manager.remember(
                text=content,
                collection_name=collection_name,
                source=kwargs.get("source", "tool_remember"),
                timestamp=kwargs.get("timestamp"),
                metadata=kwargs.get("metadata"),
            )
            msg = f"Remember action {'succeeded' if success else 'failed'} for collection '{collection_name or 'default'}'."
            self.messager.log(msg)
            return msg
        elif action == KBAction.FORGET:
            where = kwargs.get("where_filter")
            if not where or not isinstance(where, dict):
                raise ValueError(
                    "'where_filter' argument (dict) is required for the 'forget' action."
                )
            result = self.rag_manager.forget(where_filter=where, collection_name=collection_name)
            self.messager.log(f"KB Tool 'forget' result: {result}")
            return result
        else:
            raise ValueError(f"Unsupported action: {action}")
