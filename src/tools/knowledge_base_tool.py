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
    # FORGET = "forget" # Future: Remove information
    # REMEMBER = "remember" # Future: Add information


class KnowledgeBaseInput(BaseModel):
    action: KBAction = Field(description="The action to perform: remind (retrieve info).")
    query: str = Field(
        description="The specific question or topic to search for when action is 'remind'."
    )
    collection_name: Optional[str] = Field(
        None, description="Optional name of a specific collection to search within."
    )
    # Future fields for other actions:
    # content_to_add: Optional[str] = Field(None, description="Content to add when action is 'remember'.")
    # document_id_to_forget: Optional[str] = Field(None, description="ID of the document to forget when action is 'forget'.")


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

    def __init__(self, rag_manager: RAGManager, messager: Messager):
        super().__init__(
            tool_id=self.TOOL_ID,
            name="Knowledge Base Tool",
            description=(
                "Manages the knowledge base. Use 'remind' to search for information relevant to a query. "
                "Specify the query and optionally a collection name."
                # Future: "Use 'remember' to add new information. Use 'forget' to remove information."
            ),
            argument_schema=KnowledgeBaseInput,
            messager=messager,
        )
        self.rag_manager = rag_manager
        self.messager.log(f"KnowledgeBaseTool instance created.")

    def _execute(
        self, action: KBAction, query: str, collection_name: Optional[str] = None, **kwargs
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
        # elif action == KBAction.REMEMBER:
        #     # Placeholder for future implementation
        #     content = kwargs.get("content_to_add")
        #     if not content:
        #         raise ValueError("'content_to_add' is required for 'remember' action.")
        #     # self.rag_manager.add(content, metadata={...}, collection_name=collection_name)
        #     self.messager.log(f"KB Tool 'remember' action (Not Implemented Yet).", level="warning")
        #     return "Information scheduled for addition (Not Implemented Yet)."
        # elif action == KBAction.FORGET:
        #     # Placeholder for future implementation
        #     doc_id = kwargs.get("document_id_to_forget")
        #     if not doc_id:
        #         raise ValueError("'document_id_to_forget' is required for 'forget' action.")
        #     # self.rag_manager.delete(doc_id, collection_name=collection_name)
        #     self.messager.log(f"KB Tool 'forget' action (Not Implemented Yet).", level="warning")
        #     return "Information scheduled for deletion (Not Implemented Yet)."
        else:
            raise ValueError(f"Unsupported action: {action}")
