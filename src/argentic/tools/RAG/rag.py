from typing import Any, Dict, List, Literal, Optional, Protocol

from pydantic import BaseModel, field_validator

from argentic.core.protocol.message import BaseMessage

DEFAULT_COLLECTION_NAME = "default_rag_collection"


# Simple Document replacement for LangChain Document
class Document(BaseModel):
    page_content: str
    metadata: Dict[str, Any] = {}


# Simple Embeddings interface to replace LangChain Embeddings
class Embeddings(Protocol):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        ...

    def embed_query(self, text: str) -> List[float]:
        """Embed a query string."""
        ...


class AddInfoMessage(BaseMessage[None]):
    type: Literal["ADD_INFO"] = "ADD_INFO"
    text: str
    collection_name: Optional[str] = None
    source_info: Optional[str] = "add_info"
    metadata: Optional[Dict[str, Any]] = None


class ForgetMessage(BaseMessage[None]):
    type: Literal["FORGET_INFO"] = "FORGET_INFO"
    where_filter: Dict[str, Any]
    collection_name: Optional[str] = None

    @field_validator("where_filter")
    def check_where_filter_not_empty(cls, v):
        if not v:
            raise ValueError("'where_filter' cannot be empty for safety.")
        return v


class RetrieveInfoMessage(BaseMessage[None]):
    type: Literal["RETRIEVE_INFO"] = "RETRIEVE_INFO"
    query: str
    collection_name: Optional[str] = None
    k: Optional[int] = None


# Stub implementation - RAG functionality disabled after LangChain removal
class RAGManager:
    """Stub RAG manager that provides the same interface but with disabled functionality."""

    def __init__(self, *args, **kwargs):
        self.default_collection_name = kwargs.get(
            "default_collection_name", DEFAULT_COLLECTION_NAME
        )
        self.collections = {}
        self.vectorstores = {}  # For compatibility

    async def async_init(self):
        """Initialize the RAG manager (stub)."""
        pass

    async def get_or_create_collection(self, collection_name: str):
        """Get or create a collection (stub)."""
        return None

    async def add_info(self, message: AddInfoMessage) -> str:
        """Add information to the knowledge base (stub)."""
        return "RAG functionality disabled - LangChain removed. Cannot add information."

    async def retrieve_info(self, message: RetrieveInfoMessage) -> str:
        """Retrieve information from the knowledge base (stub)."""
        return "RAG functionality disabled - LangChain removed. Cannot retrieve information."

    async def forget_info(self, message: ForgetMessage) -> str:
        """Remove information from the knowledge base (stub)."""
        return "RAG functionality disabled - LangChain removed. Cannot remove information."
