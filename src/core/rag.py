from typing import List, Dict, Any, Optional
import time
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
import chromadb

# Import Messager for type hinting
from core.messager import Messager

DEFAULT_COLLECTION_NAME = "default_rag_collection"


class RAGManager:
    """Manages multiple RAG collections (vectorstores) and performs CRUD operations."""

    def __init__(
        self,
        db_client: chromadb.Client,
        retriever_k: int,
        messager: Messager,
        default_collection_name: str = DEFAULT_COLLECTION_NAME,
    ):
        self.db_client = db_client
        self.retriever_k = retriever_k
        self.messager = messager
        self.default_collection_name = default_collection_name

        self.vectorstores: Dict[str, Chroma] = {}
        self.retrievers: Dict[str, Any] = {}

        self.get_or_create_collection(self.default_collection_name)

        print(f"RAGManager initialized. Default collection: '{self.default_collection_name}'")

    def get_or_create_collection(self, collection_name: str) -> Chroma:
        """Gets an existing Chroma vectorstore for a collection or creates it."""
        if collection_name not in self.vectorstores:
            self.messager.log(f"Initializing vectorstore for collection: '{collection_name}'")
            vectorstore = Chroma(
                client=self.db_client,
                collection_name=collection_name,
            )
            self.vectorstores[collection_name] = vectorstore
            self.retrievers[collection_name] = vectorstore.as_retriever(
                search_kwargs={"k": self.retriever_k}
            )
            self.messager.log(
                f"Vectorstore and retriever created for collection: '{collection_name}'"
            )
        return self.vectorstores[collection_name]

    def remember(
        self,
        text: str,
        collection_name: Optional[str] = None,
        source: str = "manual_input",
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Adds text to a specific RAG collection."""
        target_collection = collection_name or self.default_collection_name
        vectorstore = self.get_or_create_collection(target_collection)

        if not text:
            self.messager.log("Warning: Attempted to remember empty text.", level="warning")
            return False

        if timestamp is None:
            timestamp = time.time()
        try:
            timestamp = float(timestamp)
        except (ValueError, TypeError):
            warn_msg = f"Warning: Invalid timestamp format '{timestamp}', using current time."
            self.messager.log(warn_msg, level="warning")
            timestamp = time.time()

        doc_metadata = metadata or {}
        doc_metadata["source"] = source
        doc_metadata["timestamp"] = timestamp
        doc_metadata["collection"] = target_collection

        doc = Document(page_content=text, metadata=doc_metadata)

        try:
            vectorstore.add_documents([doc])
            log_msg = f"Remembered info in collection '{target_collection}' from '{source}': '{text[:60]}...'"
            self.messager.log(log_msg)
            return True
        except Exception as e:
            err_msg = f"Error remembering document in collection '{target_collection}': {e}"
            self.messager.log(err_msg, level="error")
            return False

    def forget(
        self, where_filter: Dict[str, Any], collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Deletes documents from a specific collection based on a metadata filter."""
        target_collection = collection_name or self.default_collection_name

        if not where_filter:
            msg = "Forget command requires a non-empty 'where_filter' for safety."
            self.messager.log(f"Warning: {msg}", level="warning")
            return {"status": "error", "message": msg, "deleted_count": 0}

        try:
            collection = self.db_client.get_collection(name=target_collection)
            results = collection.get(where=where_filter, include=[])
            ids_to_delete = results.get("ids", [])

            if not ids_to_delete:
                msg = f"No documents found in collection '{target_collection}' matching filter: {where_filter}"
                self.messager.log(msg, level="info")
                return {"status": "not_found", "message": msg, "deleted_count": 0}

            collection.delete(ids=ids_to_delete)

            msg = f"Forgot {len(ids_to_delete)} document(s) in collection '{target_collection}' matching filter: {where_filter}"
            self.messager.log(msg)
            return {
                "status": "success",
                "message": msg,
                "deleted_count": len(ids_to_delete),
            }

        except Exception as e:
            err_msg = f"Error forgetting documents in collection '{target_collection}' with filter {where_filter}: {e}"
            self.messager.log(err_msg, level="error")
            return {"status": "error", "message": str(e), "deleted_count": 0}

    def retrieve(self, query: str, collection_name: Optional[str] = None) -> List[Document]:
        """Retrieves relevant documents from a specific collection based on the query."""
        target_collection = collection_name or self.default_collection_name

        if target_collection not in self.retrievers:
            self.messager.log(
                f"Warning: Attempted to retrieve from non-existent collection '{target_collection}'. Returning empty list.",
                level="warning",
            )
            return []

        retriever = self.retrievers[target_collection]

        try:
            docs = retriever.invoke(query)
            self.messager.log(
                f"Retrieved {len(docs)} documents from collection '{target_collection}' for query: '{query[:60]}...'"
            )
            return docs
        except Exception as e:
            err_msg = f"Error retrieving documents from collection '{target_collection}' for query '{query}': {e}"
            self.messager.log(err_msg, level="error")
            return []
