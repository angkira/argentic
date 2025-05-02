from typing import List, Dict, Any, Optional
import time
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.docstore.document import Document
import chromadb

# import warnings

# from langchain.schema import LangChainDeprecationWarning

# Suppress deprecated get_relevant_documents warning
# warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

# Import Messager for type hinting (assuming messager.py is in the same dir)
from core.messager import Messager


class RAGController:
    """Encapsulates RAG logic: remembering, forgetting, and querying."""

    def __init__(
        self,
        llm: OllamaLLM,
        vectorstore: Chroma,
        db_client: chromadb.Client,
        retriever_k: int,
        messager: Messager,
        use_chat: bool = False,
    ):
        self.llm = llm
        self.vectorstore = vectorstore
        self.db_client = db_client
        self.retriever_k = retriever_k
        self.messager = messager  # Store the messager instance
        self.use_chat = use_chat
        from collections import defaultdict

        # Per-user chat history when use_chat=True
        self.histories: Dict[str, list] = defaultdict(list)
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.retriever_k}
        )
        self.prompt_template = self._build_prompt_template()
        self.rag_chain = self._build_rag_chain()
        print("RAGController initialized.")

    def _build_prompt_template(self) -> PromptTemplate:
        template = """You are a highly capable assistant. Use the provided CONTEXT to inform your response when it is relevant, but regardless of context relevance you must always provide a complete, informed answer using your own knowledge.
Do not apologize or state that you lack information; simply answer the question directly.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
        # Store raw template string for manual rendering
        self.raw_template = template
        return PromptTemplate.from_template(template)

    def _format_docs_with_metadata(self, docs: List[Document]) -> str:
        """Helper function to format retrieved documents including metadata."""
        formatted_docs = []
        for i, doc in enumerate(docs):
            ts_unix = doc.metadata.get("timestamp", 0)
            ts_str = "N/A"
            if ts_unix:
                try:
                    ts_str = time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(float(ts_unix))
                    )
                except (ValueError, TypeError):
                    ts_str = f"Invalid timestamp ({ts_unix})"  # Handle potential non-float values
            source = doc.metadata.get("source", "unknown")
            formatted_docs.append(
                f"--- Document {i+1} (Source: {source}, Time: {ts_str}) ---\n{doc.page_content}"
            )
        return (
            "\n\n".join(formatted_docs)
            if formatted_docs
            else "No relevant context found."
        )

    def _build_rag_chain(self):
        return (
            {
                "context": self.retriever | self._format_docs_with_metadata,
                "question": RunnablePassthrough(),
            }
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )

    def remember(
        self, text: str, source: str = "manual_input", timestamp: Optional[float] = None
    ) -> bool:
        """Adds a piece of text information to the vector store."""
        if not text:
            print("Warning: Attempted to remember empty text.")
            # Use self.messager to publish log
            self.messager.publish_log(
                "Warning: Attempted to remember empty text.", level="warning"
            )
            return False
        if timestamp is None:
            timestamp = time.time()

        try:
            timestamp = float(timestamp)
        except (ValueError, TypeError):
            warn_msg = (
                f"Warning: Invalid timestamp format '{timestamp}', using current time."
            )
            print(warn_msg)
            self.messager.publish_log(warn_msg, level="warning")
            timestamp = time.time()

        doc = Document(
            page_content=text, metadata={"source": source, "timestamp": timestamp}
        )

        try:
            self.vectorstore.add_documents([doc])
            log_msg = f"Remembered info from '{source}': '{text[:60]}...'"
            print(log_msg)
            # Use self.messager to publish log
            self.messager.publish_log(log_msg)
            return True
        except Exception as e:
            err_msg = f"Error remembering document: {e}"
            print(err_msg)
            # Use self.messager to publish log
            self.messager.publish_log(err_msg, level="error")
            return False

    def forget(self, where_filter: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deletes documents from the vector store based on a metadata filter.

        Args:
            where_filter: A dictionary specifying metadata fields and values
                          to match for deletion (e.g., {"source": "specific_source"}).
                          See ChromaDB documentation for filter syntax.

        Returns:
            A dictionary containing the status and count of deleted documents.
        """
        if not where_filter:
            msg = "Forget command requires a non-empty 'where_filter'."
            print(f"Warning: {msg}")
            # Use self.messager to publish log
            self.messager.publish_log(f"Warning: {msg}", level="warning")
            return {"status": "error", "message": msg, "deleted_count": 0}

        try:
            # Chroma's delete method uses the collection directly
            # Ensure db_client is Client type, adjust if PersistentClient has different API
            collection = self.db_client.get_collection(
                name=self.vectorstore._collection.name,
                # embedding_function=self.vectorstore._embedding_function # May not be needed for get/delete
            )

            results = collection.get(where=where_filter, include=[])
            ids_to_delete = results.get("ids", [])

            if not ids_to_delete:
                msg = f"No documents found matching filter: {where_filter}"
                print(msg)
                # Use self.messager to publish log
                self.messager.publish_log(msg, level="info")
                return {"status": "not_found", "message": msg, "deleted_count": 0}

            collection.delete(ids=ids_to_delete)

            msg = f"Forgot {len(ids_to_delete)} document(s) matching filter: {where_filter}"
            print(msg)
            # Use self.messager to publish log
            self.messager.publish_log(msg)
            return {
                "status": "success",
                "message": msg,
                "deleted_count": len(ids_to_delete),
            }

        except Exception as e:
            err_msg = f"Error forgetting documents with filter {where_filter}: {e}"
            print(err_msg)
            # Use self.messager to publish log
            self.messager.publish_log(err_msg, level="error")
            return {"status": "error", "message": str(e), "deleted_count": 0}

    def query(self, question: str, user_id: Optional[str] = None) -> str:
        """Processes a question using the RAG chain."""
        if not question:
            return "Please provide a question."
        try:
            # Debug: show session mode and user_id
            print(
                f"DEBUG [RAGController.query] use_chat={self.use_chat}, user_id={user_id}"
            )
            # Session-based chat if enabled
            if self.use_chat:
                print("DEBUG [RAGController.query] entering chat branch")
                # Retrieve context via new API (optional)
                docs = self.retriever.invoke(question)
                context = self._format_docs_with_metadata(docs)
                # Stronger system instructions: answer directly, ignore context if irrelevant
                system_msg = (
                    "You are an expert assistant with full domain knowledge. ALWAYS answer the user's question directly, without disclaimers or apologies. "
                    "Disregard any provided context if it does not contain the answer, and do not mention the context under any circumstances."
                )
                # Build chat messages: always system message, optionally context, then user question
                messages = [{"role": "system", "content": system_msg}]
                # Only include context if it contains real content
                if docs and not context.strip().startswith("No relevant context found"):
                    messages.append(
                        {"role": "system", "content": f"CONTEXT:\n{context}"}
                    )
                # Append the user question
                messages.append({"role": "user", "content": question})
                print(f"DEBUG [RAGController.query] chat messages: {messages}")
                reply = self.llm.chat(messages)
                # If the model still refuses, fallback to a direct question-only chat
                lower = reply.lower()
                # refusal detection list expanded
                refusals = [
                    "don't have",
                    "i cannot",
                    "cannot provide",
                    "don't know",
                    "lack information",
                    "outside the scope",
                    "outside scope",
                    "request is outside",
                ]
                if any(phrase in lower for phrase in refusals):
                    print(
                        "DEBUG [RAGController.query] refusal detected, retrying without context"
                    )
                    sys_msg = (
                        "You are a highly capable assistant. Always answer directly using your knowledge; "
                        "regardless of any provided context, provide a complete informed answer."
                    )
                    retry_msgs = [
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": question},
                    ]
                    reply_retry = self.llm.chat(retry_msgs)
                    return reply_retry
                return reply
            # Single-turn RAG chain invocation (avoids deprecated get_relevant_documents)
            print("DEBUG [RAGController.query] using single-turn rag_chain.invoke")
            raw = self.rag_chain.invoke({"question": question})
            print(
                f"DEBUG [RAGController.query] raw result: {raw!r} (type: {type(raw)})"
            )
            # Normalize output to string
            if hasattr(raw, "value"):
                return raw.value
            if hasattr(raw, "content"):
                return raw.content
            if isinstance(raw, dict):
                return next(iter(raw.values()))
            return str(raw)
        except Exception as e:
            err_msg = f"Error processing query '{question}': {e}"
            print(err_msg)
            self.messager.publish_log(err_msg, level="error")
            return f"Sorry, an error occurred while processing your question: {e}"
