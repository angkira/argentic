from typing import Dict, List, Optional
import time
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

# Local imports
from core.messager import Messager
from core.rag import RAGManager  # Import RAGManager for interaction


class Agent:
    """Manages tools, prompts, and interaction with the LLM and RAGManager."""

    def __init__(self, llm: OllamaLLM, rag_manager: RAGManager, messager: Messager):
        self.llm = llm
        self.rag_manager = rag_manager
        self.messager = messager
        self.tools: Dict[str, Dict[str, str]] = {}
        self.prompt_template = self._build_prompt_template()
        print("Agent initialized.")

    def _build_prompt_template(self) -> PromptTemplate:
        # Base template - might be enhanced with tool usage instructions later
        template = """You are a highly capable assistant. Use the provided CONTEXT to inform your response when it is relevant, but regardless of context relevance you must always provide a complete, informed answer using your own knowledge.
Do not apologize or state that you lack information; simply answer the question directly.

{tool_instructions}

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
        self.raw_template = template  # Store raw template for manual formatting
        return PromptTemplate.from_template(template)

    def _format_docs_with_metadata(self, docs: List[Document]) -> str:
        """Helper function to format retrieved documents including metadata."""
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
            collection_name = doc.metadata.get("collection", "unknown")
            formatted_docs.append(
                f"--- Document {i+1} (Collection: {collection_name}, Source: {source}, Time: {ts_str}) ---\n{doc.page_content}"
            )
        return "\n\n".join(formatted_docs) if formatted_docs else "No relevant context found."

    def register_tool(self, tool_id: str, tool_name: str, tool_manual: str, tool_api: str) -> str:
        """Stores tool information."""
        if tool_id in self.tools:
            self.messager.log(
                f"Warning: Tool with ID '{tool_id}' already registered. Overwriting.",
                level="warning",
            )

        self.tools[tool_id] = {
            "name": tool_name,
            "manual": tool_manual,
            "api": tool_api,
        }

        tool_description = (
            f"Tool Name: {tool_name}\n"
            f"Tool ID: {tool_id}\n"
            f"Manual: {tool_manual}\n"
            f"API Schema: {tool_api}"
        )

        self.messager.log(f"Agent registered tool '{tool_name}' (ID: {tool_id}).")
        return tool_description

    def get_tool_prompt_segment(self) -> str:
        """Generates a string describing available tools for inclusion in an LLM prompt."""
        if not self.tools:
            return ""

        prompt_lines = ["You have access to the following tools:"]
        for tool_id, tool_data in self.tools.items():
            prompt_lines.append(f"- Tool Name: {tool_data['name']} (ID: {tool_id})")
            prompt_lines.append(f"  Description: {tool_data['manual']}")
            prompt_lines.append(f"  API Schema (use this format for requests): {tool_data['api']}")
        prompt_lines.append(
            'To use a tool, respond with a JSON object matching the tool\'s API schema within <tool_call> tags. Example: <tool_call>{"tool_id": "some_id", "parameters": {...}}</tool_call>'
        )

        return "\n".join(prompt_lines)

    def query(
        self, question: str, collection_name: Optional[str] = None, user_id: Optional[str] = None
    ) -> str:
        """Handles a user query: retrieves context, formats prompt, calls LLM."""
        if not question:
            return "Please provide a question."

        try:
            # 1. Retrieve context using RAGManager
            docs = self.rag_manager.retrieve(question, collection_name)
            context = self._format_docs_with_metadata(docs)

            # 2. Get tool instructions
            tool_instructions = self.get_tool_prompt_segment()

            # 3. Format the prompt using the raw template
            prompt = self.raw_template.format(
                tool_instructions=tool_instructions, context=context, question=question
            )

            # 4. Call the LLM
            self.messager.log(
                f"Sending prompt to LLM (query for user: {user_id or 'N/A'}):\n{prompt}"
            )
            response = self.llm(prompt)
            self.messager.log(f"Received LLM response: {response}")

            return response

        except Exception as e:
            err_msg = f"Error processing query in Agent for user '{user_id or 'N/A'}': {e}"
            self.messager.log(err_msg, level="error")
            print(err_msg)
            return f"Sorry, an error occurred while processing your question: {e}"
