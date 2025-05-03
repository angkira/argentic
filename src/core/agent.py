import time
import json
from typing import Any, List, Optional

from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

from core.messager import Messager
from core.rag import RAGManager
from core.llm import LlamaServerLLM
from core.tool_manager import ToolManager


class Agent:
    """Manages interaction with LLM, RAGManager, and ToolManager."""

    def __init__(self, llm: Any, rag_manager: RAGManager, messager: Messager):
        self.llm = llm
        self.rag_manager = rag_manager
        self.messager = messager
        self.tool_manager = ToolManager(messager)
        self.prompt_template = self._build_prompt_template()

        print("Agent initialized (uses ToolManager).")

    def _build_prompt_template(self) -> PromptTemplate:
        # Escape literal curly braces in the example JSON to prevent .format() issues
        template = """You are a highly capable assistant. Use the provided CONTEXT to inform your response when it is relevant. Always provide a complete answer using your own knowledge or the available tools.
Do not apologize or state that you lack information.

If you need to use one or more tools to answer the question, respond ONLY with a JSON object containing a 'tool_calls' list, like this:
{{"tool_calls": [{{"tool_id": "...", "arguments": {{...}}}}, ...]}}
Do not add any other text before or after the JSON.

Available Tools:
{tool_descriptions}

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
        self.raw_template = template
        return PromptTemplate.from_template(template)

    def _format_docs_with_metadata(self, docs: List[Document]) -> str:
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

    def query(
        self, question: str, collection_name: Optional[str] = None, user_id: Optional[str] = None
    ) -> str:
        max_tool_iterations = 3
        current_iteration = 0
        messages = []

        try:
            docs = self.rag_manager.retrieve(question, collection_name)
            context_str = self._format_docs_with_metadata(docs)

            tool_descriptions = self.tool_manager.generate_tool_descriptions_for_prompt()

            grammar = None
            if self.tool_manager.tools and isinstance(self.llm, LlamaServerLLM):
                grammar = self.tool_manager.generate_tool_grammar()

            initial_prompt_text = self.raw_template.format(
                tool_descriptions=tool_descriptions, context=context_str, question=question
            )
            messages = [{"role": "user", "content": initial_prompt_text}]

            while current_iteration < max_tool_iterations:
                current_iteration += 1
                self.messager.log(f"Agent Query Iteration: {current_iteration}")

                llm_kwargs = {}
                if grammar:
                    llm_kwargs["grammar"] = grammar

                if hasattr(self.llm, "chat"):
                    response_text = self.llm.chat(messages, **llm_kwargs)
                else:
                    combined_prompt = "\n".join(
                        [f"{m['role'].upper()}: {m['content']}" for m in messages]
                    )
                    response_text = self.llm(combined_prompt, **llm_kwargs)

                try:
                    potential_json = response_text.strip()
                    if potential_json.startswith("{") and potential_json.endswith("}"):
                        response_data = json.loads(potential_json)
                        tool_calls = response_data.get("tool_calls")

                        if isinstance(tool_calls, list) and tool_calls:
                            self.messager.log(f"LLM requested {len(tool_calls)} tool call(s).")
                            messages.append({"role": "assistant", "content": response_text})

                            tool_results = []
                            for call in tool_calls:
                                tool_id = call.get("tool_id")
                                arguments = call.get("arguments")
                                if tool_id and isinstance(arguments, dict):
                                    tool_result = self.tool_manager.execute_tool(tool_id, arguments)
                                    tool_results.append(
                                        {"tool_call_id": tool_id, "result": tool_result}
                                    )
                                else:
                                    self.messager.log(
                                        f"Invalid tool call structure in list: {call}",
                                        level="warning",
                                    )
                                    tool_results.append(
                                        {
                                            "tool_call_id": "unknown",
                                            "result": f"Error: Invalid tool call format received ({call})",
                                        }
                                    )

                            combined_results_str = json.dumps(tool_results)
                            messages.append({"role": "tool", "content": combined_results_str})
                            continue
                        else:
                            return response_text
                    else:
                        return response_text

                except json.JSONDecodeError:
                    return response_text
                except Exception as e:
                    self.messager.log(
                        f"Error parsing LLM response or executing tool: {e}", level="error"
                    )
                    return f"Error processing response: {e}"

            self.messager.log("Max tool iterations reached.", level="warning")
            return "Sorry, I couldn't complete the request after multiple tool calls."

        except Exception as e:
            err_msg = f"Error processing query in Agent for user '{user_id or 'N/A'}': {e}"
            self.messager.log(err_msg, level="error")
            print(err_msg)
            import traceback

            traceback.print_exc()
            return f"Sorry, an error occurred while processing your question: {e}"
