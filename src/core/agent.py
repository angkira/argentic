import time
import re
import json
from typing import Any, List, Optional

from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

from core.messager import Messager
from core.rag import RAGManager
from core.llm import LlamaServerLLM
from core.tool_manager import ToolManager
from core.protocol.task_protocol import TaskResultMessage  # Import for type hint


class Agent:
    """Manages interaction with LLM, RAGManager, and ToolManager."""

    def __init__(self, llm: Any, rag_manager: RAGManager, messager: Messager):
        self.llm = llm
        # RAGManager is still needed for the tool implementation (passed to tool constructor)
        self.rag_manager = rag_manager
        self.messager = messager
        # ToolManager now handles MQTT communication for tools
        self.tool_manager = ToolManager(messager)
        self.prompt_template = self._build_prompt_template()

        print("Agent initialized (uses ToolManager with MQTT tools).")

    def _build_prompt_template(self) -> PromptTemplate:
        # Updated prompt to explain how tool results are provided
        template = """You are a highly capable assistant. Use the available tools to gather information needed to answer the question. Always provide a complete answer using your own knowledge or the available tools.
Do not apologize or state that you lack information.

If you need to use tools to answer the question, respond ONLY with a JSON object containing a 'tool_calls' list. Each item must include 'tool_id' and 'arguments'. Example:
{{"tool_calls": [{{"tool_id": "knowledge_base_tool", "arguments": {{"action": "remind", "query": "user's question here"}}}}]}}
Do not add any other text before or after the JSON.

After requesting a tool call, you will receive the tool's output as subsequent assistant messages (with the tool name). Use these outputs to formulate your final answer. If needed, you may invoke tools multiple times or answer directly based on your knowledge.

Available Tools:
{tool_descriptions}

QUESTION: {question}

ANSWER:"""
        self.raw_template = template
        return PromptTemplate.from_template(template)

    def query(
        self, question: str, collection_name: Optional[str] = None, user_id: Optional[str] = None
    ) -> str:
        max_tool_iterations = 3
        current_iteration = 0
        messages = []  # History for the LLM chat
        final_answer = None  # Variable to store the potential final answer

        try:
            # Get tool descriptions in the new format
            tool_descriptions = self.tool_manager.generate_tool_descriptions_for_prompt()

            # Grammar generation might need adjustment for the new tool structure
            grammar = None

            initial_prompt_text = self.raw_template.format(
                tool_descriptions=tool_descriptions, question=question
            )
            messages = [{"role": "user", "content": initial_prompt_text}]

            while current_iteration < max_tool_iterations:
                current_iteration += 1
                self.messager.log(
                    f"Agent Query Iteration: {current_iteration}/{max_tool_iterations}, User: {user_id}"
                )

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

                # Record the raw assistant response
                messages.append({"role": "assistant", "content": response_text})
                self.messager.log(
                    f"Agent received LLM response (Iteration {current_iteration}): {response_text[:100]}...",
                    level="debug",
                )
                # Extract JSON block by stripping markdown code fences if present
                cleaned = response_text.strip()
                if cleaned.startswith("```"):
                    lines = cleaned.splitlines()
                    # Drop the opening fence line
                    lines = lines[1:]
                    # Drop the closing fence line if present
                    if lines and lines[-1].startswith("```"):
                        lines = lines[:-1]
                    cleaned = "\n".join(lines).strip()
                self.messager.log(f"Agent Debug: cleaned JSON content: {cleaned!r}", level="debug")
                potential_json = cleaned
                self.messager.log(
                    f"Agent Debug: potential_json for JSON parsing: {potential_json!r}",
                    level="debug",
                )

                try:
                    # Use extracted JSON for parsing tool_calls, with fallback when fences removal yields bad JSON
                    is_tool_call_json = False
                    self.messager.log(
                        "Agent Debug: Attempting JSON load on potential_json...", level="debug"
                    )
                    parsed_data = None
                    # First, try direct load
                    try:
                        parsed_data = json.loads(potential_json)
                    except json.JSONDecodeError:
                        # Fallback: extract first JSON object in original response_text
                        fallback_match = re.search(r"\{[\s\S]*\}", response_text)
                        if fallback_match:
                            fallback_str = fallback_match.group(0)
                            self.messager.log(
                                f"Agent Debug: Fallback JSON extracted: {fallback_str}",
                                level="debug",
                            )
                            try:
                                parsed_data = json.loads(fallback_str)
                                potential_json = fallback_str
                            except json.JSONDecodeError:
                                parsed_data = None
                    if parsed_data and isinstance(parsed_data, dict):
                        response_data = parsed_data
                        tool_calls = response_data.get("tool_calls")
                        self.messager.log(
                            f"Agent Debug: Parsed response_data: {response_data}", level="debug"
                        )
                    else:
                        tool_calls = None
                        self.messager.log(
                            "Agent Debug: No valid JSON tool_calls found after fallback.",
                            level="debug",
                        )
                    self.messager.log(
                        f"Agent Debug: Extracted tool_calls: {tool_calls}",
                        level="debug",
                    )
                    # If LLM requested tool calls, execute them
                    if isinstance(tool_calls, list) and tool_calls:
                        is_tool_call_json = True
                        self.messager.log(
                            f"Reasoning: LLM decided to use tools: {json.dumps(tool_calls)}",
                            level="info",
                        )
                        tool_results_for_history = []
                        for call in tool_calls:
                            tool_id = call.get("tool_id")
                            arguments = call.get("arguments")
                            if tool_id and isinstance(arguments, dict):
                                self.messager.log(
                                    f"Agent: Executing tool '{tool_id}' with args: {arguments}"
                                )
                                tool_result = self.tool_manager.execute_tool(tool_id, arguments)
                                self.messager.log(
                                    f"Agent: Received result from tool '{tool_id}': {str(tool_result)[:100]}..."
                                )
                                # Tag the message as coming from the tool
                                tool_results_for_history.append(
                                    {
                                        "role": "tool",
                                        "name": tool_id,
                                        "content": str(tool_result),
                                    }
                                )
                            else:
                                self.messager.log(
                                    f"Agent: Invalid tool call structure: {call}",
                                    level="warning",
                                )
                        # Append tool results and continue LLM loop
                        messages.extend(tool_results_for_history)
                        self.messager.log(
                            f"Agent: Appended {len(tool_results_for_history)} tool result(s) to history."
                        )
                        continue

                    # If no tool calls were requested, break with final answer
                    if not is_tool_call_json:
                        self.messager.log(
                            "Agent: LLM response is not a tool call. Treating as final answer."
                        )
                        final_answer = response_text
                        break  # Exit the loop, we have the answer

                except json.JSONDecodeError:
                    self.messager.log(
                        "Agent: LLM response is not valid JSON. Treating as final answer."
                    )
                    final_answer = response_text
                    break  # Exit the loop
                except Exception as e:
                    self.messager.log(
                        f"Agent: Error parsing LLM response or executing tool: {e}", level="error"
                    )
                    final_answer = f"Error processing response: {e}"
                    break  # Exit the loop on error

            # --- Loop finished (either by break or max iterations) ---

            if final_answer is not None:
                self.messager.log(f"Agent: Returning final answer: {final_answer[:100]}...")
                return final_answer
            else:
                # Max iterations were reached without a final answer
                self.messager.log(
                    "Agent: Max tool iterations reached without a definitive final answer.",
                    level="warning",
                )
                # Get the very last assistant message, even if it was a tool call request
                last_assistant_message = next(
                    (m["content"] for m in reversed(messages) if m["role"] == "assistant"), None
                )
                # Check if the last message was a tool call JSON
                is_last_message_tool_call = False
                if last_assistant_message:
                    try:
                        potential_json = last_assistant_message.strip()
                        if potential_json.startswith("{") and potential_json.endswith("}"):
                            data = json.loads(potential_json)
                            if isinstance(data.get("tool_calls"), list):
                                is_last_message_tool_call = True
                    except json.JSONDecodeError:
                        pass  # Not JSON

                if is_last_message_tool_call:
                    self.messager.log(
                        "Agent: Max iterations reached, and the last message was still a tool call request. Returning error.",
                        level="error",
                    )
                    return (
                        "Sorry, I got stuck trying to use tools and couldn't find a final answer."
                    )
                elif last_assistant_message:
                    self.messager.log(
                        f"Agent: Max iterations reached. Returning last assistant message as fallback: {last_assistant_message[:100]}...",
                        level="warning",
                    )
                    return last_assistant_message  # Return the last thing the assistant said as a fallback
                else:
                    self.messager.log(
                        "Agent: Max iterations reached, but no assistant messages found. Returning generic error.",
                        level="error",
                    )
                    return "Sorry, I couldn't complete the request after multiple attempts."

        except Exception as e:
            err_msg = f"Agent: Unhandled error processing query for user '{user_id or 'N/A'}': {e}"
            self.messager.log(err_msg, level="error")
            print(err_msg)  # Also print to console
            import traceback

            traceback.print_exc()  # Print traceback to console
            return f"Sorry, an error occurred while processing your question: {e}"
