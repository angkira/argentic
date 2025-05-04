import time
import re
import json
import traceback
from typing import Any, List, Optional, Union

from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

from core.messager import Messager
from core.llm import LlamaServerLLM
from core.tool_manager import ToolManager
from core.logger import get_logger, LogLevel, parse_log_level
from core.protocol.task_protocol import TaskResultMessage  # Import for type hint


class Agent:
    """Manages interaction with LLM and ToolManager."""

    def __init__(
        self, llm: Any, messager: Messager, log_level: Union[str, LogLevel] = LogLevel.INFO
    ):
        self.llm = llm
        self.messager = messager

        # Set up logger
        if isinstance(log_level, str):
            self.log_level = parse_log_level(log_level)
        else:
            self.log_level = log_level

        self.logger = get_logger("agent", self.log_level)

        # ToolManager now handles MQTT communication for tools
        self.tool_manager = ToolManager(messager, log_level=self.log_level)
        self.prompt_template = self._build_prompt_template()

        self.logger.info("Agent initialized (uses ToolManager with MQTT tools)")
        self.messager.mqtt_log("Agent initialized (uses ToolManager with MQTT tools)")

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

    def set_log_level(self, level: Union[str, LogLevel]) -> None:
        """
        Set the logger level

        Args:
            level: New log level (string or LogLevel enum)
        """
        if isinstance(level, str):
            self.log_level = parse_log_level(level)
        else:
            self.log_level = level

        self.logger.setLevel(self.log_level.value)
        self.logger.info(f"Agent log level changed to {self.log_level.name}")

        # Update handlers
        for handler in self.logger.handlers:
            handler.setLevel(self.log_level.value)

        # Update tool manager log level
        self.tool_manager.set_log_level(self.log_level)

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
                self.logger.info(
                    f"Query Iteration: {current_iteration}/{max_tool_iterations}, User: {user_id or 'anonymous'}"
                )
                self.messager.mqtt_log(
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
                self.logger.debug(
                    f"LLM response (Iteration {current_iteration}): {response_text[:100]}..."
                )
                self.messager.mqtt_log(
                    f"Agent received LLM response (Iteration {current_iteration}): {response_text[:100]}...",
                    level="debug",
                )

                # Extract JSON block by stripping markdown code fences if present
                cleaned = response_text.strip()
                if "```" in cleaned:
                    code_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
                    code_blocks = re.findall(code_block_pattern, cleaned)
                    if code_blocks:
                        # Take the first JSON code block
                        cleaned = code_blocks[0].strip()
                    else:
                        # Fallback: basic replace if regex didn't match
                        cleaned = cleaned.replace("```json", "").replace("```", "").strip()

                potential_json = cleaned
                self.logger.debug(f"Potential JSON for parsing: {potential_json!r}")
                self.messager.mqtt_log(
                    f"Agent Debug: potential_json for JSON parsing: {potential_json!r}",
                    level="debug",
                )

                try:
                    # Use extracted JSON for parsing tool_calls, with fallback when fences removal yields bad JSON
                    is_tool_call_json = False
                    self.logger.debug("Attempting JSON load on potential_json...")
                    self.messager.mqtt_log(
                        "Agent Debug: Attempting JSON load on potential_json...", level="debug"
                    )
                    parsed_data = None

                    # First, try direct load
                    try:
                        parsed_data = json.loads(potential_json)
                    except json.JSONDecodeError:
                        # Fallback 1: Fix common JSON issues like missing closing brackets
                        fixed_json = None
                        if potential_json.strip().startswith(
                            "{"
                        ) and not potential_json.strip().endswith("}"):
                            # Try adding the missing closing bracket
                            try:
                                fixed_json = potential_json.strip() + "}"
                                parsed_data = json.loads(fixed_json)
                                self.logger.debug(
                                    f"Fixed JSON by adding closing bracket: {fixed_json}"
                                )
                                self.messager.mqtt_log(
                                    f"Agent Debug: Fixed JSON by adding closing bracket: {fixed_json}",
                                    level="debug",
                                )
                            except json.JSONDecodeError:
                                fixed_json = None

                        # Fallback 2: extract first JSON object in original response_text
                        if fixed_json is None:
                            fallback_match = re.search(r"\{[\s\S]*\}", response_text)
                            if fallback_match:
                                fallback_str = fallback_match.group(0)
                                self.logger.debug(f"Fallback JSON extracted: {fallback_str}")
                                self.messager.mqtt_log(
                                    f"Agent Debug: Fallback JSON extracted: {fallback_str}",
                                    level="debug",
                                )
                                try:
                                    parsed_data = json.loads(fallback_str)
                                    potential_json = fallback_str
                                except json.JSONDecodeError:
                                    parsed_data = None

                    # Process tool calls if found in JSON
                    if parsed_data and isinstance(parsed_data, dict):
                        response_data = parsed_data
                        tool_calls = response_data.get("tool_calls")
                        self.logger.debug(f"Parsed response_data: {response_data}")
                        self.messager.mqtt_log(
                            f"Agent Debug: Parsed response_data: {response_data}", level="debug"
                        )

                        # If LLM requested tool calls, execute them
                        if isinstance(tool_calls, list) and tool_calls:
                            is_tool_call_json = True
                            self.logger.info(f"LLM decided to use tools: {json.dumps(tool_calls)}")
                            self.messager.mqtt_log(
                                f"Reasoning: LLM decided to use tools: {json.dumps(tool_calls)}",
                                level="info",
                            )

                            # Execute each tool call and collect results
                            tool_results_for_history = []
                            has_error = False

                            for call in tool_calls:
                                tool_id = call.get("tool_id")
                                arguments = call.get("arguments")
                                if tool_id and isinstance(arguments, dict):
                                    self.logger.info(
                                        f"Executing tool '{tool_id}' with args: {arguments}"
                                    )
                                    self.messager.mqtt_log(
                                        f"Agent: Executing tool '{tool_id}' with args: {arguments}"
                                    )
                                    try:
                                        tool_result = self.tool_manager.execute_tool(
                                            tool_id, arguments
                                        )
                                        self.logger.info(
                                            f"Received result from tool '{tool_id}': {str(tool_result)[:100]}..."
                                        )
                                        self.messager.mqtt_log(
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
                                    except Exception as e:
                                        has_error = True
                                        error_msg = f"Error executing tool '{tool_id}': {e}"
                                        self.logger.error(error_msg)
                                        self.messager.mqtt_log(error_msg, level="error")
                                        # Add error as tool result so LLM knows about the failure
                                        tool_results_for_history.append(
                                            {
                                                "role": "tool",
                                                "name": tool_id,
                                                "content": f"ERROR: {error_msg}",
                                            }
                                        )
                                else:
                                    has_error = True
                                    self.logger.warning(f"Invalid tool call structure: {call}")
                                    self.messager.mqtt_log(
                                        f"Agent: Invalid tool call structure: {call}",
                                        level="warning",
                                    )

                            # Add tool results to the message history
                            if tool_results_for_history:
                                messages.extend(tool_results_for_history)
                                self.logger.info(
                                    f"Appended {len(tool_results_for_history)} tool result(s) to history"
                                )
                                self.messager.mqtt_log(
                                    f"Agent: Appended {len(tool_results_for_history)} tool result(s) to history."
                                )

                                # If we had errors, add a message to help guide the LLM
                                if has_error:
                                    messages.append(
                                        {
                                            "role": "user",
                                            "content": "Some tools encountered errors. Please provide the best answer you can based on any successful results and your own knowledge.",
                                        }
                                    )

                                # Continue the loop to let the LLM process the tool results
                                continue

                    # If we reach here, either no tool calls were found or they weren't processed
                    # Check if the response is a tool call JSON (without valid tool info)
                    # This handles the case where the format is correct but tool details are wrong
                    if (
                        parsed_data
                        and isinstance(parsed_data, dict)
                        and "tool_calls" in parsed_data
                    ):
                        self.logger.warning(
                            "Found tool_calls but couldn't process them. Will try another iteration."
                        )
                        self.messager.mqtt_log(
                            "Agent: Found tool_calls but couldn't process them. Will try another iteration.",
                            level="warning",
                        )
                        # Add a clarification message for the LLM
                        messages.append(
                            {
                                "role": "user",
                                "content": "I couldn't process the tool calls. Please provide a complete answer to the original question without using tools.",
                            }
                        )
                        continue

                    # Not a tool call JSON or already processed, treat as final answer
                    self.logger.info("LLM response is not a tool call. Treating as final answer")
                    self.messager.mqtt_log(
                        "Agent: LLM response is not a tool call. Treating as final answer."
                    )
                    final_answer = response_text
                    break  # Exit the loop, we have the answer

                except json.JSONDecodeError:
                    self.logger.info("LLM response is not valid JSON. Treating as final answer")
                    self.messager.mqtt_log(
                        "Agent: LLM response is not valid JSON. Treating as final answer."
                    )
                    final_answer = response_text
                    break  # Exit the loop
                except Exception as e:
                    self.logger.error(f"Error parsing LLM response or executing tool: {e}")
                    self.logger.debug(traceback.format_exc())
                    self.messager.mqtt_log(
                        f"Agent: Error parsing LLM response or executing tool: {e}", level="error"
                    )
                    final_answer = f"Error processing response: {e}"
                    break  # Exit the loop on error

            # --- Loop finished (either by break or max iterations) ---

            if final_answer is not None:
                # Cleanup the final answer, ensuring it's not a raw tool call JSON
                # Check if what we have is still a tool call message
                try:
                    potential_json = final_answer.strip()
                    if "```" in potential_json:
                        # Handle ```json blocks
                        code_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
                        code_blocks = re.findall(code_block_pattern, potential_json)
                        if code_blocks:
                            potential_json = code_blocks[0].strip()
                        else:
                            potential_json = (
                                potential_json.replace("```json", "").replace("```", "").strip()
                            )

                    parsed = json.loads(potential_json)
                    if isinstance(parsed, dict) and "tool_calls" in parsed:
                        # It's still a tool call JSON, this shouldn't be our final answer
                        self.logger.warning(
                            "Final answer is still a tool call JSON. Getting a proper answer."
                        )
                        self.messager.mqtt_log(
                            "Agent: Final answer is still a tool call JSON. Getting a proper answer.",
                            level="warning",
                        )
                        # Add one more message asking for a proper answer
                        messages.append(
                            {
                                "role": "user",
                                "content": "Please provide a complete answer to the original question based on your knowledge and any tool results, not another tool call.",
                            }
                        )

                        # Get one more response
                        if hasattr(self.llm, "chat"):
                            final_answer = self.llm.chat(messages)
                        else:
                            combined_prompt = "\n".join(
                                [f"{m['role'].upper()}: {m['content']}" for m in messages]
                            )
                            final_answer = self.llm(combined_prompt)
                except (json.JSONDecodeError, Exception):
                    # Not JSON or error parsing, which is fine
                    pass

            # Final safety check - NEVER return a tool call JSON to the user
            final_result = final_answer
            try:
                # Check if the final answer still looks like a tool call
                if isinstance(final_result, str):
                    # Try to identify tool call JSON in the final result
                    if "tool_calls" in final_result and (
                        "arguments" in final_result or "tool_id" in final_result
                    ):
                        is_still_tool_call = False

                        # Try to parse as JSON first
                        try:
                            # Clean up any markdown formatting
                            cleaned = final_result.strip()
                            if "```" in cleaned:
                                code_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
                                code_blocks = re.findall(code_block_pattern, cleaned)
                                if code_blocks:
                                    cleaned = code_blocks[0].strip()
                                else:
                                    cleaned = (
                                        cleaned.replace("```json", "").replace("```", "").strip()
                                    )

                            parsed = json.loads(cleaned)
                            if isinstance(parsed, dict) and "tool_calls" in parsed:
                                is_still_tool_call = True
                                self.logger.error(
                                    "CRITICAL: Final answer is still a tool call JSON in final safety check!"
                                )
                                self.messager.mqtt_log(
                                    "CRITICAL: Final answer is still a tool call JSON in final safety check!",
                                    level="error",
                                )
                        except Exception:
                            # If we can't parse as JSON but has the keywords, be cautious
                            if (
                                final_result.strip().startswith("{")
                                and final_result.strip().endswith("}")
                                and "tool_calls" in final_result
                            ):
                                is_still_tool_call = True
                                self.logger.error(
                                    "CRITICAL: Final answer appears to be a tool call JSON (not parseable but has format)!"
                                )
                                self.messager.mqtt_log(
                                    "CRITICAL: Final answer appears to be a tool call JSON (not parseable but has format)!",
                                    level="error",
                                )

                        # If it still looks like a tool call, return a safe fallback instead
                        if is_still_tool_call:
                            final_result = "I'm sorry, but I couldn't retrieve the information needed to answer your question properly. Please try asking in a different way or ask another question."
            except Exception as e:
                self.logger.error(f"Error in final safety check: {e}")
                self.messager.mqtt_log(f"Error in final safety check: {e}", level="error")
                # If anything goes wrong in the safety check, don't change the answer
                pass

            return final_result

        except Exception as e:
            err_msg = f"Unhandled error processing query for user '{user_id or 'N/A'}': {e}"
            self.logger.error(err_msg)
            self.logger.error(traceback.format_exc())  # Log full traceback
            self.messager.mqtt_log(err_msg, level="error")
            return f"Sorry, an error occurred while processing your question: {e}"
