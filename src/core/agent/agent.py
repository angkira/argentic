import asyncio
import re
import json
import traceback
from typing import Any, List, Optional, Union, Tuple, Dict

from langchain.prompts import PromptTemplate

# Assuming Messager and ToolManager are updated for asyncio
from core.messager.messager import Messager
from core.tools.tool_manager import ToolManager
from core.logger import get_logger, LogLevel, parse_log_level


class Agent:
    """Manages interaction with LLM and ToolManager (Async Version)."""

    def __init__(
        self, llm: Any, messager: Messager, log_level: Union[str, LogLevel] = LogLevel.INFO,
        register_topic: str = "agent/tools/register"
    ):
        self.llm = llm
        self.messager = messager

        if isinstance(log_level, str):
            self.log_level = parse_log_level(log_level)
        else:
            self.log_level = log_level

        self.logger = get_logger("agent", self.log_level)

        # Initialize the async ToolManager (private)
        self._tool_manager = ToolManager(
            messager, log_level=self.log_level, register_topic=register_topic
        )
        self.prompt_template = self._build_prompt_template()
        self.max_tool_iterations = 10

        self.logger.info("Agent initialized (Async Version)")

    async def async_init(self):
        """Async initialization for Agent, including tool manager subscriptions."""
        # Subscribe tool manager to registration topics
        await self._tool_manager.async_init()
        self.logger.info("Agent: ToolManager initialized via async_init")

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
        self._tool_manager.set_log_level(self.log_level)

    # --- Methods updated for async logging ---

    async def _clean_and_extract_json_str(self, text: str) -> Optional[str]:  # Made async
        """Strips markdown fences and extracts the first potential JSON string."""
        cleaned = text.strip()
        if "```" in cleaned:
            code_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
            code_blocks = re.findall(code_block_pattern, cleaned)
            if code_blocks:
                cleaned = code_blocks[0].strip()
            else:
                cleaned = cleaned.replace("```json", "").replace("```", "").strip()

        if not (cleaned.startswith("{") and cleaned.endswith("}")):
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                cleaned = match.group(0)
                self.logger.debug(f"Extracted potential JSON using regex fallback: {cleaned}")
                # Use await for messager.log
                await self.messager.log(
                    f"Agent Debug: Extracted potential JSON using regex fallback: {cleaned}",
                    level="debug",
                )

        return cleaned

    async def _parse_json_tool_call(self, json_str: str) -> Optional[Dict[str, Any]]:  # Made async
        """Attempts to parse a string as JSON, specifically looking for tool calls."""
        try:
            parsed_data = json.loads(json_str)
            if isinstance(parsed_data, dict) and "tool_calls" in parsed_data:
                if isinstance(parsed_data["tool_calls"], list):
                    return parsed_data
                else:
                    self.logger.warning(
                        f"Parsed JSON has 'tool_calls' but it's not a list: {parsed_data}"
                    )
                    # Use await for messager.log
                    await self.messager.log(
                        f"Agent Warning: Parsed JSON has 'tool_calls' but it's not a list: {parsed_data}",
                        level="warning",
                    )
                    return None
            return None
        except json.JSONDecodeError:
            fixed_json_str = None
            stripped_json = json_str.strip()

            if '"tool_calls": [' in stripped_json and not stripped_json.endswith("]}"):
                if stripped_json.endswith("}"):
                    try:
                        fixed_json_str = stripped_json + "]}"
                        parsed_data = json.loads(fixed_json_str)
                        if (
                            isinstance(parsed_data, dict)
                            and "tool_calls" in parsed_data
                            and isinstance(parsed_data["tool_calls"], list)
                        ):
                            self.logger.debug(
                                f"Fixed and parsed JSON by adding closing '}}]': {fixed_json_str}"
                            )
                            # Use await for messager.log
                            await self.messager.log(
                                f"Agent Debug: Fixed and parsed JSON by adding closing '}}]': {fixed_json_str}",
                                level="debug",
                            )
                            return parsed_data
                    except json.JSONDecodeError:
                        fixed_json_str = None

            if (
                fixed_json_str is None
                and stripped_json.startswith("{")
                and not stripped_json.endswith("}")
            ):
                try:
                    fixed_json_str = stripped_json + "}"
                    parsed_data = json.loads(fixed_json_str)
                    if (
                        isinstance(parsed_data, dict)
                        and "tool_calls" in parsed_data
                        and isinstance(parsed_data["tool_calls"], list)
                    ):
                        self.logger.debug(
                            f"Fixed and parsed JSON by adding closing bracket: {fixed_json_str}"
                        )
                        # Use await for messager.log
                        await self.messager.log(
                            f"Agent Debug: Fixed and parsed JSON by adding closing bracket: {fixed_json_str}",
                            level="debug",
                        )
                        return parsed_data
                except json.JSONDecodeError:
                    pass

            self.logger.debug(
                f"Could not parse JSON string even after attempting fixes: {json_str[:100]}..."
            )
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error parsing JSON string '{json_str[:100]}...': {e}")
            # Use await for messager.log
            await self.messager.log(
                f"Agent Error: Unexpected error parsing JSON string '{json_str[:100]}...': {e}",
                level="error",
            )
            return None

    async def _call_llm(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Calls the appropriate LLM method (Potentially Async)."""
        # This method already handles async/sync LLM calls correctly
        if hasattr(self.llm, "achat"):
            self.logger.debug("Using async LLM chat method")
            # Assuming achat returns a string or object convertible to string
            result = await self.llm.achat(messages, **kwargs)
            return str(result)
        elif hasattr(self.llm, "chat"):
            self.logger.debug("Using sync LLM chat method in executor")
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, self.llm.chat, messages, **kwargs)
            return str(result)
        elif hasattr(self.llm, "ainvoke"):
            self.logger.debug("Using async LLM invoke method")
            prompt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])
            result = await self.llm.ainvoke(prompt, **kwargs)
            return str(result)
        else:
            self.logger.debug("Using sync LLM call method in executor")
            combined_prompt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, self.llm, combined_prompt, **kwargs)
            return str(result)

    async def _execute_tool_calls(
        self, tool_calls: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, str]], bool]:
        """Executes tool calls concurrently and returns results (Async Version)."""
        # This method already uses asyncio.gather correctly
        tool_results_for_history = []
        tasks = []
        call_info = []

        for call in tool_calls:
            tool_name = call.get("tool_name") or call.get("tool_id")
            arguments = call.get("arguments")

            if tool_name and isinstance(arguments, dict):
                self.logger.info(f"Scheduling tool '{tool_name}' with args: {arguments}")
                tasks.append(
                    asyncio.create_task(self._tool_manager.execute_tool(tool_name, arguments))
                )
                call_info.append({"name": tool_name})
            else:
                self.logger.warning(f"Invalid tool call structure: {call}")
                tool_results_for_history.append(
                    {
                        "role": "tool",
                        "name": "invalid_call",
                        "content": f"ERROR: Invalid tool call format received: {call}",
                    }
                )

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            has_error = False
            for i, result in enumerate(results):
                tool_name = call_info[i]["name"]
                if isinstance(result, Exception):
                    has_error = True
                    error_msg = f"Error executing tool '{tool_name}': {result}"
                    self.logger.error(error_msg)
                    tool_results_for_history.append(
                        {"role": "tool", "name": tool_name, "content": f"ERROR: {error_msg}"}
                    )
                elif isinstance(result, dict) and result.get("status") == "timeout":
                    has_error = True
                    error_msg = result.get("error", f"Tool '{tool_name}' timed out.")
                    self.logger.warning(f"Tool '{tool_name}' timed out.")
                    tool_results_for_history.append(
                        {"role": "tool", "name": tool_name, "content": f"TIMEOUT: {error_msg}"}
                    )
                elif isinstance(result, dict) and "error" in result:
                    has_error = True
                    error_msg = result.get("error", f"Unknown error in tool '{tool_name}'.")
                    self.logger.error(f"Error result from tool '{tool_name}': {error_msg}")
                    tool_results_for_history.append(
                        {"role": "tool", "name": tool_name, "content": f"ERROR: {error_msg}"}
                    )
                else:
                    result_str = json.dumps(result)
                    self.logger.info(
                        f"Received result from tool '{tool_name}': {result_str[:100]}..."
                    )
                    tool_results_for_history.append(
                        {"role": "tool", "name": tool_name, "content": result_str}
                    )
        else:
            has_error = bool(tool_results_for_history)

        return tool_results_for_history, has_error

    async def _is_tool_call_response(self, text: str) -> bool:  # Made async
        """Checks if the text is likely a tool call JSON, even if not perfectly parsable."""
        # Use await for calls to async methods
        json_str = await self._clean_and_extract_json_str(text)
        if json_str:
            parsed = await self._parse_json_tool_call(json_str)
            if parsed is not None:
                return True

            if "tool_calls" in json_str and ("arguments" in json_str or "tool_id" in json_str):
                if json_str.strip().startswith("{") and json_str.strip().endswith("}"):
                    self.logger.warning("Response looks like a tool call JSON but failed parsing.")
                    # Use await for messager.log
                    await self.messager.log(
                        "Agent Warning: Response looks like a tool call JSON but failed parsing.",
                        level="warning",
                    )
                    return True
        return False

    async def query(
        self, question: str, collection_name: Optional[str] = None, user_id: Optional[str] = None
    ) -> str:
        """Processes a user query, potentially using tools via async execution."""
        current_iteration = 0
        messages = []
        final_answer = None

        try:
            tool_descriptions = self._tool_manager.generate_tool_descriptions_for_prompt()
            initial_prompt_text = self.raw_template.format(
                tool_descriptions=tool_descriptions, question=question
            )
            messages = [{"role": "user", "content": initial_prompt_text}]

            while current_iteration < self.max_tool_iterations:
                current_iteration += 1
                self.logger.info(
                    f"Query Iteration: {current_iteration}/{self.max_tool_iterations}..."
                )

                grammar = None
                llm_kwargs = {}
                if grammar:
                    llm_kwargs["grammar"] = grammar

                response_text = await self._call_llm(messages, **llm_kwargs)
                messages.append({"role": "assistant", "content": response_text})
                self.logger.debug(
                    f"LLM response (Iter {current_iteration}): {response_text[:100]}..."
                )

                # Use await for calls to async methods
                potential_json_str = await self._clean_and_extract_json_str(response_text)
                parsed_tool_call = None
                if potential_json_str:
                    parsed_tool_call = await self._parse_json_tool_call(potential_json_str)

                if parsed_tool_call:
                    tool_calls = parsed_tool_call.get("tool_calls")
                    if tool_calls:
                        self.logger.info(f"LLM decided to use tools: {json.dumps(tool_calls)}")
                        tool_results, has_error = await self._execute_tool_calls(tool_calls)

                        if tool_results:
                            messages.extend(tool_results)
                            self.logger.info(
                                f"Appended {len(tool_results)} tool result(s) to history"
                            )

                        if has_error:
                            messages.append(
                                {
                                    "role": "user",
                                    "content": "Some tools encountered errors, timed out, or were invalid. Please provide the best answer you can based on any successful results and your own knowledge.",
                                }
                            )
                        continue
                    else:
                        self.logger.warning(
                            "Parsed JSON had 'tool_calls' key but content was invalid/empty."
                        )
                        messages.append(
                            {
                                "role": "user",
                                "content": "The tool call format was unclear. Please provide a complete answer to the original question without using tools, or try formatting the tool call correctly.",
                            }
                        )
                        continue
                else:
                    self.logger.info(
                        "LLM response is not a valid tool call JSON. Treating as final answer."
                    )
                    final_answer = response_text
                    break

            if final_answer is None and current_iteration >= self.max_tool_iterations:
                self.logger.warning(
                    f"Reached max iterations ({self.max_tool_iterations}). Asking LLM for final summary."
                )
                messages.append(
                    {
                        "role": "user",
                        "content": "Please provide the best possible answer based on our conversation so far.",
                    }
                )
                final_answer = await self._call_llm(messages)

            # Use await for call to async method
            if final_answer and await self._is_tool_call_response(final_answer):
                self.logger.warning(
                    "Final answer candidate is still a tool call. Getting a proper answer."
                )
                messages.append(
                    {
                        "role": "user",
                        "content": "Please provide a complete answer to the original question based on your knowledge and any tool results, not another tool call.",
                    }
                )
                final_answer = await self._call_llm(messages)

                # Use await for call to async method
                if await self._is_tool_call_response(final_answer):
                    self.logger.error(
                        "CRITICAL: Final answer is STILL a tool call after explicit request!"
                    )
                    # Use await for messager.log
                    await self.messager.log(
                        "Agent CRITICAL: Final answer is STILL a tool call after explicit request!",
                        level="critical",
                    )
                    return "I apologize, but I encountered difficulty in formulating a final response. Please try rephrasing your question."

            return final_answer if final_answer is not None else "I could not determine an answer."

        except Exception as e:
            err_msg = f"Unhandled error processing query for user '{user_id or 'N/A'}': {e}"
            self.logger.error(err_msg)
            self.logger.error(traceback.format_exc())
            # Use await for messager.log
            await self.messager.log(f"Agent Error: {err_msg}", level="error")
            return f"Sorry, an error occurred while processing your question: {e}"

    async def handle_ask_question(self, message):
        """
        MQTT handler for incoming questions.
        - message: the parsed AskQuestionMessage or dict
        """
        try:
            # Support both dict and Pydantic model
            if hasattr(message, "question"):
                question = message.question
                user_id = getattr(message, "user_id", None)
                source = getattr(message, "source", None)
            elif isinstance(message, dict):
                question = message.get("question")
                user_id = message.get("user_id")
                source = message.get("source")
            else:
                self.logger.error(f"Unknown message type: {type(message)}")
                return

            if not question:
                self.logger.error("Received ask_question message without a question field.")
                return

            self.logger.info(f"Received question from {user_id or source}: {question}")

            answer = await self.query(question, user_id=user_id)
            from core.protocol.message import AnswerMessage

            # Include required question field and correct source
            answer_msg = AnswerMessage(
                source=self.messager.client_id,
                question=question,
                answer=answer,
                user_id=user_id,
            )
            await self.messager.publish(self.answer_topic, answer_msg.model_dump_json())
            self.logger.info(f"Published answer to {self.answer_topic}")

        except Exception as e:
            self.logger.error(f"Error handling ask_question: {e}")
