import asyncio
import re
import json
import traceback
from typing import Any, List, Optional, Union, Tuple, Dict

from langchain.prompts import PromptTemplate

# Assuming Messager and ToolManager are updated for asyncio
from core.messager.messager import Messager
from core.protocol.message import AnswerMessage
from core.tools.tool_manager import ToolManager
from core.logger import get_logger, LogLevel, parse_log_level
from core.llm.providers.base import ModelProvider  # Import ModelProvider
from core.protocol.tool import ToolCallRequest  # Import the new request type
from core.protocol.task import (
    TaskResultMessage,
    TaskErrorMessage,
    TaskStatus,
)  # Import for result handling


class Agent:
    """Manages interaction with LLM and ToolManager (Async Version)."""

    def __init__(
        self,
        llm: ModelProvider,  # Changed type hint from Any to ModelProvider
        messager: Messager,
        log_level: Union[str, LogLevel] = LogLevel.INFO,
        register_topic: str = "agent/tools/register",
        answer_topic: str = "agent/response/answer",  # Added for clarity
    ):
        self.llm = llm
        self.messager = messager
        self.answer_topic = answer_topic  # Store answer topic

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
        # System prompt that defines the response format and rules
        system_prompt = """You are a highly capable AI assistant that MUST follow these strict response format rules:

RESPONSE FORMATS:
1. Tool Call Format (use when you need to use a tool):
```json
{{
    "type": "tool_call",
    "tool_calls": [
        {{
            "tool_id": "<exact_tool_id_from_list>",
            "arguments": {{
                "<param1>": "<value1>",
                "<param2>": "<value2>"
            }}
        }}
    ]
}}
```

2. Direct Answer Format (use when you can answer directly without tools):
```json
{{
    "type": "direct",
    "content": "<your_answer_here>"
}}
```

3. Tool Result Format (use ONLY after receiving results from a tool call to provide the final answer):
```json
{{
    "type": "tool_result",
    "tool_id": "<tool_id_of_the_executed_tool>",
    "result": "<final_answer_incorporating_tool_results_if_relevant>"
}}
```

WHEN TO USE EACH FORMAT:
1. Use "tool_call" when:
   - You need external information or actions via a tool to answer the question.
2. Use "direct" when:
   - You can answer the question directly using your general knowledge without needing tools.
   - You need to explain a tool execution error.
3. Use "tool_result" ONLY when:
   - You have just received results from a tool call (role: tool messages in history).
   - You are providing the final answer to the original question.
   - Incorporate the tool results into your answer *if they are relevant and\
       helpful*. If the tool results are not helpful or empty, state that \
        briefly and answer using your general knowledge.

STRICT RULES:
1. ALWAYS wrap your response in a markdown code block (```json ... ```).
2. ALWAYS use one of the three formats above.
3. NEVER use any other "type" value.
4. NEVER include text outside the JSON structure.
5. NEVER use markdown formatting inside the content/result fields.
6. ALWAYS use the exact tool_id from the available tools list for "tool_call".
7. ALWAYS provide complete, well-formatted JSON.
8. ALWAYS keep responses concise but complete.

HANDLING TOOL RESULTS:
- If a tool call fails (you receive an error message in the tool role), respond with a "direct" answer explaining the error.
- If you receive successful tool results (role: tool):
    - Analyze the results.
    - If the results help answer the original question, incorporate them into your final answer and use the "tool_result" format.
    - If the results are empty or not relevant to the original question, \
        briefly state that the tool didn't provide useful information, then \
            answer the original question using your general knowledge, still \
                using the "tool_result" format but explaining the situation in \
                    the 'result' field.
- If you're unsure after getting tool results, use the "tool_result" format and \
    explain your reasoning in the 'result' field.
- Never make another tool call immediately after receiving tool results unless \
    absolutely necessary and clearly justified."""

        # Main prompt that includes the system prompt and current context
        template = f"""{system_prompt}

Available Tools:
{{tool_descriptions}}

QUESTION: {{question}}

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

    async def _clean_and_extract_json_str(self, text: str) -> Optional[str]:
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
                await self.messager.log(
                    f"Agent Debug: Extracted potential JSON using regex fallback: {cleaned}",
                    level="debug",
                )

        return cleaned

    def _is_valid_tool_call_structure(self, parsed_data: Any) -> bool:
        """Checks if the parsed data has the correct structure for tool calls."""
        return (
            isinstance(parsed_data, dict)
            and "tool_calls" in parsed_data
            and isinstance(parsed_data.get("tool_calls"), list)
        )

    async def _parse_json_tool_call(self, json_str: str) -> Optional[Dict[str, Any]]:
        """Attempts to parse a string as JSON, specifically looking for tool calls."""
        try:
            # Extract JSON from markdown code block
            if "```json" in json_str:
                json_str = json_str.split("```json")[1]
            if "```" in json_str:
                json_str = json_str.split("```")[0]
            json_str = json_str.strip()

            parsed_data = json.loads(json_str)

            # Check if it's a direct response
            if isinstance(parsed_data, dict) and parsed_data.get("type") == "direct":
                return None

            # Check if it's a valid tool call
            if isinstance(parsed_data, dict) and parsed_data.get("type") == "tool_call":
                tool_calls = parsed_data.get("tool_calls", [])
                if tool_calls and all("tool_id" in call for call in tool_calls):
                    return parsed_data

            self.logger.warning(f"Invalid response format or type: {parsed_data}")
            return None

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error parsing response: {e}")
            return None

    async def _call_llm(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Calls the appropriate LLM method using the ModelProvider interface.
        ModelProvider methods (achat, chat) are expected to return a string.
        """
        # Prefer async chat method if available
        if hasattr(self.llm, "achat"):
            self.logger.debug(f"Using async chat method from provider: {type(self.llm).__name__}")
            result = await self.llm.achat(messages, **kwargs)
            return result
        elif hasattr(self.llm, "chat"):
            self.logger.debug(
                f"Using sync chat method in executor from provider: {type(self.llm).__name__}"
            )
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, self.llm.chat, messages, **kwargs)
            return result
        elif hasattr(self.llm, "ainvoke"):
            self.logger.warning(
                f"Provider {type(self.llm).__name__} does not have 'achat'. "
                "Falling back to 'ainvoke'. Chat history might not be optimally handled."
            )
            prompt = self.llm._format_chat_messages_to_prompt(messages)
            result = await self.llm.ainvoke(prompt, **kwargs)
            return result
        elif hasattr(self.llm, "invoke"):
            self.logger.warning(
                f"Provider {type(self.llm).__name__} does not have 'chat' methods. "
                "Falling back to 'invoke' in executor. Chat history might not be optimally handled."
            )
            loop = asyncio.get_running_loop()
            prompt = self.llm._format_chat_messages_to_prompt(messages)
            result = await loop.run_in_executor(None, self.llm.invoke, prompt, **kwargs)
            return result
        else:
            self.logger.error(
                f"LLM provider {type(self.llm).__name__} has no recognized "
                "callable method (achat, chat, ainvoke, invoke)."
            )
            raise TypeError(
                f"LLM provider {type(self.llm).__name__} has no recognized callable method."
            )

    async def _execute_tool_calls(
        self, tool_calls_dicts: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, str]], bool]:
        """
        Executes tool calls parsed from LLM output.
        tool_calls_dicts: List of dictionaries, each representing a tool call,
                          e.g., {'tool_id': 'some_tool', 'arguments': {...}}
        Returns a list of history-formatted messages and a boolean indicating errors.
        """
        if not tool_calls_dicts:
            return [], False

        # Convert dicts to ToolCallRequest objects
        tool_call_requests: List[ToolCallRequest] = []
        for call_dict in tool_calls_dicts:
            try:
                # Ensure 'tool_id' is present, 'arguments' defaults to {} if missing by Pydantic model
                if "tool_id" not in call_dict:
                    self.logger.error(f"Tool call dictionary missing 'tool_id': {call_dict}")
                    continue  # Skip malformed call
                tool_call_requests.append(ToolCallRequest(**call_dict))
            except Exception as e:  # Catch Pydantic validation errors or others
                self.logger.error(
                    f"Failed to parse tool call dict into ToolCallRequest: {call_dict}, Error: {e}"
                )
                continue

        if not tool_call_requests:  # If all calls were malformed
            self.logger.warning("No valid tool call requests to execute after parsing.")
            return [], True  # Indicate error as no valid calls could be processed

        # execution_outcomes is List[Union[TaskResultMessage, TaskErrorMessage]]
        execution_outcomes, any_errors_from_manager = await self._tool_manager.get_tool_results(
            tool_call_requests
        )

        history_messages: List[Dict[str, str]] = []
        final_any_errors = any_errors_from_manager  # Start with manager's error flag

        for (
            outcome
        ) in (
            execution_outcomes
        ):  # execution_outcomes is List[Union[TaskResultMessage, TaskErrorMessage]]
            content_str = ""
            tool_name = (
                outcome.tool_name
            )  # This is correct, as both TaskResultMessage and TaskErrorMessage have tool_name
            history_tool_call_id = (
                outcome.task_id if outcome.task_id else outcome.tool_id
            )  # Also correct

            if isinstance(outcome, TaskResultMessage):
                if outcome.status in [TaskStatus.COMPLETED, TaskStatus.SUCCESS]:
                    # ... (logic for successful result)
                    pass
                else:  # FAILED, TIMEOUT, etc. status on TaskResultMessage
                    content_str = f"Error: Tool '{outcome.tool_name}' failed with status {outcome.status}. Detail: {outcome.error or 'No additional error detail.'}"
                    final_any_errors = True
            elif isinstance(outcome, TaskErrorMessage):
                content_str = f"Error: Tool '{outcome.tool_name}' reported error: {outcome.error}"
                final_any_errors = True

            history_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": history_tool_call_id,  # This should align with what the LLM expects
                    "name": tool_name,  # This is the tool's actual name
                    "content": content_str,
                }
            )
        return history_messages, final_any_errors

    async def query(
        self, question: str, user_id: Optional[str] = None, max_iterations: Optional[int] = None
    ) -> str:
        """
        Processes a question through the LLM and tool interaction loop.
        """
        if max_iterations is None:
            max_iterations = self.max_tool_iterations

        history: List[Dict[str, str]] = []
        original_question = question  # Store the original question
        current_question = question

        for i in range(max_iterations):
            self.logger.info(
                f"Query Iteration: {i+1}/{max_iterations} for user '{user_id or 'Unknown'}'... Current prompt/question: {current_question[:100]}..."
            )

            tools_description_str = self._tool_manager.get_tools_description()
            tool_names_list = self._tool_manager.get_tool_names()

            # Build messages for LLM, including system prompt on first turn
            messages_for_llm: List[Dict[str, str]] = []
            system_prompt_content = (
                self._build_prompt_template().template.split("Available Tools:")[0].strip()
            )
            messages_for_llm.append({"role": "system", "content": system_prompt_content})

            if not history:
                # First turn: Add the initial user question formatted with tools
                user_prompt_content = (
                    self.prompt_template.format(
                        tool_descriptions=tools_description_str, question=current_question
                    ).split("ANSWER:")[0]
                    + "ANSWER:"
                )
                messages_for_llm.append({"role": "user", "content": user_prompt_content})
            else:
                # Subsequent turns: Add initial question, history, and current follow-up
                initial_user_prompt_content = (
                    self.prompt_template.format(
                        tool_descriptions=tools_description_str,
                        question=original_question,  # Always refer to the original question here
                    ).split("ANSWER:")[0]
                    + "ANSWER:"
                )
                messages_for_llm.append({"role": "user", "content": initial_user_prompt_content})
                messages_for_llm.extend(history)  # Add previous assistant/tool turns
                # Add the follow-up instruction/question if it exists
                if current_question != original_question:
                    messages_for_llm.append({"role": "user", "content": current_question})

            llm_response_text = await self._call_llm(messages_for_llm)
            self.logger.debug(f"LLM response (Iter {i+1}): {llm_response_text[:300]}...")

            # Append assistant response to history *before* parsing
            history.append({"role": "assistant", "content": llm_response_text})

            try:
                # Extract JSON from markdown code block
                json_str = None
                if "```json" in llm_response_text:
                    json_str = llm_response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in llm_response_text:
                    # Handle cases where ``` is used without json marker
                    json_str = llm_response_text.split("```")[1].split("```")[0].strip()
                else:
                    # Assume raw JSON if no markdown fences
                    json_str = llm_response_text.strip()

                if not json_str:
                    self.logger.error(
                        f"Could not extract JSON from LLM response: {llm_response_text}"
                    )
                    return "Error: Could not parse JSON response from AI assistant."

                parsed_response = json.loads(json_str)
                response_type = parsed_response.get("type")

                # Handle direct response
                if response_type == "direct":
                    return parsed_response.get(
                        "content", "Error: Missing content in direct response."
                    )

                # Handle tool result response (treat as final answer)
                elif response_type == "tool_result":
                    return parsed_response.get(
                        "result", "Error: Missing result in tool_result response."
                    )

                # Handle tool call
                elif response_type == "tool_call":
                    tool_calls = parsed_response.get("tool_calls", [])
                    if tool_calls:
                        tool_results_history, had_error = await self._execute_tool_calls(tool_calls)

                        if tool_results_history:
                            for res in tool_results_history:
                                history.append(res)  # Append tool results to history

                        if had_error:
                            self.logger.warning(
                                "Tool execution had errors. Asking LLM to summarize."
                            )
                            current_question = f"There were errors during tool execution (see tool messages above). Please explain the error to the user based on the original question: '{original_question}'. Use the 'direct' format."
                        else:
                            # No errors, ask LLM to process results and answer original question
                            self.logger.info(
                                "Tool execution successful. Asking LLM to process results."
                            )
                            current_question = f"The tool execution finished (see results above). Please analyze the results. If they are helpful, incorporate them into your answer to the original question: '{original_question}'. If the tool results were empty or not helpful, state that briefly and answer using your general knowledge. Respond using the 'tool_result' format."
                        continue  # Continue loop for LLM to process results/errors
                    else:
                        self.logger.warning("'tool_call' type received but no tool_calls found.")
                        return "Error: Received tool call request with no specific tools listed."
                else:
                    self.logger.error(f"Invalid response type: {response_type}")
                    return f"Error: Invalid response type '{response_type}' from AI assistant. Expected 'direct', 'tool_call', or 'tool_result'."

            except json.JSONDecodeError as e:
                self.logger.error(
                    f"Invalid JSON response from LLM: {e}. Response: {json_str}"
                )  # Log the extracted string
                return f"Error: Invalid JSON response format from AI assistant: {e}"

            except Exception as e:
                self.logger.error(f"Error processing LLM response: {e}", exc_info=True)
                return f"Error: Failed to process AI response: {str(e)}"

        self.logger.warning(f"Max iterations ({max_iterations}) reached.")
        # Fallback logic remains the same
        last_response = (
            history[-1]["content"] if history and history[-1]["role"] == "assistant" else ""
        )
        if last_response:
            try:
                json_str = None
                if "```json" in last_response:
                    json_str = last_response.split("```json")[1].split("```")[0].strip()
                elif "```" in last_response:
                    json_str = last_response.split("```")[1].split("```")[0].strip()
                else:
                    json_str = last_response.strip()

                if json_str:
                    parsed = json.loads(json_str)
                    if parsed.get("type") == "direct":
                        return parsed.get("content", "Max iterations reached.")
                    if parsed.get("type") == "tool_result":
                        return parsed.get("result", "Max iterations reached.")
            except Exception:
                pass
            return f"Error: Maximum interaction depth reached. Last response: {last_response}"
        else:
            return "Error: Maximum interaction depth reached without reaching a final answer."

    async def handle_ask_question(self, message):
        """
        MQTT handler for incoming questions.
        - message: the parsed AskQuestionMessage or dict
        """
        try:
            question: Optional[str] = None
            user_id: Optional[str] = None

            if hasattr(message, "question"):
                question = message.question
                user_id = getattr(message, "user_id", None)
            elif isinstance(message, dict):
                question = message.get("question")
                user_id = message.get("user_id")
            else:
                self.logger.error(f"Unknown message type for ask_question: {type(message)}")
                return

            if not question:
                self.logger.error("Received ask_question message without a question field.")
                return

            self.logger.info(f"Received question from user '{user_id or 'Unknown'}': {question}")

            answer_text = await self.query(question, user_id=user_id)

            answer_msg = AnswerMessage(
                source=self.messager.client_id,
                question=question,
                answer=answer_text,
                user_id=user_id,
            )
            await self.messager.publish(self.answer_topic, answer_msg.model_dump_json())
            self.logger.info(
                f"Published answer to {self.answer_topic} for user '{user_id or 'Unknown'}'"
            )

        except Exception as e:
            self.logger.error(f"Error handling ask_question: {e}", exc_info=True)

    async def _publish_answer(
        self, question: str, response: Any, user_id: Optional[str] = None
    ) -> None:
        """
        DEPRECATED or REPURPOSED: This method's original purpose of extracting content
        is now handled by ModelProviders. It might be removed or adapted if there's
        a different specific need for publishing answers outside handle_ask_question.
        For now, it's kept but likely unused by the main flow.
        """
        try:
            answer_text: str
            if isinstance(response, str):
                answer_text = response
            elif hasattr(response, "content") and isinstance(response.content, str):
                self.logger.warning(
                    "Received non-string response with .content, direct ModelProvider usage preferred."
                )
                answer_text = response.content
            else:
                self.logger.warning(
                    f"Publishing unexpected response type as string: {type(response)}"
                )
                answer_text = str(response)

            publisher_client_id = (
                self.messager.client_id if hasattr(self.messager, "client_id") else "agent_client"
            )

            answer = AnswerMessage(
                question=question,
                answer=answer_text,
                user_id=user_id,
                source=publisher_client_id,
            )

            await self.messager.publish(self.answer_topic, answer.model_dump_json())
            self.logger.info(f"Published answer (via _publish_answer) to {self.answer_topic}")
        except Exception as e:
            self.logger.error(f"Error in _publish_answer: {e}", exc_info=True)
