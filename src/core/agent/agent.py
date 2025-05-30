import asyncio
import re
import json
from typing import List, Optional, Union, Tuple, Dict, Literal

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

from core.messager.messager import Messager
from core.protocol.message import (
    BaseMessage,
    AgentSystemMessage,
    AgentLLMRequestMessage,
    AgentLLMResponseMessage,
    AskQuestionMessage,
    AnswerMessage,
    MinimalToolCallRequest,
)
from core.protocol.enums import MessageSource, LLMRole
from core.protocol.tool import ToolCallRequest
from core.protocol.task import (
    TaskResultMessage,
    TaskErrorMessage,
)
from core.tools.tool_manager import ToolManager
from core.logger import get_logger, LogLevel, parse_log_level
from core.llm.providers.base import ModelProvider


# Pydantic Models for LLM JSON Response Parsing
class LLMResponseToolCall(BaseModel):
    type: Literal["tool_call"]
    tool_calls: List[ToolCallRequest]


class LLMResponseDirect(BaseModel):
    type: Literal["direct"]
    content: str


class LLMResponseToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_id: str
    result: str


# Union type for all possible LLM responses
class LLMResponse(BaseModel):
    """Union model that can handle any of the three response types"""

    type: Literal["tool_call", "direct", "tool_result"]
    # Optional fields for different response types
    tool_calls: Optional[List[ToolCallRequest]] = None
    content: Optional[str] = None
    tool_id: Optional[str] = None
    result: Optional[str] = None


class Agent:
    """Manages interaction with LLM and ToolManager (Async Version)."""

    def __init__(
        self,
        llm: ModelProvider,
        messager: Messager,
        log_level: Union[str, LogLevel] = LogLevel.INFO,
        register_topic: str = "agent/tools/register",
        answer_topic: str = "agent/response/answer",
        system_prompt: Optional[str] = None,
    ):
        self.llm = llm
        self.messager = messager
        self.answer_topic = answer_topic
        self.raw_template: Optional[str] = None
        self.system_prompt = system_prompt  # Store the system prompt

        if isinstance(log_level, str):
            self.log_level = parse_log_level(log_level)
        else:
            self.log_level = log_level

        self.logger = get_logger("agent", self.log_level)

        # Initialize the async ToolManager (private)
        self._tool_manager = ToolManager(
            messager, log_level=self.log_level, register_topic=register_topic
        )

        # Initialize Langchain output parsers
        self.response_parser = PydanticOutputParser(pydantic_object=LLMResponse)
        self.tool_call_parser = PydanticOutputParser(pydantic_object=LLMResponseToolCall)
        self.direct_parser = PydanticOutputParser(pydantic_object=LLMResponseDirect)
        self.tool_result_parser = PydanticOutputParser(pydantic_object=LLMResponseToolResult)

        self.prompt_template = self._build_prompt_template()
        if not self.raw_template:
            raise ValueError(
                "Agent raw_template was not set during _build_prompt_template initialization."
            )
        self.max_tool_iterations = 10

        self.history: List[BaseMessage] = []
        self.logger.info(
            "Agent initialized with consistent message pattern: direct fields + data=None."
        )

    async def async_init(self):
        """Async initialization for Agent, including tool manager subscriptions."""
        # Subscribe tool manager to registration topics
        await self._tool_manager.async_init()
        self.logger.info("Agent: ToolManager initialized via async_init")

    def _build_prompt_template(self) -> PromptTemplate:
        # Use provided system prompt or default
        system_prompt = (
            self.system_prompt
            if self.system_prompt is not None
            else self._get_default_system_prompt()
        )

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

    def set_system_prompt(self, system_prompt: str) -> None:
        """
        Update the system prompt and rebuild the prompt template.

        Args:
            system_prompt: New system prompt to use
        """
        self.system_prompt = system_prompt
        self.prompt_template = self._build_prompt_template()
        self.logger.info("System prompt updated and prompt template rebuilt")

    def get_system_prompt(self) -> str:
        """
        Get the current system prompt (either custom or default).

        Returns:
            The current system prompt being used
        """
        if self.system_prompt is not None:
            return self.system_prompt
        else:
            # Return the default prompt by calling _build_prompt_template logic
            return self._get_default_system_prompt()

    def _get_default_system_prompt(self) -> str:
        """
        Returns the default system prompt.

        Returns:
            The default system prompt string
        """
        return """You are a highly capable AI assistant that MUST follow these strict response format rules:

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
   - Incorporate the tool results into your answer *if they are relevant and helpful*. If the tool results are not helpful or empty, state that briefly and answer using your general knowledge.

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
    - If the results are empty or not relevant to the original question, briefly state that the tool didn't provide useful information, then answer the original question using your general knowledge, still using the "tool_result" format but explaining the situation in the 'result' field.
- If you're unsure after getting tool results, use the "tool_result" format and explain your reasoning in the 'result' field.
- Never make another tool call immediately after receiving tool results unless absolutely necessary and clearly justified.

{format_instructions}"""

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
        self, tool_call_requests: List[ToolCallRequest]
    ) -> Tuple[List[BaseMessage], bool]:
        """
        Executes tool calls parsed from LLM output.
        tool_calls_dicts: List of dictionaries, each representing a tool call,
                          e.g., {'tool_id': 'some_tool', 'arguments': {...}}
        Returns a list of history-formatted messages and a boolean indicating errors.
        """
        if not tool_call_requests:
            return [], False

        # execution_outcomes is List[Union[TaskResultMessage, TaskErrorMessage]]
        execution_outcomes, any_errors_from_manager = await self._tool_manager.get_tool_results(
            tool_call_requests
        )

        processed_outcomes: List[BaseMessage] = []
        for outcome in execution_outcomes:
            if isinstance(outcome, (TaskResultMessage, TaskErrorMessage)):
                processed_outcomes.append(outcome)
            else:
                self.logger.error(f"Unexpected outcome type from ToolManager: {type(outcome)}")
                # Create a generic error message with direct fields + data=None
                processed_outcomes.append(
                    TaskErrorMessage(
                        tool_id=getattr(outcome, "tool_id", "unknown_id"),
                        tool_name=getattr(outcome, "tool_name", "unknown_name"),
                        task_id=getattr(outcome, "task_id", "unknown_task_id"),
                        error=f"Unexpected outcome type from ToolManager: {type(outcome)}",
                        source=MessageSource.AGENT,
                        data=None,
                    )
                )
        return processed_outcomes, any_errors_from_manager

    async def query(
        self, question: str, user_id: Optional[str] = None, max_iterations: Optional[int] = None
    ) -> str:
        """
        Processes a question through the LLM and tool interaction loop.
        """
        if max_iterations is None:
            max_iterations = self.max_tool_iterations

        self.history = []

        if not self.raw_template:
            self.logger.error(
                "Agent raw_template is not initialized before query! This should not happen if __init__ completed."
            )
            return "Error: Agent prompt template not initialized. Critical error."

        # 1. Add System Prompt to history
        system_prompt_content = self.raw_template.split("Available Tools:")[0].strip()
        self.history.append(
            AgentSystemMessage(
                content=system_prompt_content, source=MessageSource.SYSTEM, data=None
            )
        )

        # 2. Add original user question to history
        user_source = MessageSource.USER if user_id else MessageSource.CLIENT
        self.history.append(
            AskQuestionMessage(question=question, user_id=user_id, source=user_source, data=None)
        )

        current_question_for_llm_turn = question

        for i in range(max_iterations):
            self.logger.info(
                f"Query Iteration: {i+1}/{max_iterations} for user '{user_id or 'Unknown'}'... Current prompt: {current_question_for_llm_turn[:100]}..."
            )

            tools_description_str = self._tool_manager.get_tools_description()

            # Format the user prompt for THIS specific turn, including tools and current question
            current_turn_formatted_prompt = self.prompt_template.format(
                tool_descriptions=tools_description_str,
                question=current_question_for_llm_turn,
                format_instructions=self.response_parser.get_format_instructions(),
            )

            # Create AgentLLMRequestMessage for this turn's interaction (not added to history before conversion)
            llm_input_messages = self._convert_protocol_history_to_llm_format(self.history)
            # Append the specifically formatted user message for the current turn
            llm_input_messages.append(
                {"role": LLMRole.USER.value, "content": current_turn_formatted_prompt}
            )

            llm_response_raw_text = await self._call_llm(llm_input_messages)
            self.logger.debug(f"LLM raw response (Iter {i+1}): {llm_response_raw_text[:300]}...")

            # Instantiate AgentLLMResponseMessage with direct fields + data=None
            llm_response_msg = AgentLLMResponseMessage(
                raw_content=llm_response_raw_text, source=MessageSource.LLM, data=None
            )

            # Use Langchain parser to parse the response
            validated_response = await self._parse_llm_response_with_langchain(
                llm_response_raw_text
            )

            if validated_response is None:
                self.logger.error(f"Could not parse LLM response: {llm_response_raw_text}")
                # Assign to direct fields of the message object
                llm_response_msg.parsed_type = "error_parsing"
                llm_response_msg.error_details = "Could not parse LLM response."
                self.history.append(llm_response_msg)
                current_question_for_llm_turn = f"Previous response could not be parsed. Please resubmit in proper JSON format. Original question: {question}"
                continue

            # Handle the different response types
            if isinstance(validated_response, LLMResponseDirect):
                llm_response_msg.parsed_type = "direct"
                llm_response_msg.parsed_direct_content = validated_response.content
                self.history.append(llm_response_msg)
                self.logger.debug(f"LLM direct response: {validated_response.content[:100]}...")
                return validated_response.content

            elif isinstance(validated_response, LLMResponseToolResult):
                llm_response_msg.parsed_type = "tool_result"
                llm_response_msg.parsed_tool_result_content = validated_response.result
                self.history.append(llm_response_msg)
                self.logger.debug(f"LLM tool_result response: {validated_response.result[:100]}...")
                return validated_response.result

            elif isinstance(validated_response, LLMResponseToolCall):
                llm_response_msg.parsed_type = "tool_call"
                # Convert ToolCallRequest to MinimalToolCallRequest for storage
                llm_response_msg.parsed_tool_calls = [
                    MinimalToolCallRequest(tool_id=tc.tool_id, arguments=tc.arguments)
                    for tc in validated_response.tool_calls
                ]
                self.history.append(llm_response_msg)

                if not llm_response_msg.parsed_tool_calls:
                    self.logger.warning(
                        "'tool_call' type with no tool_calls. Asking LLM to clarify."
                    )
                    current_question_for_llm_turn = f"Indicated 'tool_call' but provided no tools. Please clarify or answer directly for: {question}"
                    continue

                # Convert back to ToolCallRequest for execution
                tool_call_requests = [
                    ToolCallRequest(tool_id=tc.tool_id, arguments=tc.arguments)
                    for tc in llm_response_msg.parsed_tool_calls
                ]
                tool_outcome_messages, had_error = await self._execute_tool_calls(
                    tool_call_requests
                )
                for outcome_msg in tool_outcome_messages:
                    self.history.append(outcome_msg)

                if had_error:
                    self.logger.warning(
                        "Tool execution had errors. Asking LLM to summarize for user."
                    )
                    current_question_for_llm_turn = f"Errors occurred during tool execution (see history). Explain this to the user and answer the original question: '{question}' if possible. Use 'direct' format."
                else:
                    self.logger.info("Tool execution successful. Asking LLM to process results.")
                    current_question_for_llm_turn = f"Tool execution finished (see history). Analyze results and answer the original question: '{question}'. Use 'tool_result' format."
                continue
            else:
                self.logger.error(f"Unknown validated response type: {type(validated_response)}")
                llm_response_msg.parsed_type = "error_validation"
                llm_response_msg.error_details = (
                    f"Unknown response type: {type(validated_response)}"
                )
                self.history.append(llm_response_msg)
                current_question_for_llm_turn = f"Unknown response format received. Please resubmit. Original question: {question}"
                continue

        self.logger.warning(f"Max iterations ({max_iterations}) reached for: {question}")
        if self.history:
            last_msg = self.history[-1]
            if isinstance(last_msg, AgentLLMResponseMessage):
                if last_msg.parsed_type == "direct" and last_msg.parsed_direct_content:
                    return last_msg.parsed_direct_content
                elif last_msg.parsed_type == "tool_result" and last_msg.parsed_tool_result_content:
                    return last_msg.parsed_tool_result_content
        return "Max iterations reached. Unable to provide a conclusive answer."

    async def handle_ask_question(self, message: AskQuestionMessage):
        """
        MQTT handler for incoming questions.
        - message: the parsed AskQuestionMessage or dict
        """
        try:
            # Access fields directly from the message object
            question: str = message.question
            user_id: Optional[str] = message.user_id

            self.logger.info(
                f"Received question from user '{user_id or 'Unknown'} via {message.source}': {question}"
            )

            answer_text = await self.query(question, user_id=user_id)

            # Create user-specific answer topic
            if user_id:
                user_answer_topic = f"{self.answer_topic}/{user_id}"
            else:
                # Fallback to global topic for clients without user_id
                user_answer_topic = self.answer_topic

            # Create AnswerMessage with direct fields + data=None
            answer_msg = AnswerMessage(
                question=question,
                answer=answer_text,
                user_id=user_id,
                source=MessageSource.AGENT,
                data=None,
            )
            await self.messager.publish(user_answer_topic, answer_msg)
            self.logger.info(
                f"Published answer to {user_answer_topic} for user '{user_id or 'Unknown'}'"
            )

        except Exception as e:
            self.logger.error(f"Error handling ask_question: {e}", exc_info=True)
            try:
                # Also use user-specific topic for error messages
                user_id = getattr(message, "user_id", None)
                if user_id:
                    error_topic = f"{self.answer_topic}/{user_id}"
                else:
                    error_topic = self.answer_topic

                error_msg = AnswerMessage(
                    question=getattr(message, "question", "Unknown question"),
                    error=f"Agent error: {str(e)}",
                    user_id=user_id,
                    source=MessageSource.AGENT,
                    data=None,
                )
                await self.messager.publish(error_topic, error_msg)
            except Exception as pub_e:
                self.logger.error(f"Failed to publish error answer: {pub_e}")

    async def _publish_answer(
        self, question: str, response_content: str, user_id: Optional[str] = None
    ) -> None:
        """
        DEPRECATED or REPURPOSED: This method's original purpose of extracting content
        is now handled by ModelProviders. It might be removed or adapted if there's
        a different specific need for publishing answers outside handle_ask_question.
        For now, it's kept but likely unused by the main flow.
        """
        try:
            # Use user-specific topic if user_id is provided
            if user_id:
                publish_topic = f"{self.answer_topic}/{user_id}"
            else:
                publish_topic = self.answer_topic

            answer = AnswerMessage(
                question=question,
                answer=response_content,
                user_id=user_id,
                source=MessageSource.AGENT,
                data=None,
            )
            await self.messager.publish(publish_topic, answer)
            self.logger.info(f"Published answer (via _publish_answer) to {publish_topic}")
        except Exception as e:
            self.logger.error(f"Error in _publish_answer: {e}", exc_info=True)

    def _convert_protocol_history_to_llm_format(
        self, history_messages: List[BaseMessage]
    ) -> List[Dict[str, str]]:
        llm_formatted_messages: List[Dict[str, str]] = []
        for msg in history_messages:
            if isinstance(msg, AgentSystemMessage):
                llm_formatted_messages.append(
                    {"role": LLMRole.SYSTEM.value, "content": msg.content}
                )
            elif isinstance(msg, AskQuestionMessage):
                llm_formatted_messages.append({"role": LLMRole.USER.value, "content": msg.question})
            elif isinstance(msg, AgentLLMRequestMessage):
                llm_formatted_messages.append({"role": LLMRole.USER.value, "content": msg.prompt})
            elif isinstance(msg, AgentLLMResponseMessage):
                llm_formatted_messages.append(
                    {"role": LLMRole.ASSISTANT.value, "content": msg.raw_content}
                )
            elif isinstance(msg, TaskResultMessage):
                content = ""
                if msg.result is not None:
                    if isinstance(msg.result, (str, int, float, bool)):
                        content = str(msg.result)
                    elif isinstance(msg.result, (dict, list)):
                        try:
                            content = json.dumps(msg.result)
                        except TypeError:
                            content = f"Tool returned complex object: {str(msg.result)[:100]}..."
                    else:
                        content = f"Tool returned unhandled type: {type(msg.result)}"
                else:
                    content = "Tool executed successfully but returned no content."

                llm_formatted_messages.append(
                    {
                        "role": LLMRole.TOOL.value,
                        "tool_call_id": msg.task_id or msg.tool_id or "unknown_tool_call_id",
                        "name": msg.tool_name or "unknown_tool",
                        "content": content,
                    }
                )
            elif isinstance(msg, TaskErrorMessage):
                error_content = f"Error executing tool {msg.tool_name or 'unknown'}: {msg.error}"
                llm_formatted_messages.append(
                    {
                        "role": LLMRole.TOOL.value,
                        "tool_call_id": msg.task_id or msg.tool_id or "unknown_tool_call_id",
                        "name": msg.tool_name or "unknown_tool",
                        "content": error_content,
                    }
                )
            else:
                self.logger.warning(
                    f"Unrecognized message type in history for LLM conversion: {type(msg)}"
                )
        return llm_formatted_messages

    async def _parse_llm_response_with_langchain(
        self, response_text: str
    ) -> Optional[Union[LLMResponseToolCall, LLMResponseDirect, LLMResponseToolResult]]:
        """
        Parse LLM response using Langchain output parsers.
        Returns the appropriate parsed response model or None if parsing fails.
        """
        try:
            # First try to parse with the general response parser
            parsed_response = self.response_parser.parse(response_text)

            # Based on the type, validate with the specific parser
            if parsed_response.type == "tool_call":
                # Validate with specific tool call parser
                return self.tool_call_parser.parse(response_text)
            elif parsed_response.type == "direct":
                # Validate with specific direct parser
                return self.direct_parser.parse(response_text)
            elif parsed_response.type == "tool_result":
                # Validate with specific tool result parser
                return self.tool_result_parser.parse(response_text)
            else:
                self.logger.error(f"Unknown response type: {parsed_response.type}")
                return None

        except Exception as e:
            self.logger.error(f"Langchain parser failed: {e}")
            # Fall back to original parsing method if needed
            return await self._parse_llm_response_fallback(response_text)

    async def _parse_llm_response_fallback(
        self, response_text: str
    ) -> Optional[Union[LLMResponseToolCall, LLMResponseDirect, LLMResponseToolResult]]:
        """
        Fallback parsing method using the original manual approach.
        This ensures backward compatibility if Langchain parsing fails.
        """
        try:
            # Simple JSON extraction logic
            cleaned_json_str = response_text.strip()
            if "```" in cleaned_json_str:
                code_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
                code_blocks = re.findall(code_block_pattern, cleaned_json_str)
                if code_blocks:
                    cleaned_json_str = code_blocks[0].strip()
                else:
                    cleaned_json_str = (
                        cleaned_json_str.replace("```json", "").replace("```", "").strip()
                    )

            if not (cleaned_json_str.startswith("{") and cleaned_json_str.endswith("}")):
                match = re.search(r"\{[\s\S]*\}", response_text)
                if match:
                    cleaned_json_str = match.group(0)

            if not cleaned_json_str:
                return None

            parsed_dict = json.loads(cleaned_json_str)
            response_type = parsed_dict.get("type")

            if response_type == "direct":
                return LLMResponseDirect.model_validate(parsed_dict)
            elif response_type == "tool_result":
                return LLMResponseToolResult.model_validate(parsed_dict)
            elif response_type == "tool_call":
                return LLMResponseToolCall.model_validate(parsed_dict)
            else:
                return None

        except Exception as e:
            self.logger.error(f"Fallback parsing also failed: {e}")
            return None
