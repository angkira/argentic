import asyncio
import re
import json
from typing import List, Optional, Union, Tuple, Dict, Literal, Any
import functools
import concurrent.futures

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

from argentic.core.messager.messager import Messager
from argentic.core.protocol.message import (
    BaseMessage,
    AgentSystemMessage,
    AgentLLMRequestMessage,
    AgentLLMResponseMessage,
    AskQuestionMessage,
    AnswerMessage,
    MinimalToolCallRequest,
)
from argentic.core.protocol.enums import MessageSource, LLMRole
from argentic.core.protocol.tool import ToolCallRequest
from argentic.core.protocol.task import (
    TaskResultMessage,
    TaskErrorMessage,
    TaskStatus,
)
from argentic.core.tools.tool_manager import ToolManager
from argentic.core.logger import get_logger, LogLevel, parse_log_level
from argentic.core.llm.providers.base import ModelProvider
from argentic.core.graph.state import AgentState
from langchain_core.messages import (
    BaseMessage as LangchainBaseMessage,
    AIMessage,
    HumanMessage,
    ToolMessage,
    SystemMessage,
)
from langchain_core.messages.tool import ToolCall


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
        tool_manager: Optional[ToolManager] = None,
        log_level: Union[str, LogLevel] = LogLevel.INFO,
        register_topic: str = "agent/tools/register",
        tool_call_topic_base: str = "agent/tools/call",
        tool_response_topic_base: str = "agent/tools/response",
        status_topic: str = "agent/status/info",
        answer_topic: str = "agent/response/answer",
        llm_response_topic: Optional[str] = None,
        tool_result_topic: Optional[str] = None,
        system_prompt: Optional[str] = None,
        role: str = "agent",
        graph_id: Optional[str] = None,
        expected_output_format: Literal["json", "text", "code"] = "json",  # New parameter
    ):
        self.llm = llm
        self.messager = messager
        self.answer_topic = answer_topic
        self.llm_response_topic = llm_response_topic
        self.tool_result_topic = tool_result_topic
        self.raw_template: Optional[str] = None
        self.system_prompt = system_prompt
        self.role = role
        self.graph_id = graph_id
        self.expected_output_format = expected_output_format  # Store new parameter

        if isinstance(log_level, str):
            self.log_level = parse_log_level(log_level)
        else:
            self.log_level = log_level

        self.logger = get_logger("agent", self.log_level)

        # Use provided tool manager or create a new one
        if tool_manager is not None:
            self._tool_manager = tool_manager
            self.logger.info(f"Agent '{self.role}': Using provided tool manager")
        else:
            # Initialize the async ToolManager (private)
            self._tool_manager = ToolManager(
                messager,
                log_level=self.log_level,
                register_topic=register_topic,
                tool_call_topic_base=tool_call_topic_base,
                tool_response_topic_base=tool_response_topic_base,
                status_topic=status_topic,
            )
            self.logger.info(f"Agent '{self.role}': Created new tool manager")

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

        # Create a dedicated thread pool for heavy LLM operations
        # This prevents blocking the default thread pool and allows better parallelism
        self._llm_thread_pool_size = 8  # Configurable in future
        self._llm_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self._llm_thread_pool_size, thread_name_prefix=f"agent-{self.role}-llm"
        )

        self.history: List[BaseMessage] = []
        self.logger.info(
            "Agent initialized with consistent message pattern: direct fields + data=None."
        )

    async def invoke(self, state: AgentState) -> dict[str, list[LangchainBaseMessage]]:
        """
        Invokes the agent as part of a graph. This represents one turn of the agent.
        """
        self.logger.info(f"Agent '{self.role}' invoked with {len(state['messages'])} messages.")

        # Convert Langchain messages from state to our internal protocol messages
        protocol_messages = self._convert_langchain_to_protocol_messages(state["messages"])

        llm_input_messages = self._convert_protocol_history_to_llm_format(protocol_messages)

        # The Supervisor will have agent_tools, a regular agent will not.
        tools = getattr(self, "agent_tools", None)
        # Only pass tools if there are actually tools available
        if tools and len(tools) > 0:
            llm_langchain_response: LangchainBaseMessage = await self._call_llm(
                llm_input_messages, tools=tools
            )  # Call LLM with tools
        else:
            llm_langchain_response: LangchainBaseMessage = await self._call_llm(
                llm_input_messages
            )  # Call LLM without tools

        # Get raw content string for logging and AgentLLMResponseMessage
        # This should capture either the direct text content or a string representation of tool calls
        llm_response_raw_text = ""
        if isinstance(llm_langchain_response, AIMessage) and llm_langchain_response.tool_calls:
            # Serialize tool calls to a JSON string for raw_content
            llm_response_raw_text = json.dumps(llm_langchain_response.tool_calls)
        elif llm_langchain_response.content is not None:
            # Handle cases where content might not be a simple string (e.g., list of parts for multimodal)
            if isinstance(llm_langchain_response.content, str):
                llm_response_raw_text = llm_langchain_response.content
            else:
                llm_response_raw_text = str(llm_langchain_response.content)

        self.logger.debug(f"LLM raw response for '{self.role}': {llm_response_raw_text[:300]}...")

        llm_response_msg = AgentLLMResponseMessage(
            raw_content=llm_response_raw_text, source=MessageSource.LLM, data=None  # type: ignore
        )

        # Use Langchain parser to parse the response
        validated_response = await self._parse_llm_response_with_langchain(
            llm_langchain_response
        )  # Pass BaseMessage

        output_messages: List[BaseMessage] = []
        if isinstance(validated_response, LLMResponseToolCall):
            self.logger.info(f"Agent '{self.role}' is calling tools.")
            tool_call_requests = [
                ToolCallRequest(tool_id=tc.tool_id, arguments=tc.arguments)
                for tc in validated_response.tool_calls
            ]
            tool_outcome_messages, _ = await self._execute_tool_calls(tool_call_requests)
            output_messages = [llm_response_msg] + tool_outcome_messages

        elif validated_response:
            self.logger.info(f"Agent '{self.role}' is providing a direct answer or tool result.")
            # If validated_response is LLMResponseDirect or LLMResponseToolResult
            if isinstance(validated_response, LLMResponseDirect):
                llm_response_msg.parsed_type = validated_response.type
                llm_response_msg.parsed_direct_content = validated_response.content
                llm_response_msg.parsed_tool_result_content = None  # Ensure it's explicitly None
            elif isinstance(validated_response, LLMResponseToolResult):
                llm_response_msg.parsed_type = validated_response.type
                llm_response_msg.parsed_direct_content = None  # Ensure it's explicitly None
                llm_response_msg.parsed_tool_result_content = validated_response.result
            else:
                # This case should ideally not be reached if validated_response is always one of the expected types
                self.logger.warning(
                    f"Unexpected validated_response type in invoke: {type(validated_response)}"
                )
                llm_response_msg.parsed_type = "error_validation"
                llm_response_msg.error_details = (
                    f"Unexpected response type: {type(validated_response)}"
                )
            output_messages = [llm_response_msg]

        else:
            self.logger.error(f"Could not parse LLM response for agent '{self.role}'")
            error_msg = AgentSystemMessage(
                content="Error: Could not parse LLM response. Please check the format.",
                source=MessageSource.SYSTEM,
            )
            output_messages = [error_msg]

        # Convert our protocol messages back to Langchain messages for the state
        langchain_output_messages = self._convert_protocol_to_langchain_messages(output_messages)
        return {"messages": langchain_output_messages}

    def _convert_langchain_to_protocol_messages(
        self, messages: List[LangchainBaseMessage]
    ) -> List[BaseMessage]:
        protocol_msgs = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                protocol_msgs.append(
                    AskQuestionMessage(question=str(msg.content), source=MessageSource.USER)
                )
            elif isinstance(msg, AIMessage):
                # In multi-agent scenarios, AIMessages from other agents should be treated as questions
                # Only treat as LLM response if this is truly from our own LLM
                if hasattr(msg, "content") and msg.content:
                    # Convert AIMessage content to a question for this agent
                    content_str = (
                        str(msg.content) if not isinstance(msg.content, str) else msg.content
                    )
                    protocol_msgs.append(
                        AskQuestionMessage(question=content_str, source=MessageSource.AGENT)
                    )
                else:
                    # Fallback: treat as LLM response message
                    raw_content = ""
                    if msg.tool_calls:
                        raw_content = str(json.dumps(msg.tool_calls))
                    elif msg.content is not None:
                        if isinstance(msg.content, str):
                            raw_content = msg.content
                        else:
                            raw_content = str(msg.content)
                    protocol_msgs.append(
                        AgentLLMResponseMessage(raw_content=raw_content, source=MessageSource.LLM)
                    )
            elif isinstance(msg, SystemMessage):
                protocol_msgs.append(
                    AgentSystemMessage(content=str(msg.content), source=MessageSource.SYSTEM)
                )
            elif isinstance(msg, ToolMessage):
                # This is a simplification. We might need to handle tool call IDs later.
                protocol_msgs.append(
                    TaskResultMessage(
                        tool_id=msg.tool_call_id,
                        tool_name=msg.tool_call_id,
                        status=TaskStatus.SUCCESS,
                        result=msg.content,
                    )
                )
        return protocol_msgs

    def _convert_protocol_to_langchain_messages(
        self, messages: List[BaseMessage]
    ) -> List[LangchainBaseMessage]:
        langchain_msgs = []
        for msg in messages:
            if isinstance(msg, (AskQuestionMessage, AgentLLMRequestMessage)):
                langchain_msgs.append(
                    HumanMessage(
                        content=msg.question if isinstance(msg, AskQuestionMessage) else msg.prompt
                    )
                )
            elif isinstance(msg, AgentLLMResponseMessage):
                # If it's an AgentLLMResponseMessage, its raw_content needs to be preserved
                # if it contains a tool call. Otherwise, it's just plain content.
                try:
                    parsed_raw = json.loads(msg.raw_content)
                    if isinstance(parsed_raw, list) and all("name" in tc for tc in parsed_raw):
                        # This is a list of tool calls in a simplified dict format from raw_content
                        lc_tool_calls = []
                        for tc_dict in parsed_raw:
                            # Ensure tc_dict is a dict and has 'name' and 'args' keys
                            if (
                                isinstance(tc_dict, dict)
                                and "name" in tc_dict
                                and "args" in tc_dict
                            ):
                                lc_tool_calls.append(
                                    ToolCall(
                                        name=tc_dict["name"],
                                        args=tc_dict["args"],
                                        id=tc_dict.get("id"),
                                    )
                                )
                            else:
                                self.logger.warning(
                                    f"Malformed tool call dict in raw_content: {tc_dict}"
                                )
                        langchain_msgs.append(
                            AIMessage(content="", tool_calls=lc_tool_calls)
                        )  # Pass empty content if tool_calls exist
                    else:
                        # If it's JSON but not tool calls, or just direct content, pass raw_content as content
                        langchain_msgs.append(AIMessage(content=msg.raw_content))
                except json.JSONDecodeError:
                    # If raw_content is not JSON, it's direct content
                    langchain_msgs.append(AIMessage(content=msg.raw_content))

            elif isinstance(msg, AgentSystemMessage):
                langchain_msgs.append(SystemMessage(content=str(msg.content)))
            elif isinstance(msg, TaskResultMessage):
                langchain_msgs.append(
                    ToolMessage(
                        content=str(msg.result), tool_call_id=msg.tool_name or "unknown_tool"
                    )
                )
            elif isinstance(msg, TaskErrorMessage):
                langchain_msgs.append(
                    ToolMessage(
                        content=str(msg.error), tool_call_id=msg.tool_name or "unknown_tool"
                    )
                )

        return langchain_msgs

    async def async_init(self):
        """
        Async initialization for Agent, including tool manager subscriptions.
        """
        # Initialize tool manager if not already done
        try:
            # Try to call async_init - it's safe to call multiple times
            await self._tool_manager.async_init()
            self.logger.info("Agent: ToolManager initialized via async_init")
        except Exception as e:
            self.logger.warning(
                f"Agent: ToolManager async_init error (may be already initialized): {e}"
            )

    async def stop(self) -> None:
        """Stop the agent and clean up resources."""
        self.logger.info(f"Stopping agent '{self.role}'...")

        # Shutdown the dedicated LLM thread pool
        self._llm_executor.shutdown(wait=True)
        self.logger.debug("LLM thread pool executor shut down.")

        self.logger.info(f"Agent '{self.role}' stopped.")

    def _build_prompt_template(self) -> PromptTemplate:
        # Use provided system prompt or default
        system_prompt = (
            self.system_prompt
            if self.system_prompt is not None
            else self._get_default_system_prompt()
        )

        # The prompt template itself doesn't need to change much, just how format_instructions is handled
        if self.expected_output_format == "json":
            template = """{system_prompt_content}

Available Tools:
{tool_descriptions}

QUESTION: {question}

YOUR RESPONSE MUST BE A SINGLE JSON OBJECT. DO NOT INCLUDE ANY OTHER TEXT, COMMENTS, OR MARKDOWN OUTSIDE THE JSON BLOCK. ONLY THE JSON BLOCK IS ACCEPTABLE.
{format_instructions}"""
        else:
            # Simplified template for text/code output
            template = """{system_prompt_content}

{question}"""

        self.raw_template = template

        if self.expected_output_format == "json":
            return PromptTemplate.from_template(
                template,
                partial_variables={
                    "system_prompt_content": system_prompt,
                    "tool_descriptions": (
                        self._tool_manager.get_tools_description()
                        if self.expected_output_format == "json"
                        else ""
                    ),
                    "question": "",  # Question is handled dynamically, not via partial_variables for the base template
                    "format_instructions": self.response_parser.get_format_instructions(),
                },
            )
        else:
            return PromptTemplate.from_template(
                template,
                partial_variables={
                    "system_prompt_content": system_prompt,
                    "question": "",  # Question is handled dynamically
                },
            )

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
        if self.expected_output_format == "json":
            return """You are a highly capable AI assistant that MUST follow these strict response format rules.

YOUR RESPONSE MUST BE A SINGLE JSON OBJECT. DO NOT INCLUDE ANY OTHER TEXT, COMMENTS, OR MARKDOWN OUTSIDE THE JSON BLOCK. ONLY THE JSON BLOCK IS ACCEPTABLE.

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
4. NEVER include text outside the JSON structure. THIS IS CRITICAL. ONLY THE JSON BLOCK IS ALLOWED.
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
- You're unsure after getting tool results, use the "tool_result" format and explain your reasoning in the 'result' field.
- Never make another tool call immediately after receiving tool results unless absolutely necessary and clearly justified.
"""
        elif self.expected_output_format == "text":
            return "You are an AI assistant that provides direct, concise textual answers. Do not use any special formatting or JSON."
        elif self.expected_output_format == "code":
            return "You are an AI assistant that generates code. Provide only the code, without any extra text or formatting unless specifically requested."
        else:
            raise ValueError(f"Unknown expected_output_format: {self.expected_output_format}")

    async def _call_llm(
        self, messages: List[Dict[str, str]], tools: Optional[List[Any]] = None, **kwargs
    ) -> LangchainBaseMessage:
        """
        Calls the appropriate LLM method using the ModelProvider interface.
        ModelProvider methods (achat, chat) are expected to return a BaseMessage.
        """
        self.logger.info(f"Agent '{self.role}': Starting LLM call...")

        # Prefer async chat method if available
        if hasattr(self.llm, "achat"):
            self.logger.debug(f"Using async chat method from provider: {type(self.llm).__name__}")
            # For achat, we expect tools to be passed directly for binding
            result = await self.llm.achat(messages, tools=tools, **kwargs)
        elif hasattr(self.llm, "chat"):
            self.logger.debug(
                f"Using sync chat method in executor from provider: {type(self.llm).__name__}"
            )
            loop = asyncio.get_running_loop()
            # For chat, functools.partial is needed to bind tools for sync call in executor
            chat_with_tools = functools.partial(self.llm.chat, messages, tools=tools, **kwargs)
            # Use dedicated LLM thread pool to prevent blocking the default executor
            result = await loop.run_in_executor(self._llm_executor, chat_with_tools)
        elif hasattr(self.llm, "ainvoke"):
            self.logger.warning(
                f"Provider {type(self.llm).__name__} does not have 'achat'. "
                "Falling back to 'ainvoke'. Chat history might not be optimally handled."
            )
            # If tools are provided, ainvoke might not handle them directly. This path is less optimal.
            prompt = self.llm._format_chat_messages_to_prompt(messages)
            result = await self.llm.ainvoke(prompt, **kwargs)
        elif hasattr(self.llm, "invoke"):
            self.logger.warning(
                f"Provider {type(self.llm).__name__} does not have 'chat' methods. "
                "Falling back to 'invoke' in executor. Chat history might not be optimally handled."
            )
            # If tools are provided, invoke might not handle them directly. This path is less optimal.
            loop = asyncio.get_running_loop()
            prompt = self.llm._format_chat_messages_to_prompt(messages)
            # Use dedicated LLM thread pool to prevent blocking the default executor
            invoke_with_args = functools.partial(self.llm.invoke, prompt, **kwargs)
            result = await loop.run_in_executor(self._llm_executor, invoke_with_args)
        else:
            self.logger.error(
                f"LLM provider {type(self.llm).__name__} has no recognized "
                "callable method (achat, chat, ainvoke, invoke)."
            )
            raise TypeError(
                f"LLM provider {type(self.llm).__name__} has no recognized callable method."
            )

        self.logger.info(f"Agent '{self.role}': LLM call completed")
        return result

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
                if self.tool_result_topic:
                    await self.messager.publish(self.tool_result_topic, outcome)
            else:
                self.logger.error(f"Unexpected outcome type from ToolManager: {type(outcome)}")
                # Create a generic error message with direct fields + data=None
                error_msg = TaskErrorMessage(
                    tool_id=getattr(outcome, "tool_id", "unknown_id"),
                    tool_name=getattr(outcome, "tool_name", "unknown_name"),
                    task_id=getattr(outcome, "task_id", "unknown_task_id"),
                    error=f"Unexpected outcome type from ToolManager: {type(outcome)}",
                    source=MessageSource.AGENT,
                    data=None,
                )
                processed_outcomes.append(error_msg)
                if self.tool_result_topic:
                    await self.messager.publish(self.tool_result_topic, error_msg)

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
        user_source = MessageSource.USER if user_id else MessageSource.AGENT
        self.history.append(
            AskQuestionMessage(question=question, user_id=user_id, source=user_source, data=None)
        )

        current_question_for_llm_turn = question

        for i in range(max_iterations):
            self.logger.info(
                f"Query Iteration: {i+1}/{max_iterations} for user '{user_id or 'Unknown'}'... Current prompt: {current_question_for_llm_turn[:100]}..."
            )

            tools_description_str = self._tool_manager.get_tools_description()
            # Escape curly braces in the JSON string for literal interpretation by PromptTemplate's formatter
            escaped_tool_descriptions_str = tools_description_str.replace("{", "{{").replace(
                "}", "}}"
            )

            # Format the user prompt for THIS specific turn, including tools and current question
            current_turn_formatted_prompt = self.prompt_template.format(
                tool_descriptions=escaped_tool_descriptions_str,
                question=current_question_for_llm_turn,
            )

            # Create AgentLLMRequestMessage for this turn's interaction (not added to history before conversion)
            llm_input_messages = self._convert_protocol_history_to_llm_format(self.history)
            # Append the specifically formatted user message for the current turn
            llm_input_messages.append(
                {"role": LLMRole.USER.value, "content": current_turn_formatted_prompt}
            )

            llm_langchain_response: LangchainBaseMessage = await self._call_llm(
                llm_input_messages
            )  # Call LLM directly returning BaseMessage

            # Get raw content string for logging and AgentLLMResponseMessage
            # This should capture either the direct text content or a string representation of tool calls
            llm_response_raw_text = ""
            if isinstance(llm_langchain_response, AIMessage) and llm_langchain_response.tool_calls:
                # Serialize tool calls to a JSON string for raw_content
                llm_response_raw_text = json.dumps(llm_langchain_response.tool_calls)
            elif llm_langchain_response.content is not None:
                # Handle cases where content might not be a simple string (e.g., list of parts for multimodal)
                if isinstance(llm_langchain_response.content, str):
                    llm_response_raw_text = llm_langchain_response.content
                else:
                    llm_response_raw_text = str(llm_langchain_response.content)

            self.logger.debug(f"LLM raw response (Iter {i+1}): {llm_response_raw_text[:300]}...")

            # Instantiate AgentLLMResponseMessage with direct fields + data=None
            llm_response_msg = AgentLLMResponseMessage(
                raw_content=llm_response_raw_text, source=MessageSource.LLM, data=None
            )
            if self.llm_response_topic:
                await self.messager.publish(self.llm_response_topic, llm_response_msg)

            # Use Langchain parser to parse the response
            validated_response = await self._parse_llm_response_with_langchain(
                llm_langchain_response
            )  # Pass BaseMessage

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
                    MinimalToolCallRequest(tool_name=tc.tool_id, arguments=tc.arguments)
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
                    ToolCallRequest(tool_id=tc.tool_name, arguments=tc.arguments)
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
        self, llm_message: LangchainBaseMessage
    ) -> Optional[Union[LLMResponseToolCall, LLMResponseDirect, LLMResponseToolResult]]:
        """
        Parse LLM response based on expected_output_format from a LangChain BaseMessage.
        Returns the appropriate parsed response model or None if parsing fails.
        """
        # --- Handle native tool calls directly from AIMessage (most reliable for Gemini) ---
        if isinstance(llm_message, AIMessage) and llm_message.tool_calls:
            self.logger.debug(f"Detected native tool calls in AIMessage for agent '{self.role}'.")
            # Convert tool calls (which might be dicts or ToolCall objects) to our internal ToolCallRequest format
            tool_calls_for_response = []
            for lc_tool_call_dict in llm_message.tool_calls:
                tool_id = lc_tool_call_dict.get("name") or lc_tool_call_dict.get("id")
                # Arguments can be in 'args' or 'arguments' depending on LLM/LangChain version
                arguments = lc_tool_call_dict.get("args", lc_tool_call_dict.get("arguments", {}))

                if tool_id:
                    tool_calls_for_response.append(
                        ToolCallRequest(
                            tool_id=tool_id,
                            arguments=arguments,
                        )
                    )
                else:
                    self.logger.warning(
                        f"Could not extract tool_id or arguments from tool call dict: {lc_tool_call_dict}"
                    )

            if not tool_calls_for_response:
                self.logger.warning(
                    f"No valid tool calls extracted from LLM message, falling back to direct: {llm_message.tool_calls}"
                )
                # If no valid tool calls could be extracted, treat as a direct answer
                # Fall through to the direct/tool_result parsing logic.
                pass
            else:
                return LLMResponseToolCall(type="tool_call", tool_calls=tool_calls_for_response)

        # --- Main parsing logic based on expected_output_format and coerced direct answers ---
        if self.expected_output_format == "json":
            # If it's JSON expected, but not a native tool call from AIMessage (handled above),
            # then it must be a direct response or tool_result from a previous step.
            # We attempt to parse its content as a JSON string.
            # Ensure content is a string, even if it was originally None or a list (e.g., multimodal content)
            cleaned_json_content = ""
            if llm_message.content is not None:
                if isinstance(llm_message.content, str):
                    cleaned_json_content = llm_message.content
                else:
                    # If content is not a string (e.g., list of parts), coerce to string representation
                    cleaned_json_content = str(llm_message.content)

            # Attempt to strip markdown code blocks if the LLM wrapped it despite instructions
            if cleaned_json_content.startswith("```json") and cleaned_json_content.endswith("```"):
                cleaned_json_content = cleaned_json_content[len("```json") : -len("```")].strip()
            elif cleaned_json_content.startswith("```") and cleaned_json_content.endswith("```"):
                cleaned_json_content = cleaned_json_content[len("```") : -len("```")].strip()

            try:
                # Try to parse with the general response parser for structured direct/tool_result
                parsed_response = self.response_parser.parse(cleaned_json_content)

                if parsed_response.type == "direct":
                    return self.direct_parser.parse(cleaned_json_content)
                elif parsed_response.type == "tool_result":
                    return self.tool_result_parser.parse(cleaned_json_content)
                else:
                    self.logger.error(
                        f"Unknown response type detected by general parser: {parsed_response.type}. Raw: {cleaned_json_content[:100]}..."
                    )
                    return None
            except Exception as e:
                self.logger.error(
                    f"Langchain JSON parser failed for structured output: {e}. Raw: {cleaned_json_content[:200]}..."
                )

                # If JSON parsing fails, but there's content, assume it's a direct answer.
                if cleaned_json_content:
                    self.logger.warning(
                        f"Coercing raw response into direct answer due to JSON parsing failure. Raw: {cleaned_json_content[:100]}..."
                    )
                    return LLMResponseDirect(type="direct", content=cleaned_json_content)
                else:
                    self.logger.error(
                        f"Empty content after JSON parsing failure. Raw: {llm_message.content[:100] if llm_message.content else '[EMPTY]'}..."
                    )
                    return None

        elif self.expected_output_format in ["text", "code"]:
            # For text or code, simply return the raw content of the message wrapped in LLMResponseDirect
            content_to_use = ""
            if llm_message.content is not None:
                if isinstance(llm_message.content, str):
                    content_to_use = llm_message.content
                else:
                    content_to_use = str(llm_message.content)

            if content_to_use:
                return LLMResponseDirect(type="direct", content=content_to_use)
            else:
                self.logger.warning(
                    f"LLM returned empty response for expected_output_format='{self.expected_output_format}'."
                )
                return LLMResponseDirect(type="direct", content="[Empty response]")

        else:
            self.logger.error(f"Unsupported expected_output_format: {self.expected_output_format}")
            return None

    async def _parse_llm_response_fallback(
        self, response_text: str
    ) -> Optional[Union[LLMResponseToolCall, LLMResponseDirect, LLMResponseToolResult]]:
        # This method is now effectively deprecated and will return None.
        # All parsing logic is consolidated in _parse_llm_response_with_langchain
        self.logger.debug(
            f"_parse_llm_response_fallback called (should be deprecated): {response_text[:100]}..."
        )
        return None
