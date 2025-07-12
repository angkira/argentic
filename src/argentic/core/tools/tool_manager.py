import asyncio
import json
import traceback
from uuid import uuid4
from datetime import datetime, timezone
from logging import Logger
from typing import Any, Dict, Optional, Union, List, Tuple

from argentic.core.logger import LogLevel, get_logger, parse_log_level
from argentic.core.messager.messager import Messager
from argentic.core.protocol.task import (
    TaskMessage,
    TaskResultMessage,
    TaskErrorMessage,
    TaskStatus,
)
from argentic.core.protocol.tool import (
    RegisterToolMessage,
    ToolRegisteredMessage,
    UnregisterToolMessage,
    ToolCallRequest,
)


class ToolManager:
    """Manages tool registration, execution, and description generation (Async Version)."""

    tools_by_id: Dict[str, Dict[str, Any]]
    tools_by_name: Dict[str, Dict[str, Any]]
    _pending_tasks: Dict[str, asyncio.Future]
    _result_lock: asyncio.Lock
    _default_timeout: int
    register_topic: str
    tool_call_topic_base: str
    tool_response_topic_base: str
    status_topic: str
    messager: Messager
    log_level: LogLevel
    logger: Logger

    def __init__(
        self,
        messager: Messager,
        log_level: Union[LogLevel, str] = LogLevel.INFO,
        register_topic: str = "agent/tools/register",
        tool_call_topic_base: str = "agent/tools/call",
        tool_response_topic_base: str = "agent/tools/response",
        status_topic: str = "agent/status/info",
        default_timeout: int = 30,
    ):
        self.messager = messager
        if isinstance(log_level, str):
            self.log_level = parse_log_level(log_level)
        else:
            self.log_level = log_level
        self.logger = get_logger("tool_manager", self.log_level)

        self.tools_by_id = {}
        self.tools_by_name = {}
        self._pending_tasks = {}
        self._result_lock = asyncio.Lock()
        self._default_timeout = default_timeout
        self.register_topic = register_topic
        self.tool_call_topic_base = tool_call_topic_base
        self.tool_response_topic_base = tool_response_topic_base
        self.status_topic = status_topic
        self.logger.info(
            f"ToolManager initialized. Call base: '{self.tool_call_topic_base}', Response base: '{self.tool_response_topic_base}'"
        )

    async def async_init(self) -> None:
        """Asynchronously initializes the ToolManager, subscribing to tool registration topics."""
        await self.messager.subscribe(
            self.register_topic,
            self._handle_register_tool,
            message_cls=RegisterToolMessage,
        )
        self.logger.info(
            f"ToolManager subscribed to tool registration topic: {self.register_topic}"
        )
        self.logger.info("ToolManager async initialization complete.")

    def set_log_level(self, level: Union[LogLevel, str]) -> None:
        if isinstance(level, str):
            self.log_level = parse_log_level(level)
        else:
            self.log_level = level
        self.logger.setLevel(self.log_level.value)
        for handler in self.logger.handlers:
            handler.setLevel(self.log_level.value)
        self.logger.info(f"ToolManager log level set to {self.log_level.name}")

    async def _handle_result_message(self, msg: TaskResultMessage):
        async with self._result_lock:
            task_id = msg.task_id
            if task_id in self._pending_tasks:
                future = self._pending_tasks.pop(task_id)
                if not future.done():
                    future.set_result(msg)
                    self.logger.info(f"Received result for task {task_id}, status: {msg.status}")
                else:
                    self.logger.warning(
                        f"Received result for already completed/cancelled task {task_id}."
                    )
            else:
                self.logger.warning(f"Received result for unknown or timed-out task {task_id}.")

    async def _handle_register_tool(self, reg_msg: RegisterToolMessage):
        tool_id = str(uuid4())

        tool_name = reg_msg.tool_name

        async with self._result_lock:
            if tool_id in self.tools_by_id:
                self.logger.warning(
                    f"Tool with ID {tool_id} ({tool_name}) is already registered. Re-registering."
                )

            tool_info = {
                "id": tool_id,
                "name": tool_name,
                "description": reg_msg.tool_manual,
                "parameters": reg_msg.tool_api,
                "registered_at": datetime.now(timezone.utc).isoformat(),
                "source_client_id": reg_msg.source,
            }
            self.tools_by_id[tool_id] = tool_info
            self.tools_by_name[tool_name] = tool_info

        result_topic = f"{self.tool_response_topic_base}/{tool_id}"

        await self.messager.subscribe(
            result_topic,
            self._handle_result_message,
            message_cls=TaskResultMessage,
        )
        self.logger.info(
            f"Registered tool '{tool_name}' (ID: {tool_id}). Subscribed to {result_topic} for results."
        )

        confirmation_msg = ToolRegisteredMessage(
            tool_id=tool_id,
            tool_name=tool_name,
            status="registered",
            message=f"Tool '{tool_name}' registered successfully by agent.",
            source=self.messager.client_id,
        )
        await self.messager.publish(self.status_topic, confirmation_msg.model_dump_json())
        self.logger.info(
            f"Sent registration confirmation for '{tool_name}' to {reg_msg.source} via {self.status_topic}"
        )

    async def _handle_unregister_tool(self, unreg_msg: UnregisterToolMessage):
        tool_id = unreg_msg.tool_id
        async with self._result_lock:
            if tool_id in self.tools_by_id:
                tool_info = self.tools_by_id.pop(tool_id)
                if tool_info["name"] in self.tools_by_name:
                    del self.tools_by_name[tool_info["name"]]

                result_topic = f"{self.tool_response_topic_base}/{tool_id}"
                await self.messager.unsubscribe(result_topic)
                self.logger.info(
                    f"Unregistered tool '{tool_info['name']}' (ID: {tool_id}). Unsubscribed from {result_topic}."
                )
            else:
                self.logger.warning(f"Attempted to unregister unknown tool ID: {tool_id}")

    async def execute_tool(
        self,
        tool_name_or_id: str,
        arguments: Dict[str, Any],
        task_id_override: Optional[str] = None,
    ) -> Union[TaskResultMessage, TaskErrorMessage]:
        tool_info = self.tools_by_id.get(tool_name_or_id) or self.tools_by_name.get(tool_name_or_id)

        actual_tool_name = tool_info["name"] if tool_info else tool_name_or_id
        tool_id_val = tool_info["id"] if tool_info else "unknown"

        if not tool_info:
            self.logger.error(f"Tool '{tool_name_or_id}' not found.")
            return TaskErrorMessage(
                tool_id=tool_id_val,
                tool_name=actual_tool_name,
                task_id=task_id_override or str(uuid4()),
                status=TaskStatus.FAILED,
                error=f"Tool '{tool_name_or_id}' not found.",
                arguments=arguments,
            )

        tool_id = tool_info["id"]
        task_id = task_id_override or str(uuid4())
        effective_timeout = self._default_timeout

        task_msg = TaskMessage(
            task_id=task_id,
            tool_id=tool_id,
            tool_name=actual_tool_name,
            arguments=arguments,
            source=self.messager.client_id,
        )

        future = asyncio.Future()
        async with self._result_lock:
            self._pending_tasks[task_id] = future

        task_topic = f"{self.tool_call_topic_base}/{tool_id}"

        try:
            await self.messager.publish(task_topic, task_msg.model_dump_json())
            self.logger.info(
                f"Sent task message to {task_topic} (tool: {actual_tool_name}, task_id: {task_id}, timeout: {effective_timeout}s)"
            )

            result_message: TaskResultMessage = await asyncio.wait_for(
                future, timeout=effective_timeout
            )

            if result_message.status not in [
                TaskStatus.FAILED,
                TaskStatus.ERROR,
                TaskStatus.TIMEOUT,
            ]:
                result_message.error = None
            return result_message

        except asyncio.TimeoutError:
            self.logger.warning(
                f"Tool '{actual_tool_name}' timed out after {effective_timeout}s (task_id: {task_id})."
            )
            async with self._result_lock:
                if task_id in self._pending_tasks:
                    self._pending_tasks.pop(task_id)
            return TaskErrorMessage(
                task_id=task_id,
                tool_id=tool_id,
                tool_name=actual_tool_name,
                status=TaskStatus.TIMEOUT,
                error=f"Tool execution timed out after {effective_timeout} seconds.",
                arguments=arguments,
            )
        except Exception as e:
            self.logger.error(
                f"Error executing tool '{actual_tool_name}' (task_id: {task_id}): {e}",
                exc_info=True,
            )
            async with self._result_lock:
                if task_id in self._pending_tasks:
                    self._pending_tasks.pop(task_id)
            return TaskErrorMessage(
                task_id=task_id,
                tool_id=tool_id,
                tool_name=actual_tool_name,
                status=TaskStatus.FAILED,
                error=str(e),
                traceback=traceback.format_exc() if self.log_level <= LogLevel.DEBUG else None,
                arguments=arguments,
            )

    def generate_tool_descriptions_for_prompt(self) -> str:
        """Generates a JSON string describing available tools for the LLM prompt."""
        tool_defs = []
        current_tools = list(self.tools_by_id.values())
        for tool_info in current_tools:
            tool_defs.append(
                {
                    "tool_id": tool_info["id"],
                    "name": tool_info["name"],
                    "description": tool_info["description"],
                    "parameters": tool_info["parameters"],
                }
            )
        return json.dumps(tool_defs, indent=2)

    def get_tools_description(self) -> str:
        return self.generate_tool_descriptions_for_prompt()

    def get_tool_names(self) -> List[str]:
        return list(self.tools_by_name.keys())

    async def get_tool_results(
        self, tool_call_requests: List[ToolCallRequest]
    ) -> Tuple[List[Union[TaskResultMessage, TaskErrorMessage]], bool]:
        if not tool_call_requests:
            return [], False

        results: List[Union[TaskResultMessage, TaskErrorMessage]] = []
        any_errors = False

        tasks = []
        for call_req in tool_call_requests:
            tool_identifier = call_req.tool_id
            arguments = call_req.arguments
            task_id_for_exec = str(uuid4())
            self.logger.info(
                f"Scheduling tool '{tool_identifier}' with args: {arguments} (exec_task_id: {task_id_for_exec})"
            )
            tasks.append(
                self.execute_tool(tool_identifier, arguments, task_id_override=task_id_for_exec)
            )

        tool_execution_outcomes = await asyncio.gather(*tasks, return_exceptions=True)

        for i, outcome_or_exc in enumerate(tool_execution_outcomes):
            original_request = tool_call_requests[i]
            tool_info = self.tools_by_id.get(original_request.tool_id) or self.tools_by_name.get(
                original_request.tool_id
            )
            tool_name_for_error = tool_info["name"] if tool_info else original_request.tool_id
            tool_id_for_error = tool_info["id"] if tool_info else original_request.tool_id

            if isinstance(outcome_or_exc, Exception):
                self.logger.error(
                    f"Unhandled exception during tool call for '{original_request.tool_id}': {outcome_or_exc}",
                    exc_info=outcome_or_exc,
                )
                results.append(
                    TaskErrorMessage(
                        tool_id=tool_id_for_error,
                        tool_name=tool_name_for_error,
                        task_id=str(uuid4()),
                        status=TaskStatus.FAILED,
                        error=f"System exception during execution: {str(outcome_or_exc)}",
                        arguments=original_request.arguments,
                    )
                )
                any_errors = True
            elif isinstance(outcome_or_exc, (TaskResultMessage, TaskErrorMessage)):
                results.append(outcome_or_exc)
                current_status = outcome_or_exc.status
                if current_status in [TaskStatus.FAILED, TaskStatus.ERROR, TaskStatus.TIMEOUT]:
                    error_msg = (
                        outcome_or_exc.error
                        if hasattr(outcome_or_exc, "error") and outcome_or_exc.error
                        else "Tool execution failed or timed out."
                    )
                    self.logger.error(
                        f"Error result from tool '{outcome_or_exc.tool_name}': {error_msg}"
                    )
                    any_errors = True
                else:
                    self.logger.info(
                        f"Result from tool '{outcome_or_exc.tool_name}': {getattr(outcome_or_exc, 'result', 'N/A')}"
                    )
            else:
                self.logger.error(
                    f"Unexpected result type from gather for tool '{original_request.tool_id}': {type(outcome_or_exc)}"
                )
                results.append(
                    TaskErrorMessage(
                        tool_id=tool_id_for_error,
                        tool_name=tool_name_for_error,
                        task_id=str(uuid4()),
                        status=TaskStatus.FAILED,
                        error="Unknown error or unexpected result type during tool execution.",
                        arguments=original_request.arguments,
                    )
                )
                any_errors = True
        return results, any_errors
