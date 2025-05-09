import asyncio
import json
import traceback
import uuid
from datetime import datetime, timezone
from logging import Logger
from typing import Any, Dict, Optional, Union

from core.logger import LogLevel, get_logger, parse_log_level
from core.messager.messager import Messager
from core.protocol.task import (
    TaskMessage,
    TaskResultMessage,
    TaskStatus,
)
from core.protocol.tool import (
    RegisterToolMessage,
    ToolRegisteredMessage,
    UnregisterToolMessage,
)


class ToolManager:
    """Manages tool registration, execution, and description generation (Async Version)."""

    tools_by_id: Dict[str, Dict[str, Any]]
    tools_by_name: Dict[str, Dict[str, Any]]
    _pending_tasks: Dict[str, asyncio.Future]
    _result_lock: asyncio.Lock
    _default_timeout: int
    register_topic: str
    status_topic: str
    messager: Messager
    log_level: LogLevel
    logger: Logger

    def __init__(
        self,
        messager: Messager,
        log_level: Union[LogLevel, str] = LogLevel.INFO,
        register_topic: str = "agent/tools/register",
        status_topic: str = "agent/status/info",
    ):
        self.messager = messager

        if isinstance(log_level, str):
            self.log_level = parse_log_level(log_level)
        else:
            self.log_level = log_level

        self.logger = get_logger("tool_manager", level=self.log_level)

        # Stores for tools: by ID and by name
        self.tools_by_id: Dict[str, Dict[str, Any]] = {}
        self.tools_by_name: Dict[str, Dict[str, Any]] = {}
        self._pending_tasks: Dict[str, asyncio.Future] = {}
        self._result_lock = asyncio.Lock()
        self._default_timeout = 30  # seconds

        self.register_topic = register_topic
        self.status_topic = status_topic

        # Subscriptions moved to async_init()

        self.logger.info("ToolManager async initialization complete.")

    async def async_init(self) -> None:
        """Asynchronously subscribe to registration and unregistration message types."""
        # Subscribe separately to each tool message type
        await self.messager.subscribe(
            self.register_topic,
            self._handle_register_tool,
            message_cls=RegisterToolMessage,
        )
        await self.messager.subscribe(
            self.register_topic,
            self._handle_unregister_tool,
            message_cls=UnregisterToolMessage,
        )
        self.logger.info(
            f"ToolManager subscribed to tool registration and unregistration on: {self.register_topic}"
        )

    def set_log_level(self, level: Union[LogLevel, str]) -> None:
        """Set logging level for the ToolManager"""
        if isinstance(level, str):
            self.log_level = parse_log_level(level)
        else:
            self.log_level = level

        self.logger.setLevel(self.log_level.value)
        self.logger.info(f"Log level changed to {self.log_level.name}")

        for handler in self.logger.handlers:
            handler.setLevel(self.log_level.value)

    async def _handle_result_message(self, msg: TaskResultMessage):
        """Handles incoming task result messages (protocol message only)."""
        self.logger.debug(f"Received TaskResultMessage: {msg}")
        try:
            task_id = msg.task_id

            async with self._result_lock:
                future = self._pending_tasks.get(task_id)
                if future and not future.done():
                    self.logger.info(f"Received expected result for task {task_id}")
                    future.set_result(msg)
                    self.logger.info(f"Set result for task {task_id} in future.")
                else:
                    self.logger.warning(
                        f"Received result for unknown or already timed-out task {task_id}. Discarding."
                    )
        except Exception as e:
            self.logger.error(f"Unexpected error handling result message: {e}")
            self.logger.error(traceback.format_exc())

    async def _handle_register_tool(self, reg_msg: RegisterToolMessage):
        """Handles tool registration messages (Async Version)."""
        self.logger.info(f"Registering tool '{reg_msg.tool_name}' from '{reg_msg.source}'")

        tool_id = str(uuid.uuid4())
        tool_info = {
            "tool_id": tool_id,
            "name": reg_msg.tool_name,
            "description": reg_msg.tool_manual,
            "api_schema": json.loads(reg_msg.tool_api),
            "source_service_id": reg_msg.source,
            "task_topic": f"tool/{tool_id}/task",
            "result_topic": f"tool/{tool_id}/result",
            "timeout": getattr(reg_msg, "timeout", self._default_timeout),
        }

        async with self._result_lock:
            # store in both maps
            self.tools_by_id[tool_id] = tool_info
            self.tools_by_name[reg_msg.tool_name] = tool_info
            # subscribe for results
            await self.messager.subscribe(
                tool_info["result_topic"],
                self._handle_result_message,
                message_cls=TaskResultMessage,
            )

        self.logger.info(
            f"Registered tool '{tool_info['name']}' ({tool_id}). Subscribed to {tool_info['result_topic']} for results."
        )

        confirmation = ToolRegisteredMessage(
            source=self.messager.client_id,
            tool_id=tool_id,
            tool_name=reg_msg.tool_name,
            recipient=reg_msg.source,
        )

        # publish confirmation on configured status topic
        await self.messager.publish(self.status_topic, confirmation)

        self.logger.info(
            f"Sent registration confirmation for '{reg_msg.tool_name}' to {reg_msg.source} via {self.status_topic}"
        )

    async def _handle_unregister_tool(self, unreg_msg: UnregisterToolMessage):
        """Handles tool unregistration messages (Async Version)."""
        tool_id = unreg_msg.tool_id
        tool_name = unreg_msg.tool_name

        self.logger.info(
            f"Received unregistration request for tool '{tool_name}' (ID: {tool_id}) from '{unreg_msg.source}'"
        )

        async with self._result_lock:
            if tool_id not in self.tools_by_id:
                self.logger.warning(f"Cannot unregister unknown tool ID '{tool_id}'. Ignoring.")
                return

            tool_info = self.tools_by_id[tool_id]

            registered_source = tool_info.get("source_service_id")
            if registered_source != unreg_msg.source:
                self.logger.warning(
                    f"Unregistration request from '{unreg_msg.source}' doesn't match registered source '{registered_source}' for tool ID '{tool_id}'. Ignoring."
                )
                return

            result_topic = tool_info.get("result_topic")
            if result_topic:
                try:
                    await self.messager.unsubscribe(result_topic)
                    self.logger.info(f"Unsubscribed from result topic: {result_topic}")
                except Exception as e:
                    self.logger.error(f"Error unsubscribing from result topic {result_topic}: {e}")

            # remove from both stores
            del self.tools_by_id[tool_id]
            self.tools_by_name.pop(tool_info["name"], None)
            self.logger.info(f"Successfully unregistered tool '{tool_name}' (ID: {tool_id})")

    async def execute_tool(
        self, tool_name: str, arguments: Dict[str, Any], timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Execute a tool with timeout.

        Args:
            tool_name (str): The name of the tool to execute.
            arguments (Dict[str, Any]): The arguments to pass to the tool.
            timeout (Optional[float], optional): Override timeout in seconds. Defaults to tool's registered timeout or global default.

        Returns:
            Dict[str, Any]: The result parameters from TaskResultMessage or an error dictionary.
        """

        # lookup by tool name
        tool = self.tools_by_name.get(tool_name)

        if not tool:
            err_msg = f"Tool '{tool_name}' not found or not registered."
            self.logger.error(err_msg)
            return {"error": err_msg, "status": TaskStatus.ERROR}

        effective_timeout = (
            timeout if timeout is not None else tool.get("timeout", self._default_timeout)
        )

        task_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        future: asyncio.Future[TaskResultMessage] = loop.create_future()

        async with self._result_lock:
            self._pending_tasks[task_id] = future

        self.logger.info(
            f"Executing tool '{tool_name}' (ID: {tool['tool_id']}) with args: {arguments} "
            f"(task_id: {task_id}, timeout: {effective_timeout}s)"
        )
        task_start_time = datetime.now(timezone.utc).timestamp()

        try:
            task_topic = tool["task_topic"]
            tool_id = tool["tool_id"]

            task_message = TaskMessage(
                source=self.messager.client_id,
                tool_id=tool_id,
                task_id=task_id,
                payload=arguments,
            )

            await self.messager.publish(task_topic, task_message.model_dump_json())
            self.logger.info(f"Sent task message to {task_topic} (task_id: {task_id})")

            try:
                result_message = await asyncio.wait_for(future, timeout=effective_timeout)
                execution_time = datetime.now(timezone.utc).timestamp() - task_start_time
                self.logger.info(
                    f"Tool '{tool_name}' executed successfully in {execution_time:.2f}s (task_id: {task_id})"
                )
                async with self._result_lock:
                    self._pending_tasks.pop(task_id, None)

                if result_message.status == TaskStatus.SUCCESS:
                    return result_message.params
                else:
                    self.logger.warning(
                        f"Tool '{tool_name}' (task_id: {task_id}) reported status: {result_message.status}. Result: {result_message.params}"
                    )
                    error_detail = result_message.params.get(
                        "error", f"Tool reported status {result_message.status}"
                    )
                    return {
                        "error": error_detail,
                        "status": result_message.status,
                        "task_id": task_id,
                    }

            except asyncio.TimeoutError:
                self.logger.warning(
                    f"Tool '{tool_name}' timed out after {effective_timeout}s (task_id: {task_id})."
                )
                async with self._result_lock:
                    self._pending_tasks.pop(task_id, None)
                return {
                    "error": f"Tool execution timed out after {effective_timeout}s",
                    "status": TaskStatus.TIMEOUT,
                    "task_id": task_id,
                }

        except Exception as e:
            self.logger.error(
                f"Error during setup or publishing for tool '{tool_name}' (task_id: {task_id}): {e}"
            )
            self.logger.error(traceback.format_exc())
            async with self._result_lock:
                if task_id in self._pending_tasks:
                    pending_future = self._pending_tasks.pop(task_id)
                    if not pending_future.done():
                        pending_future.set_exception(e)
            return {
                "error": f"Error executing tool: {str(e)}",
                "status": TaskStatus.ERROR,
                "task_id": task_id,
            }

    def generate_tool_descriptions_for_prompt(self) -> str:
        """Generates a JSON string describing available tools for the LLM prompt."""
        if not self.tools_by_name:
            return "[]"

        tool_defs = []
        for tool_info in self.tools_by_name.values():
            if not isinstance(tool_info, dict):
                self.logger.warning(
                    f"Skipping tool in description generation: Invalid format {type(tool_info)}"
                )
                continue
            try:
                if not all(k in tool_info for k in ["name", "description", "api_schema"]):
                    self.logger.warning(
                        f"Skipping tool '{tool_info.get('name', 'N/A')}' in description generation: Missing required fields."
                    )
                    continue

                tool_def = {
                    "type": "function",
                    "function": {
                        "name": tool_info["name"].replace(" ", "_").replace("-", "_").strip(),
                        "description": tool_info["description"],
                        "parameters": tool_info["api_schema"],
                    },
                }
                tool_defs.append(tool_def)
            except (KeyError, TypeError, json.JSONDecodeError) as e:
                self.logger.error(
                    f"Error generating definition for tool '{tool_info.get('name', 'N/A')}': {e}"
                )

        if not tool_defs:
            return "[]"

        return json.dumps(tool_defs, indent=2)
