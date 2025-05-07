import asyncio
import uuid
import json
import traceback
from typing import Dict, Any, Optional, Union
from datetime import datetime, timezone

import aiomqtt  # Import aiomqtt for type hint
from pydantic import ValidationError

from core.protocol.message import (
    RegisterToolMessage,
    UnregisterToolMessage,
    ToolRegisteredMessage,
    TaskMessage,
    TaskResultMessage,
    TaskStatus,
    AnyMessage,
)
from core.messager import Messager
from core.logger import get_logger, LogLevel, parse_log_level


class ToolManager:
    """Manages tool registration, execution via MQTT, and description generation (Async Version)."""

    def __init__(self, messager: Messager, log_level: Union[LogLevel, str] = LogLevel.INFO):
        self.messager = messager

        if isinstance(log_level, str):
            self.log_level = parse_log_level(log_level)
        else:
            self.log_level = log_level

        self.logger = get_logger("tool_manager", level=self.log_level)

        self.tools: Dict[str, Dict[str, Any]] = {}
        self._pending_tasks: Dict[str, asyncio.Future] = {}
        self._task_results: Dict[str, TaskResultMessage] = {}
        self._late_results_cache: Dict[str, tuple[float, TaskResultMessage]] = {}
        self._pending_late_results: Dict[str, Dict[str, Any]] = {}
        self._result_lock = asyncio.Lock()
        self._subscribed_result_topics = set()
        self._late_results_ttl = 60  # seconds
        self._default_timeout = 30  # seconds

        self.register_topic = "agent/tools/register"
        self.call_topic = "agent/tools/call"
        self.response_topic_base = "agent/tools/response"

        self.logger.info("ToolManager initialized (Async Version)")

    async def async_init(self):
        """Perform any async initialization needed after MQTT connection."""
        self.logger.info("Running ToolManager async initialization...")
        self.logger.info("ToolManager async initialization complete.")

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

    async def _handle_tool_message(self, msg: AnyMessage):
        """Handles tool registration and unregistration messages (protocol message only)."""
        self.logger.debug(f"Received tool message: {msg}")
        try:
            if isinstance(msg, RegisterToolMessage):
                await self._handle_register_tool(msg)
            elif isinstance(msg, UnregisterToolMessage):
                await self._handle_unregister_tool(msg)
            else:
                self.logger.warning(
                    f"Received unexpected message type {type(msg).__name__} on tool message topic. Ignoring."
                )
        except (ValueError, ValidationError) as e:
            self.logger.error(f"Failed to parse/validate tool message: {e}. Message: '{msg}'")
        except Exception as e:
            self.logger.error(f"Failed to handle tool message: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")

    async def _handle_result_message(self, msg: TaskResultMessage):
        """Handles incoming task result messages (protocol message only)."""
        self.logger.debug(f"Received TaskResultMessage: {msg}")
        try:
            task_id = msg.task_id

            async with self._result_lock:
                future = self._pending_tasks.get(task_id)
                if future and not future.done():
                    self.logger.info(f"Received expected result for task {task_id}")
                    self._task_results[task_id] = msg
                    future.set_result(msg)
                elif task_id in self._pending_late_results:
                    self.logger.info(f"Received late result for task {task_id}. Caching.")
                    late_info = self._pending_late_results.get(task_id)
                    if late_info:
                        cache_key = f"{late_info['tool_name']}_{json.dumps(late_info['arguments'], sort_keys=True)}"
                        self._late_results_cache[cache_key] = (
                            datetime.now(timezone.utc).timestamp(),
                            msg,
                        )
                        del self._pending_late_results[task_id]
                    else:
                        self.logger.warning(
                            f"Could not cache late result for {task_id}, pending info missing."
                        )
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
            self.tools[tool_id] = tool_info
            await self.messager.subscribe(tool_info["result_topic"], self._handle_result_message)
            self._subscribed_result_topics.add(tool_info["result_topic"])

        self.logger.info(
            f"Registered tool '{tool_info['name']}' ({tool_id}). Subscribed to {tool_info['result_topic']} for results."
        )

        confirmation = ToolRegisteredMessage(
            source=self.messager.client_id,
            tool_id=tool_id,
            tool_name=reg_msg.tool_name,
            recipient=reg_msg.source,
        )
        status_topic = "agent/status/info"
        await self.messager.publish(status_topic, confirmation.model_dump_json())
        self.logger.info(
            f"Sent registration confirmation for '{reg_msg.tool_name}' to {reg_msg.source} via {status_topic}"
        )

    async def _handle_unregister_tool(self, unreg_msg: UnregisterToolMessage):
        """Handles tool unregistration messages (Async Version)."""
        tool_id = unreg_msg.tool_id
        tool_name = unreg_msg.tool_name

        self.logger.info(
            f"Received unregistration request for tool '{tool_name}' (ID: {tool_id}) from '{unreg_msg.source}'"
        )

        async with self._result_lock:
            if tool_id not in self.tools:
                self.logger.warning(f"Cannot unregister unknown tool ID '{tool_id}'. Ignoring.")
                return

            tool_info = self.tools[tool_id]

            registered_source = tool_info.get("source_service_id")
            if registered_source != unreg_msg.source:
                self.logger.warning(
                    f"Unregistration request from '{unreg_msg.source}' doesn't match registered source '{registered_source}' for tool ID '{tool_id}'. Ignoring."
                )
                return

            result_topic = tool_info.get("result_topic")
            if result_topic and result_topic in self._subscribed_result_topics:
                try:
                    await self.messager.unsubscribe(result_topic)
                    self._subscribed_result_topics.remove(result_topic)
                    self.logger.info(f"Unsubscribed from result topic: {result_topic}")
                except Exception as e:
                    self.logger.error(f"Error unsubscribing from result topic {result_topic}: {e}")

            del self.tools[tool_id]
            self.logger.info(f"Successfully unregistered tool '{tool_name}' (ID: {tool_id})")

    async def execute_tool(
        self, tool_name: str, arguments: Dict[str, Any], timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Execute a tool with timeout (Async Version).

        Args:
            tool_name (str): The name of the tool to execute.
            arguments (Dict[str, Any]): The arguments to pass to the tool.
            timeout (Optional[float], optional): Override timeout in seconds. Defaults to tool's registered timeout or global default.

        Returns:
            Dict[str, Any]: The result parameters from TaskResultMessage or an error dictionary.
        """
        async with self._result_lock:
            self._clean_expired_cache_entries()
            cached_result = self._check_late_results_cache(tool_name, arguments)

        if cached_result is not None:
            self.logger.info(
                f"Using cached late result for tool '{tool_name}' with args {arguments}"
            )
            return cached_result.params

        tool: Optional[Dict[str, Any]] = None
        async with self._result_lock:
            for t_id, t_info in self.tools.items():
                if isinstance(t_info, dict) and t_info.get("name") == tool_name:
                    tool = t_info
                    break

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
                    if task_id in self._pending_tasks:
                        del self._pending_tasks[task_id]
                    if task_id in self._task_results:
                        del self._task_results[task_id]

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
                    f"Tool '{tool_name}' timed out after {effective_timeout}s (task_id: {task_id}). "
                    f"Will cache any late responses."
                )
                async with self._result_lock:
                    if task_id in self._pending_tasks:
                        del self._pending_tasks[task_id]
                    self._pending_late_results[task_id] = {
                        "tool_name": tool_name,
                        "arguments": arguments,
                    }
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

    def _find_tool(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Finds a registered tool by name. Assumes lock is held or not needed."""
        for tool_id, tool_info in self.tools.items():
            if isinstance(tool_info, dict) and tool_info.get("name") == tool_name:
                return tool_info
        return None

    def _check_late_results_cache(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Optional[TaskResultMessage]:
        """Checks for cached late results. Assumes lock is held."""
        cache_key = f"{tool_name}_{json.dumps(arguments, sort_keys=True)}"
        if cache_key in self._late_results_cache:
            timestamp, result = self._late_results_cache[cache_key]
            if datetime.now(timezone.utc).timestamp() - timestamp <= self._late_results_ttl:
                self.logger.debug(f"Using valid late result cache entry for key: {cache_key}")
                del self._late_results_cache[cache_key]
                return result
            else:
                self.logger.debug(f"Found expired late result cache entry, removing: {cache_key}")
                del self._late_results_cache[cache_key]
        return None

    def _clean_expired_cache_entries(self):
        """Removes expired entries from the late results cache. Assumes lock is held."""
        now = datetime.now(timezone.utc).timestamp()
        expired_keys = [
            key
            for key, (timestamp, _) in self._late_results_cache.items()
            if now - timestamp > self._late_results_ttl
        ]
        if expired_keys:
            self.logger.debug(f"Cleaning {len(expired_keys)} expired late result cache entries.")
            for key in expired_keys:
                del self._late_results_cache[key]

    def generate_tool_descriptions_for_prompt(self) -> str:
        """Generates a JSON string describing available tools for the LLM prompt."""
        if not self.tools:
            return "[]"

        tool_defs = []
        for tool_id, tool_info in list(self.tools.items()):
            if not isinstance(tool_info, dict):
                self.logger.warning(
                    f"Skipping tool {tool_id} in description generation: Invalid format {type(tool_info)}"
                )
                continue
            try:
                if not all(k in tool_info for k in ["name", "description", "api_schema"]):
                    self.logger.warning(
                        f"Skipping tool {tool_id} ('{tool_info.get('name', 'N/A')}') in description generation: Missing required fields (name, description, api_schema)."
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
                    f"Error generating definition for tool {tool_id} ('{tool_info.get('name', 'N/A')}'): {e}"
                )

        if not tool_defs:
            return "[]"

        return json.dumps(tool_defs, indent=2)
