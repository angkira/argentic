import json
import threading
import uuid
import traceback
from typing import Dict, Any, Optional, Union
from datetime import datetime, timezone
import time

from pydantic import ValidationError
from core.protocol.message import (
    RegisterToolMessage,
    UnregisterToolMessage,
    ToolRegisteredMessage,
    TaskMessage,
    TaskResultMessage,
    TaskStatus,
    from_mqtt_message,
    AnyMessage,
    MessageType,
)
from .messager import Messager, MQTTMessage
from .tool_base import BaseTool
from .logger import get_logger, LogLevel, parse_log_level


class ToolManager:
    """Manages tool registration, execution via MQTT, and description generation."""

    def __init__(self, messager: Messager, log_level: Union[LogLevel, str] = LogLevel.INFO):
        self.messager = messager

        # Configure logger with the specified level
        if isinstance(log_level, str):
            self.log_level = parse_log_level(log_level)
        else:
            self.log_level = log_level

        self.logger = get_logger("tool_manager", self.log_level)

        self.tools: Dict[str, Dict[str, Any]] = {}
        self._pending_tasks: Dict[str, threading.Event] = {}
        self._task_results: Dict[str, TaskResultMessage] = {}
        self._late_results_cache: Dict[str, (float, TaskResultMessage)] = {}
        self._pending_late_results: Dict[str, Dict[str, Any]] = {}
        self._result_lock = threading.Lock()
        self._subscribed_result_topics = set()
        self._late_results_ttl = 60  # seconds to keep late results in cache
        self._default_timeout = 30  # default timeout in seconds for tool execution

        self.logger.info("ToolManager initialized")
        self.messager.mqtt_log("ToolManager initialized.")

        # Also subscribe to dynamic tool registration requests
        register_topic = "rag/command/register_tool"
        self.messager.subscribe(register_topic, self._handle_tool_message)
        self.logger.info(f"Subscribed to register topic: {register_topic}")
        self.messager.mqtt_log(f"ToolManager subscribed to register topic: {register_topic}")

    def set_log_level(self, level: Union[LogLevel, str]) -> None:
        """Set logging level for the ToolManager"""
        if isinstance(level, str):
            self.log_level = parse_log_level(level)
        else:
            self.log_level = level

        self.logger.setLevel(self.log_level.value)
        self.logger.info(f"Log level changed to {self.log_level.name}")

        # Update handlers
        for handler in self.logger.handlers:
            handler.setLevel(self.log_level.value)

    def _handle_tool_message(self, message: MQTTMessage):
        """Handles tool registration and unregistration messages."""
        try:
            # Deserialize and validate using the helper
            parsed_message: AnyMessage = from_mqtt_message(message)

            # Handle different tool-related message types
            if isinstance(parsed_message, RegisterToolMessage):
                self._handle_register_tool(parsed_message)
            elif isinstance(parsed_message, UnregisterToolMessage):
                self._handle_unregister_tool(parsed_message)
            else:
                self.logger.warning(
                    f"Received unexpected message type {type(parsed_message).__name__} on tool message topic {message.topic}. Ignoring."
                )
                self.messager.mqtt_log(
                    f"ToolManager: Received unexpected message type {type(parsed_message).__name__} on tool message topic {message.topic}. Ignoring.",
                    level="warning",
                )
        except (ValueError, ValidationError) as e:
            payload_preview = message.payload[:100].decode("utf-8", errors="replace")
            self.logger.error(
                f"Failed to parse/validate tool message: {e}. Payload: '{payload_preview}...'"
            )
            self.messager.mqtt_log(
                f"ToolManager: Failed to parse/validate tool message: {e}. Payload: '{payload_preview}...'",
                level="error",
            )
        except Exception as e:
            self.logger.error(f"Failed to handle tool message: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            self.messager.mqtt_log(
                f"ToolManager: Failed to handle tool message: {e}", level="error"
            )

    def _handle_result_message(self, message: MQTTMessage):
        """Handles incoming task result messages using Pydantic."""
        self.logger.debug(f"Received message on result topic: {message.topic}")
        self.messager.mqtt_log(
            f"ToolManager: Received message on result topic: {message.topic}", level="debug"
        )
        try:
            # Deserialize and validate using the helper
            parsed_message: AnyMessage = from_mqtt_message(message)

            if not isinstance(parsed_message, TaskResultMessage):
                self.logger.warning(
                    f"Received unexpected message type {type(parsed_message).__name__} on result topic {message.topic}. Ignoring."
                )
                self.messager.mqtt_log(
                    f"ToolManager: Received unexpected message type {type(parsed_message).__name__} on result topic {message.topic}. Ignoring.",
                    level="warning",
                )
                return

            result_msg = parsed_message  # Now we know it's TaskResultMessage
            task_id = result_msg.task_id

            with self._result_lock:
                if task_id in self._pending_tasks:
                    self.logger.info(
                        f"Received expected result for task {task_id} (Status: {result_msg.status})"
                    )
                    self.messager.mqtt_log(
                        f"ToolManager: Received expected result for task {task_id} (Status: {result_msg.status})"
                    )
                    self._task_results[task_id] = result_msg
                    event = self._pending_tasks.pop(task_id)
                    event.set()
                    self.logger.debug(f"Signaled completion for task {task_id}")
                    self.messager.mqtt_log(
                        f"ToolManager: Signaled completion for task {task_id}", level="debug"
                    )
                else:
                    self.logger.warning(
                        f"Received result for unknown or already completed task {task_id}. Caching."
                    )
                    self.messager.mqtt_log(
                        f"ToolManager: Received result for unknown or already completed task {task_id}. Caching.",
                        level="warning",
                    )
                    # Cache late result
                    self._late_results_cache[task_id] = (datetime.now().timestamp(), result_msg)
        except (ValueError, ValidationError) as e:  # Catch errors from from_mqtt_message
            payload_preview = message.payload[:100].decode("utf-8", errors="replace")
            self.logger.error(
                f"Failed to parse/validate result message payload: '{payload_preview}...'. Error: {e}"
            )
            self.messager.mqtt_log(
                f"ToolManager: Failed to parse/validate result message payload: '{payload_preview}...'. Error: {e}",
                level="error",
            )
        except Exception as e:
            self.logger.error(f"Unexpected error handling result message: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            self.messager.mqtt_log(
                f"ToolManager: Unexpected error handling result message: {e}", level="error"
            )

    def _handle_register_tool(self, reg_msg: RegisterToolMessage):
        """Handles tool registration messages."""
        self.logger.info(f"Registering tool '{reg_msg.tool_name}' from '{reg_msg.source}'")
        self.messager.mqtt_log(
            f"ToolManager: Registering tool '{reg_msg.tool_name}' from '{reg_msg.source}'",
            level="info",
        )

        # Generate a unique ID for this tool
        tool_id = str(uuid.uuid4())

        # Create a simple dictionary to store the tool information
        tool_info = {
            "tool_id": tool_id,
            "name": reg_msg.tool_name,
            "description": reg_msg.tool_manual,
            "api_schema": json.loads(reg_msg.tool_api),
            "source_service_id": reg_msg.source,
            # Define topics for communication with this tool
            "task_topic": f"tool/{tool_id}/task",
            "result_topic": f"tool/{tool_id}/result",
        }

        # Store the tool info in the tools dictionary
        self.tools[tool_id] = tool_info

        # Subscribe to the tool's result topic
        self.messager.subscribe(tool_info["result_topic"], self._handle_result_message)
        self._subscribed_result_topics.add(tool_info["result_topic"])

        self.logger.info(
            f"Registered tool '{tool_info['name']}' ({tool_id}). Will listen on {tool_info['result_topic']} for results."
        )
        self.messager.mqtt_log(
            f"ToolManager: Registered tool '{tool_info['name']}' ({tool_id}). Will listen on {tool_info['result_topic']} for results."
        )

        # Send back confirmation
        confirmation = ToolRegisteredMessage(
            source=self.messager.client_id,
            tool_id=tool_id,
            tool_name=reg_msg.tool_name,
            recipient=reg_msg.source,
        )

        # Use a default status topic instead of trying to get from config
        status_topic = "agent/status/info"

        # Try to access config if it exists, otherwise use the default
        try:
            if hasattr(self.messager, "config"):
                status_topic = (
                    self.messager.config.get("mqtt", {})
                    .get("publish_topics", {})
                    .get("status", status_topic)
                )
        except AttributeError:
            # Config doesn't exist, use the default
            pass

        self.messager.publish(status_topic, confirmation.model_dump_json())
        self.logger.info(
            f"Sent registration confirmation for '{reg_msg.tool_name}' to {reg_msg.source}"
        )
        self.messager.mqtt_log(
            f"ToolManager: Sent registration confirmation for '{reg_msg.tool_name}' to {reg_msg.source}",
            level="info",
        )

    def _handle_unregister_tool(self, unreg_msg):
        """Handles tool unregistration messages."""
        tool_id = unreg_msg.tool_id
        tool_name = unreg_msg.tool_name

        self.logger.info(
            f"Received unregistration request for tool '{tool_name}' (ID: {tool_id}) from '{unreg_msg.source}'"
        )
        self.messager.mqtt_log(
            f"ToolManager: Received unregistration request for tool '{tool_name}' (ID: {tool_id}) from '{unreg_msg.source}'",
            level="info",
        )

        # Check if the tool exists
        if tool_id not in self.tools:
            self.logger.warning(f"Cannot unregister unknown tool ID '{tool_id}'. Ignoring.")
            self.messager.mqtt_log(
                f"ToolManager: Cannot unregister unknown tool ID '{tool_id}'. Ignoring.",
                level="warning",
            )
            return

        # Get the tool info before removal
        tool_info = self.tools[tool_id]

        # Check if the source matches (security check)
        if tool_info.get("source_service_id") != unreg_msg.source:
            self.logger.warning(
                f"Unregistration request from '{unreg_msg.source}' doesn't match registered source '{tool_info.get('source_service_id')}'. Ignoring."
            )
            self.messager.mqtt_log(
                f"ToolManager: Unregistration request from '{unreg_msg.source}' doesn't match registered source '{tool_info.get('source_service_id')}'. Ignoring.",
                level="warning",
            )
            return

        # Unsubscribe from result topic if we're subscribed
        result_topic = tool_info.get("result_topic")
        if result_topic and result_topic in self._subscribed_result_topics:
            try:
                self.messager.unsubscribe(result_topic)
                self._subscribed_result_topics.remove(result_topic)
                self.logger.info(f"Unsubscribed from result topic: {result_topic}")
                self.messager.mqtt_log(
                    f"ToolManager: Unsubscribed from result topic: {result_topic}"
                )
            except Exception as e:
                self.logger.error(f"Error unsubscribing from result topic: {e}")
                self.messager.mqtt_log(
                    f"ToolManager: Error unsubscribing from result topic: {e}",
                    level="error",
                )

        # Remove the tool from our registry
        del self.tools[tool_id]
        self.logger.info(f"Successfully unregistered tool '{tool_name}' (ID: {tool_id})")
        self.messager.mqtt_log(
            f"ToolManager: Successfully unregistered tool '{tool_name}' (ID: {tool_id})",
            level="info",
        )

        # Send confirmation (optional - could use same message format as registration confirmation)
        status_message = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": self.messager.client_id,
            "recipient": unreg_msg.source,
            "type": "TOOL_UNREGISTERED",
            "tool_id": tool_id,
            "tool_name": tool_name,
            "status": "success",
        }

        # Use the same status topic as for registration confirmation
        status_topic = "agent/status/info"
        self.messager.publish(status_topic, json.dumps(status_message))
        self.logger.info(
            f"Sent unregistration confirmation for '{tool_name}' to {unreg_msg.source}"
        )
        self.messager.mqtt_log(
            f"ToolManager: Sent unregistration confirmation for '{tool_name}' to {unreg_msg.source}",
            level="info",
        )

    def register_tool(self, tool: BaseTool):
        """Registers a tool instance and subscribes to its result topic."""
        if tool.tool_id in self.tools:
            self.logger.warning(f"Tool ID '{tool.tool_id}' already registered. Overwriting.")
            self.messager.mqtt_log(
                f"Tool ID '{tool.tool_id}' already registered. Overwriting.", level="warning"
            )
        self.tools[tool.tool_id] = tool
        self.logger.info(f"Tool '{tool.name}' ({tool.tool_id}) registered with ToolManager")
        self.messager.mqtt_log(f"Tool '{tool.name}' ({tool.tool_id}) registered with ToolManager.")

        # Subscribe to the tool's result topic if not already subscribed
        if tool.result_topic not in self._subscribed_result_topics:
            self.messager.subscribe(tool.result_topic, self._handle_result_message)
            self._subscribed_result_topics.add(tool.result_topic)
            self.logger.info(f"Subscribed to result topic: {tool.result_topic}")
            self.messager.mqtt_log(f"ToolManager subscribed to result topic: {tool.result_topic}")
        self.logger.info(
            f"Registered tool '{tool.name}' ({tool.tool_id}). Will listen on {tool.result_topic} for results."
        )
        self.messager.mqtt_log(
            f"ToolManager: Registered tool '{tool.name}' ({tool.tool_id}). Will listen on {tool.result_topic} for results."
        )

    def initialize_tools(self):
        """
        Initializes all registered tools by ensuring subscriptions are active.
        With the new message-based approach, this simply ensures subscriptions
        are set up for all result topics.
        """
        if not self.messager.is_connected():
            self.logger.error("Cannot initialize tools: Messager not connected.")
            self.messager.mqtt_log(
                "Cannot initialize tools: Messager not connected.", level="error"
            )
            return

        self.logger.info("Initializing registered tools...")
        self.messager.mqtt_log("ToolManager: Initializing registered tools...")

        # Ensure we're subscribed to all the result topics
        for tool_id, tool_info in self.tools.items():
            try:
                # For tool dictionaries, ensure we've subscribed to their result topics
                if isinstance(tool_info, dict) and "result_topic" in tool_info:
                    result_topic = tool_info["result_topic"]
                    if result_topic not in self._subscribed_result_topics:
                        self.messager.subscribe(result_topic, self._handle_result_message)
                        self._subscribed_result_topics.add(result_topic)
                        self.logger.info(
                            f"Subscribed to result topic: {result_topic} for tool {tool_info['name']}"
                        )
                        self.messager.mqtt_log(
                            f"ToolManager: Subscribed to result topic: {result_topic} for tool {tool_info['name']}"
                        )
                # Handle any legacy tools that may still be BaseTool instances
                elif hasattr(tool_info, "initialize") and callable(
                    getattr(tool_info, "initialize")
                ):
                    tool_info.initialize()
                else:
                    self.logger.warning(
                        f"Unknown tool format for tool_id {tool_id}. Skipping initialization."
                    )
                    self.messager.mqtt_log(
                        f"ToolManager: Unknown tool format for tool_id {tool_id}. Skipping initialization.",
                        level="warning",
                    )
            except Exception as e:
                tool_name = (
                    tool_info.get("name", tool_id)
                    if isinstance(tool_info, dict)
                    else getattr(tool_info, "name", tool_id)
                )
                self.logger.error(f"Failed to initialize tool '{tool_name}': {e}")
                self.logger.error(traceback.format_exc())
                self.messager.mqtt_log(
                    f"Failed to initialize tool '{tool_name}': {e}", level="error"
                )

        self.logger.info("Tool initialization complete.")
        self.messager.mqtt_log("ToolManager: Tool initialization complete.")

    def execute_tool(
        self, tool_name: str, arguments: Dict[str, Any], timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Execute a tool with timeout.

        Args:
            tool_name (str): The name of the tool to execute.
            arguments (Dict[str, Any]): The arguments to pass to the tool.
            timeout (Optional[float], optional): The timeout in seconds. Defaults to None.

        Returns:
            Dict[str, Any]: The result of the tool execution.
        """
        # First check if we have a late result cached from a previous similar request
        cached_result = self._check_late_results_cache(tool_name, arguments)
        if cached_result is not None:
            self.logger.info(f"Using cached late result for {tool_name}")
            return cached_result.params

        # Find the tool
        tool = self._find_tool(tool_name)
        if not tool:
            err_msg = f"Tool {tool_name} not found"
            self.logger.error(err_msg)
            return {"error": err_msg}

        # Get the timeout from the tool instance or use the default timeout
        if timeout is None:
            # If the tool is a dictionary (from registration message)
            if isinstance(tool, dict) and "timeout" in tool:
                timeout = tool["timeout"]
            # If the tool is a BaseTool instance with a timeout attribute
            elif hasattr(tool, "timeout") and tool.timeout is not None:
                timeout = tool.timeout
            # Otherwise, use the default timeout
            else:
                timeout = self._default_timeout

        # Prepare task parameters
        task_id = str(uuid.uuid4())
        task_done = threading.Event()

        # Register the task in the pending tasks
        with self._result_lock:
            self._pending_tasks[task_id] = task_done
            self._task_results[task_id] = None

        # Log that we're executing the tool
        self.logger.info(
            f"Executing tool {tool_name} with args: {arguments} (task_id: {task_id}, timeout: {timeout}s)"
        )

        task_start_time = datetime.now().timestamp()

        try:
            # Create and send task message - do this BEFORE waiting
            if isinstance(tool, dict):
                # For tools registered via message, use the stored task topic
                task_topic = tool["task_topic"]
                tool_id = tool["tool_id"]
                task_message = TaskMessage(
                    source=self.messager.client_id,
                    tool_id=tool_id,
                    task_id=task_id,
                    payload=arguments,
                )
            else:
                # For directly registered tools (BaseTool instances)
                task_topic = tool.task_topic
                tool_id = tool.tool_id
                task_message = TaskMessage(
                    source=self.messager.client_id,
                    tool_id=tool_id,
                    task_id=task_id,
                    payload=arguments,
                )

            # Send the task message
            self.messager.publish(task_topic, task_message.model_dump_json())
            self.logger.info(f"Sent task message to {task_topic} (task_id: {task_id})")
            self.messager.mqtt_log(f"ToolManager: Sent task to {tool_name} (task_id: {task_id})")

            # NOW wait for the task to complete or timeout
            if task_done.wait(timeout=timeout):
                # Task completed within timeout
                execution_time = datetime.now().timestamp() - task_start_time
                self.logger.info(
                    f"Tool {tool_name} executed successfully in {execution_time:.2f}s (task_id: {task_id})"
                )

                # Get the result
                with self._result_lock:
                    result = self._task_results.get(task_id)
                    if result:
                        # Clean up
                        del self._task_results[task_id]
                        return result.params
                    else:
                        return {"error": "Tool completed but no result was stored"}
            else:
                # Task timed out
                self.logger.warning(
                    f"Tool {tool_name} timed out after {timeout}s (task_id: {task_id}). "
                    f"Will cache any late responses."
                )
                self.messager.mqtt_log(
                    f"ToolManager: ⚠️ Tool {tool_name} timed out after {timeout}s. "
                    f"Will cache any late responses."
                )

                # Clean up the pending task but keep listening for late results
                with self._result_lock:
                    if task_id in self._pending_tasks:
                        del self._pending_tasks[task_id]

                    # Set up to catch late results
                    self._pending_late_results[task_id] = {
                        "tool_name": tool_name,
                        "arguments": arguments,
                    }

                # Return a timeout error
                return {
                    "error": f"Tool execution timed out after {timeout}s",
                    "status": "timeout",
                    "task_id": task_id,
                }

        except Exception as e:
            self.logger.error(f"Error executing tool {tool_name}: {e}")
            self.logger.error(traceback.format_exc())
            self.messager.mqtt_log(
                f"ToolManager: Error executing tool {tool_name}: {e}", level="error"
            )

            # Clean up
            with self._result_lock:
                if task_id in self._pending_tasks:
                    del self._pending_tasks[task_id]

            return {"error": f"Error executing tool: {str(e)}"}

    def _execute_task(
        self, task_id: str, tool, tool_name: str, arguments: Dict[str, Any], on_result_callback
    ):
        """
        Execute a tool task by sending a message to the tool's task topic and registering for result.

        Args:
            task_id (str): The ID of the task
            tool: The tool to execute (can be dict or BaseTool)
            tool_name (str): Name of the tool
            arguments (Dict[str, Any]): Arguments to pass to the tool
            on_result_callback: Callback function to call when result is received
        """
        try:
            # Register the task in the pending tasks
            with self._result_lock:
                self._pending_tasks[task_id] = threading.Event()
                self._task_results[task_id] = None

            # Create a task message
            if isinstance(tool, dict):
                # For tools registered via message, use the stored task topic
                task_topic = tool["task_topic"]
                task_message = TaskMessage(
                    source=self.messager.client_id,
                    tool_id=tool["tool_id"],
                    task_id=task_id,
                    payload=arguments,
                )
            else:
                # For directly registered tools (BaseTool instances)
                task_topic = tool.task_topic
                task_message = TaskMessage(
                    source=self.messager.client_id,
                    tool_id=tool.tool_id,
                    task_id=task_id,
                    payload=arguments,
                )

            # Send the task message
            self.messager.publish(task_topic, task_message.model_dump_json())
            self.logger.info(f"Sent task message to {task_topic} (task_id: {task_id})")
            self.messager.mqtt_log(f"ToolManager: Sent task to {tool_name} (task_id: {task_id})")

        except Exception as e:
            self.logger.error(f"Error executing task: {e}")
            self.logger.error(traceback.format_exc())
            self.messager.mqtt_log(f"ToolManager: Error executing task: {e}", level="error")

            # Create an error result
            error_result = TaskResultMessage(
                source=self.messager.client_id,
                task_id=task_id,
                status=TaskStatus.ERROR,
                params={"error": str(e)},
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

            # Call the callback with the error
            on_result_callback(error_result)

    def _find_tool(self, tool_name: str) -> Optional[Union[Dict[str, Any], BaseTool]]:
        """
        Find a tool by name in the registered tools.

        Args:
            tool_name (str): The name of the tool to find

        Returns:
            Optional[Union[Dict[str, Any], BaseTool]]: The tool if found, None otherwise
        """
        # First try to find by exact tool_id match
        if tool_name in self.tools:
            return self.tools[tool_name]

        # Then try to find by name attribute
        for tool_id, tool in self.tools.items():
            if isinstance(tool, dict) and tool.get("name") == tool_name:
                return tool
            elif hasattr(tool, "name") and tool.name == tool_name:
                return tool

        # If still not found, try case-insensitive matching
        tool_name_lower = tool_name.lower()
        for tool_id, tool in self.tools.items():
            if isinstance(tool, dict) and tool.get("name", "").lower() == tool_name_lower:
                return tool
            elif hasattr(tool, "name") and tool.name.lower() == tool_name_lower:
                return tool

        return None

    def _check_late_results_cache(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[Any]:
        """
        Check if there's a late result available in the cache for a tool with similar arguments.

        Args:
            tool_name (str): Name of the tool to check
            arguments (Dict[str, Any]): Arguments for the tool

        Returns:
            Optional[Any]: The cached result if available, None otherwise
        """
        current_time = time.time()
        matched_key = None
        matched_result = None

        # Look for a matching late result
        for cache_key, (cached_time, cached_result) in list(self._late_results_cache.items()):
            # Check if cache entry is still valid (not too old)
            if current_time - cached_time > 300:  # 5 minutes expiration
                del self._late_results_cache[cache_key]
                continue

            cached_tool_name, cached_args_str = cache_key

            # Check if tool names match
            if cached_tool_name != tool_name:
                continue

            # Parse the cached arguments string back to a dictionary
            try:
                cached_args = json.loads(cached_args_str)
            except json.JSONDecodeError:
                continue

            # Check if arguments match sufficiently
            if self._arguments_match(cached_args, arguments):
                matched_key = cache_key
                matched_result = cached_result
                break

        # Remove the entry from cache if found
        if matched_key:
            self.log.info(f"Found cached late result for {tool_name}")
            del self._late_results_cache[matched_key]
            return matched_result

        return None

    def _arguments_match(self, cached_args: Dict[str, Any], current_args: Dict[str, Any]) -> bool:
        """
        Check if two sets of arguments match enough to use a cached result.

        Args:
            cached_args: The cached arguments
            current_args: The current arguments

        Returns:
            bool: True if arguments match sufficiently, False otherwise
        """
        # Simple implementation - check if the main action and query keys match
        if "action" in cached_args and "action" in current_args:
            if cached_args["action"] != current_args["action"]:
                return False

        if "query" in cached_args and "query" in current_args:
            cached_query = cached_args["query"].lower()
            current_query = current_args["query"].lower()

            # If the queries are similar enough
            if (
                cached_query == current_query
                or cached_query in current_query
                or current_query in cached_query
            ):
                return True

        # For other types of arguments, require exact match
        return cached_args == current_args

    def _add_late_result_handler(self, task_id: str, tool_name: str, callback):
        """
        Register a handler for late results for a particular task.

        Args:
            task_id (str): The ID of the task
            tool_name (str): Name of the tool
            callback: Function to call when a late result arrives
        """
        with self._result_lock:
            cache_key = (tool_name, json.dumps(task_id))
            self._pending_late_results[task_id] = {
                "tool_name": tool_name,
                "callback": callback,
                "timestamp": datetime.now().timestamp(),
            }

    def _clean_expired_cache_entries(self):
        """Remove expired entries from the late results cache."""
        current_time = datetime.now().timestamp()
        expired_keys = []

        for task_id, (timestamp, _) in self._late_results_cache.items():
            if current_time - timestamp > self._late_results_ttl:
                expired_keys.append(task_id)

        for task_id in expired_keys:
            self.logger.debug(f"Removing expired late result from cache: {task_id}")
            self._late_results_cache.pop(task_id, None)

    def generate_tool_descriptions_for_prompt(self) -> str:
        """Generates a JSON string describing available tools for the LLM prompt."""
        if not self.tools:
            return "[]"  # Return empty JSON array if no tools

        # Generate tool definitions in the format expected by LLMs
        tool_defs = []
        for tool_id, tool_info in self.tools.items():
            # Create a function definition structure using the schema from registration
            try:
                tool_def = {
                    "type": "function",
                    "function": {
                        "name": tool_info["name"].replace(" ", "_"),  # Ensure valid function name
                        "description": tool_info["description"],
                        "parameters": tool_info["api_schema"],  # Use the schema from registration
                        "tool_id": tool_id,  # Include the tool_id for mapping in execute_tool
                    },
                }
                tool_defs.append(tool_def)
            except (KeyError, TypeError) as e:
                self.logger.error(f"Error generating definition for tool {tool_id}: {e}")
                self.messager.mqtt_log(
                    f"ToolManager: Error generating definition for tool {tool_id}: {e}",
                    level="error",
                )

        # Return JSON string with all tool definitions
        return json.dumps(tool_defs, indent=2)
