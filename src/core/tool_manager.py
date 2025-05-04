import json
import threading
import uuid
import traceback
from typing import Dict, Any, Optional, Union
from datetime import datetime, timezone

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
        self._result_lock = threading.Lock()
        self._subscribed_result_topics = set()

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
                        f"Received result for unknown or already completed task {task_id}. Ignoring."
                    )
                    self.messager.mqtt_log(
                        f"ToolManager: Received result for unknown or already completed task {task_id}. Ignoring.",
                        level="warning",
                    )
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

    def execute_tool(self, tool_id: str, arguments: Dict[str, Any], timeout: float = 30.0) -> Any:
        """Executes a tool by sending a TaskMessage and waiting for TaskResultMessage."""
        if tool_id not in self.tools:
            self.logger.error(f"Attempted to execute unknown tool_id '{tool_id}'")
            self.messager.mqtt_log(
                f"ToolManager: Attempted to execute unknown tool_id '{tool_id}'", level="error"
            )
            raise ValueError(f"Tool '{tool_id}' not found.")

        tool_info = self.tools[tool_id]
        task_id = str(uuid.uuid4())  # Generate unique task ID

        # Create TaskMessage using Pydantic model
        task_message = TaskMessage(
            task_id=task_id,
            tool_id=tool_id,
            arguments=arguments,
            source=self.messager.client_id,  # Identify the sender
            recipient=tool_info.get("source_service_id"),  # Send to the tool's service
        )

        event = threading.Event()
        with self._result_lock:
            self._pending_tasks[task_id] = event
            if task_id in self._task_results:
                del self._task_results[task_id]

        try:
            task_topic = tool_info["task_topic"]
            self.logger.info(
                f"Publishing task {task_id} for tool '{tool_info['name']}' to topic {task_topic}"
            )
            self.messager.mqtt_log(
                f"ToolManager: Publishing task {task_id} for tool '{tool_info['name']}' to topic {task_topic}"
            )
            # Publish using Pydantic's JSON serialization
            self.messager.publish(task_topic, task_message.model_dump_json())
            self.logger.debug(f"Waiting for result for task {task_id} (timeout: {timeout}s)")
            self.messager.mqtt_log(
                f"ToolManager: Waiting for result for task {task_id} (timeout: {timeout}s)",
                level="debug",
            )

            # Wait for the result
            if event.wait(timeout=timeout):
                with self._result_lock:
                    result_msg = self._task_results.pop(task_id)  # Get TaskResultMessage

                # Process TaskResultMessage
                if result_msg.status == TaskStatus.SUCCESS:
                    self.logger.info(f"Task {task_id} completed successfully.")
                    self.messager.mqtt_log(f"ToolManager: Task {task_id} completed successfully.")
                    return result_msg.result  # Return the actual result field
                else:
                    error_details = result_msg.error or "No error details provided."
                    self.logger.error(
                        f"Task {task_id} failed on tool side (Tool: {result_msg.tool_id}): {error_details}"
                    )
                    self.messager.mqtt_log(
                        f"ToolManager: Task {task_id} failed on tool side (Tool: {result_msg.tool_id}): {error_details}",
                        level="error",
                    )
                    # Return the error string from the result message
                    return f"Error executing tool '{tool_info['name']}': {error_details}"
            else:
                # Timeout occurred
                self.logger.error(
                    f"Task {task_id} timed out after {timeout}s waiting for result from tool '{tool_info['name']}'."
                )
                self.messager.mqtt_log(
                    f"ToolManager: Task {task_id} timed out after {timeout}s waiting for result from tool '{tool_info['name']}'.",
                    level="error",
                )
                with self._result_lock:
                    # Clean up pending task if timeout occurred
                    if task_id in self._pending_tasks:
                        del self._pending_tasks[task_id]
                return (
                    f"Error: Tool '{tool_info['name']}' did not respond within the timeout period."
                )

        except Exception as e:
            # Error during publishing or waiting
            self.logger.error(f"Error during task {task_id} execution/waiting phase: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            self.messager.mqtt_log(
                f"ToolManager: Error during task {task_id} execution/waiting phase: {e}",
                level="error",
            )
            # Clean up pending task in case of exception
            with self._result_lock:
                if task_id in self._pending_tasks:
                    del self._pending_tasks[task_id]
                if task_id in self._task_results:
                    del self._task_results[task_id]
            return f"Error communicating with or waiting for tool '{tool_info['name']}': {e}"

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
