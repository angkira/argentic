import json
import threading
import time
import uuid
from typing import Dict, Any, Type, Callable, Optional, List

from pydantic import BaseModel, ValidationError

from .messager import Messager, MQTTMessage
from .protocol.task_protocol import TaskMessage, TaskResultMessage, TaskStatus
from .tool_base import BaseTool


class ToolManager:
    """Manages tool registration, execution via MQTT, and description generation."""

    def __init__(self, messager: Messager):
        self.messager = messager
        self.tools: Dict[str, BaseTool] = {}
        self._pending_tasks: Dict[str, threading.Event] = {}
        self._task_results: Dict[str, TaskResultMessage] = {}
        self._result_lock = threading.Lock()
        self._subscribed_result_topics = set()
        self.messager.log("ToolManager initialized.")

    def _handle_result_message(self, message: MQTTMessage):
        """Handles incoming task result messages from MQTT."""
        self.messager.log(
            f"ToolManager: Received message on result topic: {message.topic}", level="debug"
        )
        try:
            payload_dict = json.loads(message.payload)
            result_msg = TaskResultMessage(**payload_dict)
            task_id = result_msg.task_id

            with self._result_lock:
                if task_id in self._pending_tasks:
                    self.messager.log(
                        f"ToolManager: Received expected result for task {task_id} (Status: {result_msg.status})"
                    )
                    self._task_results[task_id] = result_msg
                    event = self._pending_tasks.pop(task_id)  # Remove before setting event
                    event.set()  # Signal that the result is ready
                    self.messager.log(
                        f"ToolManager: Signaled completion for task {task_id}", level="debug"
                    )
                else:
                    self.messager.log(
                        f"ToolManager: Received result for unknown or already completed task {task_id}. Ignoring.",
                        level="warning",
                    )
        except (json.JSONDecodeError, ValidationError) as e:
            self.messager.log(
                f"ToolManager: Failed to parse result message payload: {message.payload}. Error: {e}",
                level="error",
            )
        except Exception as e:
            self.messager.log(
                f"ToolManager: Unexpected error handling result message: {e}", level="error"
            )

    def register_tool(self, tool: BaseTool):
        """Registers a tool instance and subscribes to its result topic."""
        if tool.tool_id in self.tools:
            self.messager.log(
                f"Tool ID '{tool.tool_id}' already registered. Overwriting.", level="warning"
            )
        self.tools[tool.tool_id] = tool
        self.messager.log(f"Tool '{tool.name}' ({tool.tool_id}) registered with ToolManager.")

        # Subscribe to the tool's result topic if not already subscribed
        if tool.result_topic not in self._subscribed_result_topics:
            self.messager.subscribe(tool.result_topic, self._handle_result_message)
            self._subscribed_result_topics.add(tool.result_topic)
            self.messager.log(f"ToolManager subscribed to result topic: {tool.result_topic}")
        self.messager.log(
            f"ToolManager: Registered tool '{tool.name}' ({tool.tool_id}). Will listen on {tool.result_topic} for results."
        )

    def initialize_tools(self):
        """Initializes all registered tools (subscribes them to task topics)."""
        if not self.messager.is_connected():
            self.messager.log("Cannot initialize tools: Messager not connected.", level="error")
            return
        self.messager.log("ToolManager: Initializing registered tools...")
        for tool_id, tool in self.tools.items():
            try:
                tool.initialize()
            except Exception as e:
                self.messager.log(f"Failed to initialize tool '{tool.name}': {e}", level="error")
        self.messager.log("ToolManager: Tool initialization complete.")

    def execute_tool(self, tool_id: str, arguments: Dict[str, Any], timeout: float = 30.0) -> Any:
        """Executes a tool by sending a task message and waiting for the result via MQTT."""
        if tool_id not in self.tools:
            self.messager.log(
                f"ToolManager: Attempted to execute unknown tool_id '{tool_id}'", level="error"
            )
            raise ValueError(f"Tool '{tool_id}' not found.")

        tool = self.tools[tool_id]
        task_id = str(uuid.uuid4())
        task_message = TaskMessage(task_id=task_id, tool_id=tool_id, arguments=arguments)

        event = threading.Event()
        with self._result_lock:
            self._pending_tasks[task_id] = event
            # Clear any potential stale result for this task_id (highly unlikely)
            if task_id in self._task_results:
                del self._task_results[task_id]

        try:
            self.messager.log(
                f"ToolManager: Publishing task {task_id} for tool '{tool_id}' to topic {tool.task_topic}"
            )
            self.messager.publish(tool.task_topic, task_message.model_dump_json())
            self.messager.log(
                f"ToolManager: Waiting for result for task {task_id} (timeout: {timeout}s)",
                level="debug",
            )

            # Wait for the result
            if event.wait(timeout=timeout):
                # Result received
                with self._result_lock:
                    result_msg = self._task_results.pop(task_id)  # Get and remove result

                if result_msg.status == TaskStatus.SUCCESS:
                    self.messager.log(f"ToolManager: Task {task_id} completed successfully.")
                    return result_msg.result
                else:
                    # Task failed on the tool side
                    error_details = result_msg.error or "No error details provided."
                    self.messager.log(
                        f"ToolManager: Task {task_id} failed on tool side: {error_details}",
                        level="error",
                    )
                    return f"Error executing tool '{tool_id}': {error_details}"
            else:
                # Timeout occurred
                self.messager.log(
                    f"ToolManager: Task {task_id} timed out after {timeout}s waiting for result from tool '{tool_id}'.",
                    level="error",
                )
                with self._result_lock:
                    # Clean up pending task if timeout occurred
                    if task_id in self._pending_tasks:
                        del self._pending_tasks[task_id]
                return f"Error: Tool '{tool_id}' did not respond within the timeout period."

        except Exception as e:
            # Error during publishing or waiting (not timeout or tool-side error)
            self.messager.log(
                f"ToolManager: Error during task {task_id} execution/waiting phase: {e}",
                level="error",
            )
            # Clean up pending task in case of exception
            with self._result_lock:
                if task_id in self._pending_tasks:
                    del self._pending_tasks[task_id]
                if task_id in self._task_results:
                    del self._task_results[task_id]
            return f"Error communicating with or waiting for tool '{tool_id}': {e}"

    def generate_tool_descriptions_for_prompt(self) -> str:
        """Generates a JSON string describing available tools for the LLM prompt."""
        if not self.tools:
            return "No tools available."

        tool_defs = [tool.get_definition_for_prompt() for tool in self.tools.values()]
        return json.dumps(tool_defs, indent=2)
