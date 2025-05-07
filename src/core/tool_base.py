import json
import threading
import time
from typing import Dict, Any, Type, Callable, Optional
from abc import ABC, abstractmethod
import uuid

from pydantic import BaseModel, ValidationError

# Modified import - import aiomqtt.Message directly
import aiomqtt
from .messager import Messager

# Import specific message types and helper
from .protocol.message import (
    TaskMessage,
    TaskResultMessage,
    TaskStatus,
    from_mqtt_message,
    AnyMessage,  # Import the Union type
)


class BaseTool(ABC):
    """Abstract base class for tools that communicate via MQTT."""

    def __init__(
        self,
        tool_id: str,
        name: str,
        description: str,
        argument_schema: Type[BaseModel],
        messager: Messager,
    ):
        self.tool_id = tool_id
        self.name = name
        self.description = description
        self.argument_schema = argument_schema
        self.messager = messager
        # Use the same topic format as ToolManager (tool/ instead of tools/)
        self.task_topic = f"tool/{self.tool_id}/task"
        self.result_topic = f"tool/{self.tool_id}/result"
        self._initialized = False

    def initialize(self):
        """Subscribes the tool to its task topic."""
        if not self._initialized:
            self.messager.subscribe(self.task_topic, self._handle_task_message)
            self.messager.log(
                f"Tool '{self.name}' ({self.tool_id}): Initialized and listening on task topic '{self.task_topic}'"
            )
            self._initialized = True
        else:
            self.messager.log(
                f"Tool '{self.name}' ({self.tool_id}): Already initialized.", level="warning"
            )

    def _handle_task_message(self, message: aiomqtt.Message):
        """Handles incoming task messages from MQTT using Pydantic."""
        self.messager.log(
            f"Tool '{self.name}' ({self.tool_id}): Received task message on {message.topic}",
            level="debug",
        )
        task: Optional[TaskMessage] = None
        result_message: Optional[TaskResultMessage] = None

        try:
            # 1. Deserialize and Validate Payload using the helper
            # This automatically selects the correct Pydantic model (should be TaskMessage)
            parsed_message: AnyMessage = from_mqtt_message(message)

            # 2. Check if it's the correct message type
            if not isinstance(parsed_message, TaskMessage):
                self.messager.log(
                    f"Tool '{self.name}': Received unexpected message type {type(parsed_message).__name__} on task topic {message.topic}. Ignoring.",
                    level="warning",
                )
                return

            task = parsed_message  # Now we know it's a TaskMessage
            self.messager.log(f"Tool '{self.name}': Parsed Task {task.task_id}", level="debug")

            # 3. Validate tool_id
            if task.tool_id != self.tool_id:
                error_msg = f"Mismatched tool_id. Expected '{self.tool_id}', got '{task.tool_id}'."
                self.messager.log(f"Tool '{self.name}': {error_msg}", level="error")
                result_message = TaskResultMessage(
                    task_id=task.task_id,
                    tool_id=self.tool_id,  # Report as originating from this tool instance
                    status=TaskStatus.ERROR,
                    error=error_msg,
                    source=self.messager.client_id,  # Identify the source of the result
                )
                # Skip to publishing the error result
            else:
                # 4. Validate arguments against schema
                try:
                    validated_args = self.argument_schema.model_validate(task.arguments)
                    self.messager.log(
                        f"Tool '{self.name}': Validated arguments for task {task.task_id}",
                        level="debug",
                    )

                    # 5. Execute the tool's core logic
                    self.messager.log(
                        f"Tool '{self.name}': Executing task {task.task_id} with args: {validated_args.model_dump()}"
                    )
                    try:
                        tool_output = self._execute(**validated_args.model_dump())
                        result_message = TaskResultMessage(
                            task_id=task.task_id,
                            tool_id=self.tool_id,
                            status=TaskStatus.SUCCESS,
                            result=tool_output,
                            source=self.messager.client_id,
                        )
                        self.messager.log(
                            f"Tool '{self.name}': Task {task.task_id} executed successfully.",
                            level="debug",
                        )
                    except Exception as exec_e:
                        error_msg = f"Execution failed for task {task.task_id}: {exec_e}"
                        self.messager.log(f"Tool '{self.name}': {error_msg}", level="error")
                        # Optionally log traceback here
                        result_message = TaskResultMessage(
                            task_id=task.task_id,
                            tool_id=self.tool_id,
                            status=TaskStatus.ERROR,
                            error=error_msg,
                            source=self.messager.client_id,
                        )

                except ValidationError as e:
                    error_msg = f"Argument validation failed for task {task.task_id}: {e}"
                    self.messager.log(f"Tool '{self.name}': {error_msg}", level="error")
                    result_message = TaskResultMessage(
                        task_id=task.task_id,
                        tool_id=self.tool_id,
                        status=TaskStatus.ERROR,
                        error=f"Invalid arguments: {e}",
                        source=self.messager.client_id,
                    )

            # 6. Publish the result (if one was generated)
            if result_message:
                self.messager.publish(self.result_topic, result_message.model_dump_json())
                self.messager.log(
                    f"Tool '{self.name}': Published result for task {task.task_id} (Status: {result_message.status}) to {self.result_topic}",
                    level="debug",
                )

        except (
            ValueError,
            ValidationError,
        ) as e:  # Catch errors from from_mqtt_message or Pydantic validation
            # Error parsing the TaskMessage structure itself or invalid format
            payload_preview = message.payload[:100].decode("utf-8", errors="replace")
            self.messager.log(
                f"Tool '{self.name}': Failed to parse/validate incoming message on {message.topic}: {e}. Payload preview: '{payload_preview}...'",
                level="error",
            )
            # Cannot send error result as task_id might be unknown/invalid
        except Exception as e:
            task_id_str = f"task {task.task_id}" if task else "unknown task"
            self.messager.log(
                f"Tool '{self.name}': Unexpected error handling {task_id_str} on {message.topic}: {e}",
                level="error",
            )
            # Optionally log traceback
            # import traceback
            # self.messager.log(traceback.format_exc(), level="error")
            # Cannot reliably send error result if task parsing failed or task_id is missing

    @abstractmethod
    def _execute(self, **kwargs) -> Any:
        """The core logic of the tool. Must be implemented by subclasses."""
        pass

    def get_definition_for_prompt(self) -> Dict[str, Any]:
        """Generates the tool definition structure expected by the LLM prompt."""
        # Generate a JSON schema for the arguments
        schema = self.argument_schema.model_json_schema()
        # Ensure required fields are marked correctly if not automatically handled by Pydantic schema generation
        # (Pydantic usually handles this well based on Optional/default values)
        return {
            "type": "function",  # Standard type for function calling
            "function": {
                "name": self.name.replace(" ", "_"),  # Ensure name is valid identifier
                "description": self.description,
                "parameters": schema,
                "tool_id": self.tool_id,  # Include tool_id for mapping back
            },
        }
