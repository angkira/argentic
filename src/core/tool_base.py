import json
import threading
import time
from typing import Dict, Any, Type, Callable, Optional
from abc import ABC, abstractmethod
import uuid

from pydantic import BaseModel, ValidationError

# Assuming Messager and task_protocol are in core
from .messager import Messager, MQTTMessage
from .protocol.task_protocol import TaskMessage, TaskResultMessage, TaskStatus


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
        self.task_topic = f"tools/{self.tool_id}/task"
        self.result_topic = f"tools/{self.tool_id}/result"
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

    def _handle_task_message(self, message: MQTTMessage):
        """Handles incoming task messages from MQTT."""
        self.messager.log(
            f"Tool '{self.name}' ({self.tool_id}): Received task message on {message.topic}",
            level="debug",
        )
        task: Optional[TaskMessage] = None  # Keep track for logging/error reporting
        try:
            # 1. Deserialize payload
            payload_dict = json.loads(message.payload)
            task = TaskMessage(**payload_dict)
            self.messager.log(f"Tool '{self.name}': Parsed Task {task.task_id}", level="debug")

            # 2. Validate tool_id (should match)
            if task.tool_id != self.tool_id:
                self.messager.log(
                    f"Tool '{self.name}': Received task {task.task_id} for wrong tool_id ({task.tool_id}). Ignoring.",
                    level="warning",
                )
                return

            # 3. Validate arguments against schema
            try:
                validated_args = self.argument_schema(**task.arguments)
                self.messager.log(
                    f"Tool '{self.name}': Task {task.task_id} arguments validated successfully.",
                    level="debug",
                )
            except ValidationError as e:
                error_msg = f"Argument validation failed: {e}"
                self.messager.log(
                    f"Tool '{self.name}': Task {task.task_id} {error_msg}", level="error"
                )
                result_message = TaskResultMessage(
                    task_id=task.task_id,
                    tool_id=self.tool_id,
                    status=TaskStatus.ERROR,
                    error=error_msg,
                )
                self.messager.publish(self.result_topic, result_message.model_dump_json())
                self.messager.log(
                    f"Tool '{self.name}': Published error result for task {task.task_id} to {self.result_topic}",
                    level="debug",
                )
                return

            # 4. Execute the tool's core logic
            self.messager.log(
                f"Tool '{self.name}': Executing task {task.task_id} with args: {validated_args.model_dump()}"
            )
            try:
                execution_result = self._execute(**validated_args.model_dump())
                result_message = TaskResultMessage(
                    task_id=task.task_id,
                    tool_id=self.tool_id,
                    status=TaskStatus.SUCCESS,
                    result=execution_result,
                )
                self.messager.log(
                    f"Tool '{self.name}': Task {task.task_id} completed successfully."
                )
            except Exception as exec_e:
                error_msg = f"Execution failed: {exec_e}"
                self.messager.log(
                    f"Tool '{self.name}': Task {task.task_id} {error_msg}", level="error"
                )
                # Optionally log traceback
                # import traceback
                # self.messager.log(traceback.format_exc(), level="error")
                result_message = TaskResultMessage(
                    task_id=task.task_id,
                    tool_id=self.tool_id,
                    status=TaskStatus.ERROR,
                    error=error_msg,
                )

            # 5. Publish the result
            self.messager.publish(self.result_topic, result_message.model_dump_json())
            self.messager.log(
                f"Tool '{self.name}': Published result for task {task.task_id} (Status: {result_message.status}) to {self.result_topic}",
                level="debug",
            )

        except json.JSONDecodeError:
            self.messager.log(
                f"Tool '{self.name}': Failed to decode JSON payload on {message.topic}: {message.payload}",
                level="error",
            )
            # Cannot send error result as task_id is unknown
        except ValidationError as e:
            # Error parsing the TaskMessage itself
            self.messager.log(
                f"Tool '{self.name}': Failed to parse TaskMessage structure on {message.topic}: {e}. Payload: {message.payload}",
                level="error",
            )
            # Cannot send error result as task_id is unknown
        except Exception as e:
            task_id_str = f"task {task.task_id}" if task else "unknown task"
            self.messager.log(
                f"Tool '{self.name}': Unexpected error handling {task_id_str} on {message.topic}: {e}",
                level="error",
            )
            # Optionally log traceback
            # import traceback
            # self.messager.log(traceback.format_exc(), level="error")
            # Cannot reliably send error result if task parsing failed

    @abstractmethod
    def _execute(self, **kwargs) -> Any:
        """The core logic of the tool. Must be implemented by subclasses."""
        pass

    def get_definition_for_prompt(self) -> Dict[str, Any]:
        """Generates the tool definition structure expected by the LLM prompt."""
        # Generate a JSON schema for the arguments
        schema = self.argument_schema.model_json_schema()
        return {
            "tool_id": self.tool_id,
            "name": self.name,
            "description": self.description,
            "parameters": schema,
        }
