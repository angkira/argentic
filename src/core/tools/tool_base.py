from typing import Coroutine, Dict, Any, Type, Optional, Union
from abc import ABC, abstractmethod

from pydantic import BaseModel, ValidationError

from core.protocol.task import TaskErrorMessage
from core.protocol.tool import (
    RegisterToolMessage,
    ToolRegisteredMessage,
    ToolRegistrationErrorMessage,
    UnregisterToolMessage,
)
from ..messager.messager import Messager

from core.protocol.task import TaskMessage, TaskResultMessage, TaskStatus


class BaseTool(ABC):
    id: str
    _initialized: bool = False
    manual: str
    api: str

    def __init__(
        self,
        name: str,
        manual: str,
        api: str,
        argument_schema: Type[BaseModel],
        messager: Messager,
    ):
        self.name = name
        self.argument_schema = argument_schema
        self.messager = messager
        self.manual = manual
        self.api = api
        # task_topic and result_topic will be set after registration (when self.id is available)

    async def initialize(self):
        """Subscribes the tool to its task topic."""
        if self._initialized:
            await self.messager.log(
                f"Tool '{self.name}' ({self.id}): Already initialized.", level="warning"
            )
            return

        await self.messager.subscribe(
            self.task_topic, self._handle_task_message, message_cls=TaskMessage
        )

        await self.messager.log(
            f"Tool '{self.name}' ({self.id}): Initialized and listening on task topic '{self.task_topic}'"
        )
        self._initialized = True

    async def register(self, registration_topic: str):
        registration_message = RegisterToolMessage(
            tool_name=self.name,
            tool_manual=self.manual,
            tool_api=self.api,
        )

        await self.messager.publish(
            registration_topic,
            registration_message,
        )

        async def registration_handler(
            message: Union[ToolRegisteredMessage, ToolRegistrationErrorMessage],
        ):
            # assign tool ID and set topics
            self.id = message.tool_id
            self.task_topic = f"tool/{self.id}/task"
            self.result_topic = f"tool/{self.id}/result"

            if isinstance(message, ToolRegistrationErrorMessage):
                await self.messager.log(
                    f"Tool '{self.name}': Registration error: {message.error}",
                    level="error",
                )
                return

            await self.messager.log(
                f"Tool '{self.name}': Registered successfully with ID: {self.id}",
                level="info",
            )

            # subscribe to task messages now that topics are available
            await self.initialize()

        # subscribe to both registration success and error responses
        await self.messager.subscribe(
            registration_topic,
            registration_handler,
            message_cls=ToolRegisteredMessage,
        )
        await self.messager.subscribe(
            registration_topic,
            registration_handler,
            message_cls=ToolRegistrationErrorMessage,
        )

    async def unregister(self, registration_topic: str):
        unregister_message = UnregisterToolMessage(
            tool_id=self.id,
        )

        await self.messager.publish(
            registration_topic,
            unregister_message,
        )

    async def _handle_task_message(self, task: TaskMessage):
        """Handles incoming task messages from MQTT using Pydantic."""
        await self.messager.log(
            f"Tool '{self.name}' ({self.id}): Received task message",
            level="debug",
        )
        result_message: Optional[TaskResultMessage] = None

        try:
            # verify message is for this tool
            if task.tool_id != self.id:
                error_msg = f"Mismatched tool id. Expected '{self.id}', got '{task.tool_id}'."

                await self.messager.log(f"Tool '{self.name}': {error_msg}", level="error")

                error_message = TaskErrorMessage(
                    task_id=task.task_id,
                    tool_id=self.id,
                    error=error_msg,
                    source=self.messager.client_id,
                )

                await self.messager.publish(
                    self.result_topic,
                    error_message,
                )
            else:
                # 4. Validate arguments against schema
                try:
                    validated_args = self.argument_schema.model_validate(task.data)
                    await self.messager.log(
                        f"Tool '{self.name}': Validated arguments for task {task.task_id}",
                        level="debug",
                    )

                    await self.messager.log(
                        f"Tool '{self.name}': Executing task {task.task_id} with args: {validated_args.model_dump()}"
                    )
                    try:
                        tool_output = await self._execute(**validated_args.model_dump())

                        result_message = TaskResultMessage(
                            task_id=task.task_id,
                            tool_id=self.id,
                            status=TaskStatus.SUCCESS,
                            result=tool_output,
                            source=self.messager.client_id,
                        )

                        await self.messager.log(
                            f"Tool '{self.name}': Task {task.task_id} executed successfully.",
                            level="debug",
                        )
                    except Exception as exec_e:
                        error_msg = f"Execution failed for task {task.task_id}: {exec_e}"

                        await self.messager.log(f"Tool '{self.name}': {error_msg}", level="error")
                        # Optionally log traceback here
                        result_message = TaskErrorMessage(
                            task_id=task.task_id,
                            tool_id=self.id,
                            error=error_msg,
                            source=self.messager.client_id,
                        )

                except ValidationError as e:
                    error_msg = f"Argument validation failed for task {task.task_id}: {e}"

                    await self.messager.log(f"Tool '{self.name}': {error_msg}", level="error")

                    result_message = TaskErrorMessage(
                        task_id=task.task_id,
                        tool_id=self.id,
                        error=f"Invalid arguments: {e}",
                        source=self.messager.client_id,
                    )

            if result_message:
                await self.messager.publish(self.result_topic, result_message)

                await self.messager.log(
                    f"Tool '{self.name}': Published result for task {task.task_id} (Status: {result_message.status}) to {self.result_topic}",
                    level="debug",
                )

        except (
            ValueError,
            ValidationError,
        ) as e:  # Catch errors from from_mqtt_message or Pydantic validation
            # Error parsing the TaskMessage structure itself or invalid format
            # Preview the raw data payload
            payload_preview = str(task.data)[:100]

            await self.messager.log(
                f"Tool '{self.name}': Failed to parse/validate incoming message: {e}. Payload preview: '{payload_preview}...'",
                level="error",
            )
            # Cannot send error result as task_id might be unknown/invalid
        except Exception as e:
            task_id_str = f"task {task.task_id}" if task else "unknown task"

            await self.messager.log(
                f"Tool '{self.name}': Unexpected error handling {task_id_str}: {e}",
                level="error",
            )
            # Optionally log traceback
            # import traceback
            # self.messager.log(traceback.format_exc(), level="error")
            # Cannot reliably send error result if task parsing failed or task_id is missing

    @abstractmethod
    async def _execute(self, **kwargs) -> Coroutine[Any, Any, Any]:
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
                "description": self.manual,
                "parameters": schema,
                "id": self.id,  # Include id for mapping back
            },
        }
