import logging
import traceback
from typing import Coroutine, Dict, Any, Type, Optional, Union
from abc import ABC, abstractmethod

from pydantic import BaseModel, ValidationError

from argentic.core.protocol.task import TaskErrorMessage
from argentic.core.protocol.tool import (
    RegisterToolMessage,
    ToolRegisteredMessage,
    ToolRegistrationErrorMessage,
    UnregisterToolMessage,
)
from ..messager.messager import Messager

from argentic.core.protocol.task import TaskMessage, TaskResultMessage, TaskStatus


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

    async def _initialize(self):
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

    async def register(
        self,
        registration_topic: str,
        status_topic: str,
        call_topic_base: str,
        response_topic_base: str,
    ):
        """Publishes RegisterToolMessage and listens on status_topic for confirmation."""
        # store topic bases
        self.registration_topic = registration_topic
        self.call_topic_base = call_topic_base
        self.response_topic_base = response_topic_base

        async def registration_handler(
            message: Union[ToolRegisteredMessage, ToolRegistrationErrorMessage],
        ):
            # assign tool ID and set task/result topics
            self.id = message.tool_id
            self.task_topic = f"{self.call_topic_base}/{self.id}"
            self.result_topic = f"{self.response_topic_base}/{self.id}"

            if isinstance(message, ToolRegistrationErrorMessage):
                await self.messager.log(
                    f"Tool '{self.name}': Registration error: {message.error}",
                    level="error",
                )
                return

            # on successful registration
            self.logger.info(f"Tool '{self.name}': Registration confirmed. Tool ID: {self.id}")
            # subscribe to incoming task messages now that id is set
            await self._initialize()

        # subscribe to confirmation messages on status_topic before publishing
        await self.messager.subscribe(
            status_topic,
            registration_handler,
            message_cls=ToolRegisteredMessage,
        )

        self.logger.info(
            f"Tool '{self.name}': Subscribed to status topic '{status_topic}' for registration confirmation."
        )

        self.logger.info(
            f"Tool '{self.name}': Publishing registration message to topic '{registration_topic}'."
        )
        # send registration request after subscriptions in place
        registration_message = RegisterToolMessage(
            source=self.messager.client_id,
            tool_name=self.name,
            tool_manual=self.manual,
            tool_api=self.api,
        )
        await self.messager.publish(registration_topic, registration_message)

    async def unregister(self):
        if self.id is None:
            await self.messager.log(
                f"Tool '{self.name}': Cannot unregister. Tool ID is not set.",
                level="warning",
            )

            return

        unregister_message = UnregisterToolMessage(
            tool_id=self.id,
        )

        await self.messager.publish(
            self.registration_topic,
            unregister_message,
        )

    async def _handle_task_message(self, task: TaskMessage):
        """Handles incoming task messages from MQTT using Pydantic."""
        self.logger.info(
            f"Tool '{self.name}' ({self.id}): Received task message for task_id: {task.task_id}",
        )
        result_message_to_publish: Optional[Union[TaskResultMessage, TaskErrorMessage]] = None

        try:
            # verify message is for this tool
            if task.tool_id != self.id:
                error_msg = f"Mismatched tool id. Expected '{self.id}', got '{task.tool_id}'."
                self.logger.error(f"Tool '{self.name}': {error_msg}")
                result_message_to_publish = TaskErrorMessage(
                    task_id=task.task_id,
                    tool_id=self.id,
                    tool_name=self.name,
                    arguments=task.arguments,
                    status=TaskStatus.FAILED,
                    error=error_msg,
                    source=self.messager.client_id,
                )
            else:
                # Validate arguments against schema using task.arguments
                try:
                    if task.arguments is None:
                        raise ValueError("Task arguments are missing (None).")

                    validated_args = self.argument_schema.model_validate(task.arguments)
                    self.logger.debug(
                        f"Tool '{self.name}': Validated arguments for task {task.task_id}"
                    )

                    self.logger.info(
                        f"Tool '{self.name}': Executing task {task.task_id} with args: {validated_args.model_dump()}"
                    )
                    try:
                        tool_output = await self._execute(**validated_args.model_dump())

                        result_message_to_publish = TaskResultMessage(
                            task_id=task.task_id,
                            tool_id=self.id,
                            tool_name=self.name,
                            arguments=task.arguments,
                            status=TaskStatus.SUCCESS,
                            result=tool_output,
                            source=self.messager.client_id,
                        )
                        self.logger.debug(
                            f"Tool '{self.name}': Task {task.task_id} executed successfully."
                        )
                    except Exception as exec_e:
                        error_msg = f"Execution failed for task {task.task_id}: {exec_e}"
                        self.logger.error(f"Tool '{self.name}': {error_msg}", exc_info=True)
                        result_message_to_publish = TaskErrorMessage(
                            task_id=task.task_id,
                            tool_id=self.id,
                            tool_name=self.name,
                            arguments=task.arguments,
                            status=TaskStatus.FAILED,
                            error=error_msg,
                            traceback=(
                                traceback.format_exc()
                                if self.logger.level <= logging.DEBUG
                                else None
                            ),
                            source=self.messager.client_id,
                        )

                except ValidationError as e:
                    error_msg = f"Argument validation failed for task {task.task_id} using task.arguments: {e}"
                    self.logger.error(f"Tool '{self.name}': {error_msg}")
                    result_message_to_publish = TaskErrorMessage(
                        task_id=task.task_id,
                        tool_id=self.id,
                        tool_name=self.name,
                        arguments=task.arguments,
                        status=TaskStatus.FAILED,
                        error=f"Invalid arguments: {e}",
                        source=self.messager.client_id,
                    )
                except ValueError as ve:
                    error_msg = f"Argument error for task {task.task_id}: {ve}"
                    self.logger.error(f"Tool '{self.name}': {error_msg}")
                    result_message_to_publish = TaskErrorMessage(
                        task_id=task.task_id,
                        tool_id=self.id,
                        tool_name=self.name,
                        arguments=task.arguments,
                        status=TaskStatus.FAILED,
                        error=error_msg,
                        source=self.messager.client_id,
                    )

            if result_message_to_publish:
                if not hasattr(self, "result_topic") or not self.result_topic:
                    self.logger.error(
                        f"Tool '{self.name}': result_topic not set. Cannot publish result for task {task.task_id}."
                    )
                    return

                await self.messager.publish(
                    self.result_topic, result_message_to_publish.model_dump_json()
                )
                self.logger.info(
                    f"Tool '{self.name}': Published result for task {task.task_id} (Status: {result_message_to_publish.status}) to {self.result_topic}"
                )

        except Exception as e:
            task_id_str = task.task_id if task and hasattr(task, "task_id") else "unknown task"
            self.logger.error(
                f"Tool '{self.name}': Unexpected critical error in _handle_task_message for {task_id_str}: {e}",
                exc_info=True,
            )

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
