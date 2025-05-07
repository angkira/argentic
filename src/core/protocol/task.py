from enum import Enum
from typing import Any, Optional
import uuid
from core.protocol.message import BaseMessage, MessageType
from pydantic import Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    SUCCESS = "success"
    ERROR = "error"


class TaskMessage(BaseMessage):
    type: MessageType = MessageType.TASK
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_id: str


class TaskResultMessage(TaskMessage):
    status: TaskStatus
    result: Optional[Any] = None


class TaskErrorMessage(TaskMessage):
    status: TaskStatus = TaskStatus.ERROR
    error: str
    traceback: Optional[str] = None
