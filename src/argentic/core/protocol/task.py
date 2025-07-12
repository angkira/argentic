from enum import Enum
from typing import Any, Optional, Dict
import uuid
from argentic.core.protocol.message import BaseMessage
from pydantic import Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ERROR = "error"
    SUCCESS = "success"
    TIMEOUT = "timeout"


class TaskMessage(BaseMessage[None]):
    type: str = Field(default="TASK")
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_id: str
    tool_name: str
    arguments: Optional[Dict[str, Any]] = None  # This is where the LLM arguments go


class TaskResultMessage(TaskMessage):
    status: TaskStatus
    result: Optional[Any] = None
    error: Optional[str] = None


class TaskErrorMessage(TaskMessage):
    status: TaskStatus = TaskStatus.ERROR
    error: str
    traceback: Optional[str] = None
