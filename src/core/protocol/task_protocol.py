from enum import Enum
from typing import Dict, Any, Optional
import uuid

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    SUCCESS = "success"
    ERROR = "error"


class BaseTaskMessage(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_id: str


class TaskMessage(BaseTaskMessage):
    arguments: Dict[str, Any]


class TaskResultMessage(BaseTaskMessage):
    status: TaskStatus
    result: Optional[Any] = None
    error: Optional[str] = None
