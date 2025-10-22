from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, model_validator
from typing_extensions import TypedDict

try:
    import numpy as np

    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False
    np = None

try:
    from PIL import Image

    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False
    Image = None


class ToolCallDict(TypedDict):
    name: str
    args: Dict[str, Union[str, int, float, bool, List, Dict]]
    id: Optional[str]


class ChatMessage(BaseModel):
    """Base class for chat messages."""

    role: str
    content: str

    @model_validator(mode="after")
    def _validate_role(self):
        allowed = {"system", "user", "assistant", "tool"}
        if self.role not in allowed:
            raise ValueError(f"Invalid role '{self.role}'. Must be one of {sorted(allowed)}")
        return self


class SystemMessage(ChatMessage):
    role: str = Field(default="system")

    @model_validator(mode="after")
    def check_role(self):
        if self.role != "system":
            raise ValueError("SystemMessage must have role='system'")
        return self


class MultimodalContent(BaseModel):
    """Container for multimodal content (text + images + audio)."""

    text: str
    images: Optional[List[Any]] = (
        None  # Can be base64 strings, np.ndarray, PIL.Image, or file paths
    )
    audio: Optional[Any] = None  # Can be base64 string, np.ndarray, or file path

    class Config:
        arbitrary_types_allowed = True  # Allow numpy arrays and PIL images


class UserMessage(ChatMessage):
    role: str = Field(default="user")
    content: Union[str, Dict[str, Any]] = Field(default="")  # Allow dict for multimodal content

    # Optional fields for multimodal content (backward compatibility)
    images: Optional[List[Any]] = None
    audio: Optional[Any] = None

    @model_validator(mode="after")
    def check_role(self):
        if self.role != "user":
            raise ValueError("UserMessage must have role='user'")
        return self

    class Config:
        arbitrary_types_allowed = True  # Allow numpy arrays and PIL images


class AssistantMessage(ChatMessage):
    role: str = Field(default="assistant")

    @model_validator(mode="after")
    def check_role(self):
        if self.role != "assistant":
            raise ValueError("AssistantMessage must have role='assistant'")
        return self

    tool_calls: Optional[List[ToolCallDict]] = None


class ToolMessage(ChatMessage):
    role: str = Field(default="tool")

    @model_validator(mode="after")
    def check_role(self):
        if self.role != "tool":
            raise ValueError("ToolMessage must have role='tool'")
        return self

    tool_call_id: Optional[str] = None
    content: str  # Tool execution result or error


class LLMChatResponse(BaseModel):
    """Unified response from LLM providers."""

    message: AssistantMessage
    usage: Optional[Dict[str, int]] = None  # Optional token usage info
    finish_reason: Optional[str] = None
