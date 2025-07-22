from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Literal, Optional, TypeVar, Generic, List, Union
from pydantic import BaseModel, Field
import uuid
import time
import json
from uuid import uuid4

from .enums import MessageSource


T = TypeVar("T")


class BaseMessage(BaseModel, Generic[T]):
    type: str
    source: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)
    data: Optional[T] = None
    message_id: str = Field(default_factory=lambda: str(uuid4()))

    def model_dump_json(self, **kwargs: Any) -> str:
        # Pydantic's model_dump_json is suitable for this
        return super().model_dump_json(**kwargs)

    @classmethod
    def model_validate_json(cls, json_data: Union[str, bytes, bytearray], **kwargs: Any):
        # Pydantic's model_validate_json is suitable for this
        return super().model_validate_json(json_data, **kwargs)


class SystemMessage(BaseMessage[Dict[str, Any]]):
    type: Literal["SYSTEM"] = "SYSTEM"
    data: Dict[str, Any] = Field(default_factory=dict)


class DataMessage(BaseMessage[Dict[str, Any]]):
    type: Literal["DATA"] = "DATA"
    data: Dict[str, Any] = Field(default_factory=dict)


class InfoMessage(BaseMessage[Dict[str, Any]]):
    type: Literal["INFO"] = "INFO"
    data: Dict[str, Any] = Field(default_factory=dict)


class ErrorMessage(BaseMessage[Dict[str, Any]]):
    type: Literal["ERROR"] = "ERROR"
    data: Dict[str, Any] = Field(default_factory=dict)


class AskQuestionMessage(BaseMessage):
    type: Literal["ASK_QUESTION"] = "ASK_QUESTION"
    question: str
    user_id: Optional[str] = None
    collection_name: Optional[str] = None


class AnswerMessage(BaseMessage):
    type: Literal["ANSWER"] = "ANSWER"
    question: str
    answer: Optional[str] = None
    error: Optional[str] = None
    user_id: Optional[str] = None


class StatusRequestMessage(BaseMessage[None]):
    type: Literal["TASK"] = "TASK"
    request_details: Optional[str] = None


class AgentSystemMessage(BaseMessage):
    type: Literal["AGENT_SYSTEM"] = "AGENT_SYSTEM"
    content: str


class AgentLLMRequestMessage(BaseMessage):
    type: Literal["AGENT_LLM_REQUEST"] = "AGENT_LLM_REQUEST"
    prompt: str


class AgentLLMResponseMessage(BaseMessage):
    type: Literal["AGENT_LLM_RESPONSE"] = "AGENT_LLM_RESPONSE"
    raw_content: str
    parsed_type: Optional[str] = None
    parsed_tool_calls: Optional[Any] = None
    parsed_direct_content: Optional[str] = None
    parsed_tool_result_content: Optional[str] = None
    error_details: Optional[str] = None


class MinimalToolCallRequest(BaseModel):
    type: Literal["MINIMAL_TOOL_CALL_REQUEST"] = "MINIMAL_TOOL_CALL_REQUEST"
    tool_name: str
    arguments: Dict[str, Any]


class AgentTaskMessage(BaseMessage):
    type: Literal["AGENT_TASK"] = "AGENT_TASK"
    task: str
    task_id: str = Field(default_factory=lambda: str(uuid4()))
    sender_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class AgentTaskResultMessage(BaseMessage):
    type: Literal["AGENT_TASK_RESULT"] = "AGENT_TASK_RESULT"
    task_id: str
    result: str
    success: bool = True
    error: Optional[str] = None
    agent_id: Optional[str] = None
