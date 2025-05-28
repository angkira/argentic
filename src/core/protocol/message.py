from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Literal, Optional, TypeVar, Generic, List
from pydantic import BaseModel, Field
import uuid


class MessageType(str, Enum):
    SYSTEM = "SYSTEM"
    DATA = "DATA"
    INFO = "INFO"
    ERROR = "ERROR"
    TASK = "TASK"
    AGENT_SYSTEM_PROMPT = "AGENT_SYSTEM_PROMPT"
    AGENT_LLM_REQUEST = "AGENT_LLM_REQUEST"
    AGENT_LLM_RESPONSE = "AGENT_LLM_RESPONSE"
    ASK_QUESTION = "ASK_QUESTION"
    ANSWER = "ANSWER"


Payload = TypeVar("Payload")


class MinimalToolCallRequest(BaseModel):
    tool_id: str
    arguments: Dict[str, Any]


class BaseMessage(BaseModel, Generic[Payload]):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = Field(default="agent")
    type: str  # This will be set by each subclass

    data: Optional[Payload] = None


class SystemMessage(BaseMessage[Dict[str, Any]]):
    type: Literal[MessageType.SYSTEM] = MessageType.SYSTEM
    data: Dict[str, Any] = Field(default_factory=dict)


class DataMessage(BaseMessage[Dict[str, Any]]):
    type: Literal[MessageType.DATA] = MessageType.DATA
    data: Dict[str, Any] = Field(default_factory=dict)


class InfoMessage(BaseMessage[Dict[str, Any]]):
    type: Literal[MessageType.INFO] = MessageType.INFO
    data: Dict[str, Any] = Field(default_factory=dict)


class ErrorMessage(BaseMessage[Dict[str, Any]]):
    type: Literal[MessageType.ERROR] = MessageType.ERROR
    data: Dict[str, Any] = Field(default_factory=dict)


class AskQuestionMessage(BaseMessage[None]):
    type: Literal[MessageType.ASK_QUESTION] = MessageType.ASK_QUESTION
    question: str
    user_id: Optional[str] = None
    collection_name: Optional[str] = None


class AnswerMessage(BaseMessage[None]):
    type: Literal[MessageType.ANSWER] = MessageType.ANSWER
    question: str
    answer: Optional[str] = None
    error: Optional[str] = None
    user_id: Optional[str] = None


class StatusRequestMessage(BaseMessage[None]):
    type: Literal[MessageType.TASK] = MessageType.TASK
    request_details: Optional[str] = None


class AgentSystemMessage(BaseMessage[None]):
    type: Literal[MessageType.AGENT_SYSTEM_PROMPT] = MessageType.AGENT_SYSTEM_PROMPT
    content: str


class AgentLLMRequestMessage(BaseMessage[None]):
    type: Literal[MessageType.AGENT_LLM_REQUEST] = MessageType.AGENT_LLM_REQUEST
    prompt: str


class AgentLLMResponseMessage(BaseMessage[None]):
    type: Literal[MessageType.AGENT_LLM_RESPONSE] = MessageType.AGENT_LLM_RESPONSE
    raw_content: str
    parsed_type: Optional[
        Literal[
            "direct", "tool_call", "tool_result", "error_parsing", "error_validation", "error_llm"
        ]
    ] = None
    parsed_direct_content: Optional[str] = None
    parsed_tool_calls: Optional[List[MinimalToolCallRequest]] = None
    parsed_tool_result_content: Optional[str] = None
    error_details: Optional[str] = None
