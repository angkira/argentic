from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Literal, Optional, Union
from paho.mqtt.client import MQTTMessage as PahoMQTTMessage  # Rename to avoid conflict
from pydantic import BaseModel, Field, ValidationError, RootModel, field_validator
import json
import uuid


# Keep the MessageType Enum as is
class MessageType(str, Enum):
    SYSTEM = "SYSTEM"
    DATA = "DATA"
    INFO = "INFO"
    ERROR = "ERROR"
    TASK = "TASK"
    TASK_RESULT = "TASK_RESULT"
    STATUS = "STATUS"
    REGISTER_TOOL = "REGISTER_TOOL"
    TOOL_REGISTERED = "TOOL_REGISTERED"
    UNREGISTER_TOOL = "UNREGISTER_TOOL"  # New message type for tool unregistration
    ASK_QUESTION = "ASK_QUESTION"
    ANSWER = "ANSWER"
    ADD_INFO = "ADD_INFO"
    FORGET_INFO = "FORGET_INFO"
    STATUS_REQUEST = "STATUS_REQUEST"


# Base Pydantic model for all messages
class BaseMessage(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: str
    recipient: Optional[str] = None
    # The 'type' field will be added by subclasses using Literal


# Specific message models inheriting from BaseMessage
# Use Literal[...] for the 'type' field to enable discriminated unions


class SystemMessage(BaseMessage):
    type: Literal[MessageType.SYSTEM] = MessageType.SYSTEM
    data: Dict[str, Any]  # Keep generic data for system messages


class RegisterToolMessage(BaseMessage):
    type: Literal[MessageType.REGISTER_TOOL] = MessageType.REGISTER_TOOL
    tool_name: str
    tool_manual: str
    tool_api: str  # Assuming API spec is a string, adjust if needed


class ToolRegisteredMessage(BaseMessage):
    type: Literal[MessageType.TOOL_REGISTERED] = MessageType.TOOL_REGISTERED
    tool_id: str
    tool_name: str


class UnregisterToolMessage(BaseMessage):
    type: Literal[MessageType.UNREGISTER_TOOL] = MessageType.UNREGISTER_TOOL
    tool_id: str
    tool_name: str


class DataMessage(BaseMessage):
    type: Literal[MessageType.DATA] = MessageType.DATA
    data: Dict[str, Any]


class InfoMessage(BaseMessage):
    type: Literal[MessageType.INFO] = MessageType.INFO
    data: Dict[str, Any]


class ErrorMessage(BaseMessage):
    type: Literal[MessageType.ERROR] = MessageType.ERROR
    data: Dict[str, Any]  # Consider defining specific error fields like 'code', 'message'


class AskQuestionMessage(BaseMessage):
    type: Literal[MessageType.ASK_QUESTION] = MessageType.ASK_QUESTION
    question: str
    user_id: Optional[str] = None  # Optional user identifier for context/session
    collection_name: Optional[str] = None  # Optional target collection


class AnswerMessage(BaseMessage):
    type: Literal[MessageType.ANSWER] = MessageType.ANSWER
    question: str  # Echo the original question
    answer: Optional[str] = None
    error: Optional[str] = None
    user_id: Optional[str] = None  # Echo user_id if provided


class AddInfoMessage(BaseMessage):
    type: Literal[MessageType.ADD_INFO] = MessageType.ADD_INFO
    text: str
    collection_name: Optional[str] = None
    source_info: Optional[str] = "mqtt_add_info"  # Default source if not specified
    metadata: Optional[Dict[str, Any]] = None


class ForgetMessage(BaseMessage):
    type: Literal[MessageType.FORGET_INFO] = MessageType.FORGET_INFO
    where_filter: Dict[str, Any]
    collection_name: Optional[str] = None

    @field_validator("where_filter")
    def check_where_filter_not_empty(cls, v):
        if not v:
            raise ValueError("'where_filter' cannot be empty for safety.")
        return v


class StatusRequestMessage(BaseMessage):
    type: Literal[MessageType.STATUS_REQUEST] = MessageType.STATUS_REQUEST
    request_details: Optional[str] = None


class TaskMessage(BaseMessage):
    type: Literal[MessageType.TASK] = MessageType.TASK
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_id: str
    payload: Any  # Generic payload field that can contain any data without strict validation


class TaskResultMessage(BaseMessage):
    type: Literal[MessageType.TASK_RESULT] = MessageType.TASK_RESULT
    task_id: str
    tool_id: str  # Include tool_id for context
    status: "TaskStatus"  # Use the existing TaskStatus enum
    result: Optional[Any] = None
    error: Optional[str] = None


class StatusMessage(BaseMessage):
    type: Literal[MessageType.STATUS] = MessageType.STATUS
    data: Dict[str, Any]


class TaskStatus(str, Enum):
    PENDING = "pending"
    SUCCESS = "success"
    ERROR = "error"


# --- Discriminated Union ---
# Create a Union of all specific message types
AnyMessage = Union[
    SystemMessage,
    RegisterToolMessage,
    ToolRegisteredMessage,
    UnregisterToolMessage,
    DataMessage,
    InfoMessage,
    ErrorMessage,
    AskQuestionMessage,
    AnswerMessage,
    AddInfoMessage,
    ForgetMessage,
    StatusRequestMessage,
    TaskMessage,
    TaskResultMessage,
    StatusMessage,
]


# Use RootModel for easy parsing of the discriminated union
class MessageContainer(RootModel[AnyMessage]):
    root: AnyMessage = Field(..., discriminator="type")


# --- Helper Functions ---


def to_mqtt_message(message: BaseMessage) -> PahoMQTTMessage:
    """Converts a Pydantic message model to a Paho MQTTMessage."""
    # Use model_dump_json for direct JSON serialization respecting Pydantic settings
    payload = message.model_dump_json()
    msg = PahoMQTTMessage()  # Use the renamed import
    msg.payload = payload.encode("utf-8")
    # msg.topic would typically be set by the publisher
    return msg


def from_mqtt_message(mqtt_msg: PahoMQTTMessage) -> AnyMessage:
    """Creates a specific Pydantic Message instance from a Paho MQTT message."""
    try:
        payload_str = mqtt_msg.payload.decode("utf-8")
        # Use the MessageContainer to parse and validate the correct subclass
        container = MessageContainer.model_validate_json(payload_str)
        return container.root  # Return the actual message object
    except (json.JSONDecodeError, UnicodeDecodeError, ValidationError) as e:
        # Log the problematic payload and error for debugging
        payload_preview = mqtt_msg.payload[:100].decode("utf-8", errors="replace")
        print(f"Error decoding MQTT message payload: '{payload_preview}...'. Error: {e}")
        raise ValueError("Invalid MQTT message format or content") from e
    except Exception as e:  # Catch any other unexpected errors
        payload_preview = mqtt_msg.payload[:100].decode("utf-8", errors="replace")
        print(
            f"Unexpected error processing MQTT message payload: '{payload_preview}...'. Error: {e}"
        )
        raise ValueError("Unexpected error processing MQTT message") from e
