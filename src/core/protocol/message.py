from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Literal, Optional, TypeVar, Generic
from paho.mqtt.client import MQTTMessage as PahoMQTTMessage
from pydantic import BaseModel, Field
import uuid


class MessageType(str, Enum):
    SYSTEM = "SYSTEM"
    DATA = "DATA"
    INFO = "INFO"
    ERROR = "ERROR"
    TASK = "TASK"


Payload = TypeVar("Payload", bound=Any)


class BaseMessage(BaseModel, Generic[Payload]):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: str
    data: Payload = Field(default_factory=dict)  # Generic type, defaults to Dict
    type: str = Field(default=MessageType.SYSTEM)


class SystemMessage(BaseMessage[Dict[str, Any]]):
    type: Literal["SYSTEM"] = "SYSTEM"
    data: Dict[str, Any]


class DataMessage(BaseMessage[Dict[str, Any]]):
    type: Literal["DATA"] = "DATA"
    data: Dict[str, Any]


class InfoMessage(BaseMessage[Dict[str, Any]]):
    type: Literal["INFO"] = "INFO"
    data: Dict[str, Any]


class ErrorMessage(BaseMessage[Dict[str, Any]]):
    type: Literal["ERROR"] = "ERROR"
    data: Dict[str, Any]


class AskQuestionMessage(BaseMessage[None]):
    type: Literal["ASK_QUESTION"] = "ASK_QUESTION"
    question: str
    user_id: Optional[str] = None
    collection_name: Optional[str] = None


class AnswerMessage(BaseMessage[None]):
    type: Literal["ANSWER"] = "ANSWER"
    question: str
    answer: Optional[str] = None
    error: Optional[str] = None
    user_id: Optional[str] = None


class StatusRequestMessage(BaseMessage[None]):
    type: Literal["STATUS_REQUEST"] = "STATUS_REQUEST"
    request_details: Optional[str] = None


def to_mqtt_message(message: BaseMessage) -> PahoMQTTMessage:
    """Converts a Pydantic message model to a Paho MQTTMessage."""
    payload = message.model_dump_json()
    msg = PahoMQTTMessage()
    msg.payload = payload.encode("utf-8")
    return msg
