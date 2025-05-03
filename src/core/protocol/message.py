from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional, Type, TypeVar
from paho.mqtt.client import MQTTMessage
import json

T = TypeVar("T", bound="Message")


class MessageType(Enum):
    SYSTEM = "SYSTEM"
    DATA = "DATA"
    INFO = "INFO"
    ERROR = "ERROR"
    TASK = "TASK"
    TASK_RESULT = "TASK_RESULT"
    STATUS = "STATUS"
    REGISTER_TOOL = "REGISTER_TOOL"
    TOOL_REGISTERED = "TOOL_REGISTERED"


class Message:
    """Base class for all protocol messages."""

    def __init__(
        self,
        source: str,
        type: MessageType,
        data: Dict[str, Any],
        recipient: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ):
        self.timestamp = timestamp or datetime.now(timezone.utc)
        self.source = source
        self.recipient = recipient
        self.type = type
        self.data = data

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the message object to a dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "recipient": self.recipient,
            "type": self.type.value,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls: Type[T], data_dict: Dict[str, Any]) -> T:
        """Deserializes a dictionary into a Message object or its subclass."""
        message_type_str = data_dict.get("type")
        if not message_type_str:
            raise ValueError("Message dictionary missing 'type' field")

        try:
            message_type = MessageType(message_type_str)
        except ValueError as e:
            raise ValueError(f"Unknown message type: {message_type_str}") from e

        target_cls = message_subclass_map.get(message_type, cls)

        return target_cls(
            timestamp=datetime.fromisoformat(data_dict["timestamp"]),
            source=data_dict["source"],
            recipient=data_dict.get("recipient"),
            type=message_type,
            data=data_dict["data"],
        )

    def to_mqtt_message(self) -> Any:
        """Converts the message to an MQTT message format."""
        payload = json.dumps(self.to_dict())

        msg = MQTTMessage()
        msg.payload = payload.encode("utf-8")
        return msg

    def from_mqtt_message(cls: Type[T], mqtt_msg: Any) -> T:
        """Creates a Message instance from an MQTT message."""
        try:
            payload_str = mqtt_msg.payload.decode("utf-8")
            data_dict = json.loads(payload_str)
            return cls.from_dict(data_dict)
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError, ValueError) as e:
            print(f"Error decoding MQTT message: {e}")
            raise ValueError("Invalid MQTT message format") from e


class SystemMessage(Message):
    def __init__(
        self,
        source: str,
        data: Dict[str, Any],
        recipient: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ):
        super().__init__(source, MessageType.SYSTEM, data, recipient, timestamp)


class RegisterToolMessage(Message):
    def __init__(
        self,
        source: str,
        tool_name: str,
        tool_manual: str,
        tool_api: str,
        recipient: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ):
        data = {
            "tool_name": tool_name,
            "tool_manual": tool_manual,
            "tool_api": tool_api,
        }
        super().__init__(source, MessageType.REGISTER_TOOL, data, recipient, timestamp)

    @property
    def tool_name(self) -> str:
        return self.data.get("tool_name", "")

    @property
    def tool_manual(self) -> str:
        return self.data.get("tool_manual", "")

    @property
    def tool_api(self) -> str:
        return self.data.get("tool_api", "")


class ToolRegisteredMessage(Message):
    def __init__(
        self,
        source: str,
        tool_id: str,
        tool_name: str,
        recipient: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ):
        data = {
            "tool_id": tool_id,
            "tool_name": tool_name,
        }
        super().__init__(source, MessageType.TOOL_REGISTERED, data, recipient, timestamp)

    @property
    def tool_id(self) -> str:
        return self.data.get("tool_id", "")

    @property
    def tool_name(self) -> str:
        return self.data.get("tool_name", "")


class DataMessage(Message):
    def __init__(
        self,
        source: str,
        data: Dict[str, Any],
        recipient: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ):
        super().__init__(source, MessageType.DATA, data, recipient, timestamp)


class InfoMessage(Message):
    def __init__(
        self,
        source: str,
        data: Dict[str, Any],
        recipient: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ):
        super().__init__(source, MessageType.INFO, data, recipient, timestamp)


class ErrorMessage(Message):
    def __init__(
        self,
        source: str,
        data: Dict[str, Any],
        recipient: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ):
        super().__init__(source, MessageType.ERROR, data, recipient, timestamp)


class TaskMessage(Message):
    def __init__(
        self,
        source: str,
        data: Dict[str, Any],
        recipient: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ):
        super().__init__(source, MessageType.TASK, data, recipient, timestamp)


class TaskResultMessage(Message):
    def __init__(
        self,
        source: str,
        data: Dict[str, Any],
        recipient: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ):
        super().__init__(source, MessageType.TASK_RESULT, data, recipient, timestamp)


class StatusMessage(Message):
    def __init__(
        self,
        source: str,
        data: Dict[str, Any],
        recipient: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ):
        super().__init__(source, MessageType.STATUS, data, recipient, timestamp)


message_subclass_map: Dict[MessageType, Type[Message]] = {
    MessageType.SYSTEM: SystemMessage,
    MessageType.DATA: DataMessage,
    MessageType.INFO: InfoMessage,
    MessageType.ERROR: ErrorMessage,
    MessageType.TASK: TaskMessage,
    MessageType.TASK_RESULT: TaskResultMessage,
    MessageType.STATUS: StatusMessage,
    MessageType.REGISTER_TOOL: RegisterToolMessage,
    MessageType.TOOL_REGISTERED: ToolRegisteredMessage,
}
