from core.protocol.message import BaseMessage
from typing import Literal, Optional


class RegisterToolMessage(BaseMessage):
    type: Literal["REGISTER_TOOL"] = "REGISTER_TOOL"
    tool_name: str
    tool_manual: str
    tool_api: str


class ToolRegisteredMessage(BaseMessage):
    type: Literal["TOOL_REGISTERED"] = "TOOL_REGISTERED"
    tool_id: str
    tool_name: str


class ToolUnregisteredMessage(BaseMessage):
    type: Literal["TOOL_UNREGISTERED"] = "TOOL_UNREGISTERED"
    tool_id: str


class ToolRegistrationErrorMessage(BaseMessage):
    type: Literal["TOOL_REGISTRATION_ERROR"] = "TOOL_REGISTRATION_ERROR"
    error: str
    traceback: Optional[str] = None
    tool_id: Optional[str] = None
    tool_name: Optional[str] = None


class UnregisterToolMessage(BaseMessage):
    type: Literal["UNREGISTER_TOOL"] = "UNREGISTER_TOOL"
    tool_id: str
