from core.protocol.message import BaseMessage
from typing import Literal


class RegisterToolMessage(BaseMessage):
    type: Literal["REGISTER_TOOL"] = "REGISTER_TOOL"
    tool_name: str
    tool_manual: str
    tool_api: str


class ToolRegisteredMessage(BaseMessage):
    type: Literal["TOOL_REGISTERED"] = "TOOL_REGISTERED"
    tool_id: str
    tool_name: str


class UnregisterToolMessage(BaseMessage):
    type: Literal["UNREGISTER_TOOL"] = "UNREGISTER_TOOL"
    tool_id: str
    tool_name: str
