from typing import List, Optional, Dict, Any, Union, Literal
from pydantic import BaseModel, Field

from .message import BaseMessage


class ToolCallRequest(BaseModel):
    tool_id: str = Field(..., description="Unique ID for the tool to be called")
    arguments: Dict[str, Any] = Field(..., description="Arguments for the tool call")


class RegisterToolMessage(BaseMessage):
    type: Literal["REGISTER_TOOL"] = "REGISTER_TOOL"
    tool_name: str = Field(..., description="Name of the tool to register")
    tool_manual: str = Field(..., description="Manual/description for the tool")
    tool_api: str = Field(..., description="API spec for the tool (JSON string)")


class UnregisterToolMessage(BaseMessage):
    type: Literal["UNREGISTER_TOOL"] = "UNREGISTER_TOOL"
    tool_id: str = Field(..., description="Unique ID of the tool to unregister")


class ToolRegisteredMessage(BaseMessage):
    type: Literal["TOOL_REGISTERED"] = "TOOL_REGISTERED"
    tool_id: str = Field(..., description="Unique ID assigned to the registered tool")
    tool_name: str = Field(..., description="Name of the registered tool")


class ToolUnregisteredMessage(BaseMessage):
    type: Literal["TOOL_UNREGISTERED"] = "TOOL_UNREGISTERED"
    tool_id: str = Field(..., description="Unique ID of the unregistered tool")


class ToolRegistrationErrorMessage(BaseMessage):
    type: Literal["TOOL_REGISTRATION_ERROR"] = "TOOL_REGISTRATION_ERROR"
    tool_name: str = Field(..., description="Name of the tool that failed to register")
    error: str = Field(..., description="Error message describing the registration failure")
