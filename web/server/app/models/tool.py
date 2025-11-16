"""Tool configuration models."""

from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class ToolConfig(BaseModel):
    """Base tool configuration."""

    name: str = Field(..., description="Tool name", min_length=1)
    description: str = Field(..., description="Tool description", min_length=1)
    tool_type: str = Field(..., description="Tool type (rag, environment, custom)")
    config: Dict[str, Any] = Field(default_factory=dict, description="Tool-specific configuration")


class ToolCreate(ToolConfig):
    """Schema for creating a new tool."""

    pass


class ToolResponse(ToolConfig):
    """Schema for tool response."""

    id: str = Field(..., description="Tool ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    status: str = Field("stopped", description="Tool status")

    class Config:
        from_attributes = True
