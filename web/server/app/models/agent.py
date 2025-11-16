"""Agent configuration models."""

from typing import Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """Base agent configuration."""

    role: str = Field(..., description="Agent role/identifier", min_length=1)
    description: str = Field(..., description="Agent description", min_length=1)
    system_prompt: Optional[str] = Field(None, description="System prompt for the agent")
    expected_output_format: Literal["json", "text", "code"] = Field(
        "json", description="Expected output format"
    )
    enable_dialogue_logging: bool = Field(False, description="Enable dialogue logging")
    max_consecutive_tool_calls: int = Field(3, description="Max consecutive tool calls", ge=1)
    max_dialogue_history_items: int = Field(
        100, description="Max dialogue history items", ge=1
    )
    max_context_iterations: int = Field(10, description="Max context iterations", ge=1)
    enable_adaptive_context_management: bool = Field(
        True, description="Enable adaptive context management"
    )


class AgentCreate(AgentConfig):
    """Schema for creating a new agent."""

    llm_config_id: Optional[str] = Field(None, description="LLM configuration ID to use")


class AgentUpdate(BaseModel):
    """Schema for updating an agent."""

    description: Optional[str] = Field(None, description="Agent description")
    system_prompt: Optional[str] = Field(None, description="System prompt")
    expected_output_format: Optional[Literal["json", "text", "code"]] = Field(
        None, description="Expected output format"
    )
    enable_dialogue_logging: Optional[bool] = Field(None, description="Enable dialogue logging")
    max_consecutive_tool_calls: Optional[int] = Field(
        None, description="Max consecutive tool calls", ge=1
    )
    max_dialogue_history_items: Optional[int] = Field(
        None, description="Max dialogue history items", ge=1
    )


class AgentResponse(AgentConfig):
    """Schema for agent response."""

    id: str = Field(..., description="Agent ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    status: Literal["stopped", "running", "error"] = Field("stopped", description="Agent status")

    class Config:
        from_attributes = True
