"""Supervisor configuration models."""

from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field


class WorkerAgentConfig(BaseModel):
    """Configuration for a worker agent in a supervisor."""

    role: str = Field(..., description="Worker agent role")
    description: str = Field(..., description="Worker agent description")


class SupervisorConfig(BaseModel):
    """Base supervisor configuration."""

    role: str = Field(..., description="Supervisor role/identifier", min_length=1)
    description: str = Field(..., description="Supervisor description", min_length=1)
    system_prompt: Optional[str] = Field(None, description="System prompt for the supervisor")
    worker_agents: List[WorkerAgentConfig] = Field(
        default_factory=list, description="List of worker agents"
    )
    enable_dialogue_logging: bool = Field(True, description="Enable dialogue logging")
    max_dialogue_history_items: int = Field(
        100, description="Max dialogue history items", ge=1
    )


class SupervisorCreate(SupervisorConfig):
    """Schema for creating a new supervisor."""

    llm_config_id: Optional[str] = Field(None, description="LLM configuration ID to use")


class SupervisorUpdate(BaseModel):
    """Schema for updating a supervisor."""

    description: Optional[str] = Field(None, description="Supervisor description")
    system_prompt: Optional[str] = Field(None, description="System prompt")
    worker_agents: Optional[List[WorkerAgentConfig]] = Field(
        None, description="List of worker agents"
    )
    enable_dialogue_logging: Optional[bool] = Field(None, description="Enable dialogue logging")
    max_dialogue_history_items: Optional[int] = Field(
        None, description="Max dialogue history items", ge=1
    )


class SupervisorResponse(SupervisorConfig):
    """Schema for supervisor response."""

    id: str = Field(..., description="Supervisor ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    status: str = Field("stopped", description="Supervisor status")

    class Config:
        from_attributes = True
