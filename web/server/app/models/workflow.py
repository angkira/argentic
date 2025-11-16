"""Workflow/graph configuration models."""

from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field


class WorkflowNodeData(BaseModel):
    """Data associated with a workflow node."""

    label: str = Field(..., description="Node label")
    config: Dict[str, Any] = Field(default_factory=dict, description="Node configuration")


class WorkflowNode(BaseModel):
    """A node in the workflow graph."""

    id: str = Field(..., description="Node ID")
    type: Literal["agent", "supervisor", "tool", "input", "output"] = Field(
        ..., description="Node type"
    )
    position: Dict[str, float] = Field(..., description="Node position (x, y)")
    data: WorkflowNodeData = Field(..., description="Node data")


class WorkflowEdge(BaseModel):
    """An edge connecting nodes in the workflow graph."""

    id: str = Field(..., description="Edge ID")
    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    label: Optional[str] = Field(None, description="Edge label")
    type: Optional[str] = Field("default", description="Edge type")


class WorkflowConfig(BaseModel):
    """Base workflow configuration."""

    name: str = Field(..., description="Workflow name", min_length=1)
    description: str = Field(..., description="Workflow description", min_length=1)
    nodes: List[WorkflowNode] = Field(default_factory=list, description="Workflow nodes")
    edges: List[WorkflowEdge] = Field(default_factory=list, description="Workflow edges")


class WorkflowCreate(WorkflowConfig):
    """Schema for creating a new workflow."""

    pass


class WorkflowUpdate(BaseModel):
    """Schema for updating a workflow."""

    name: Optional[str] = Field(None, description="Workflow name")
    description: Optional[str] = Field(None, description="Workflow description")
    nodes: Optional[List[WorkflowNode]] = Field(None, description="Workflow nodes")
    edges: Optional[List[WorkflowEdge]] = Field(None, description="Workflow edges")


class WorkflowResponse(WorkflowConfig):
    """Schema for workflow response."""

    id: str = Field(..., description="Workflow ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    status: str = Field("stopped", description="Workflow status")

    class Config:
        from_attributes = True
