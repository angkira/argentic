"""API routes for workflow management."""

from typing import List
from fastapi import APIRouter, HTTPException, Depends, status
from app.models import WorkflowCreate, WorkflowUpdate, WorkflowResponse
from app.services import WorkflowService
from app.core.dependencies import get_workflow_service

router = APIRouter(prefix="/api/workflows", tags=["workflows"])


@router.post("", response_model=WorkflowResponse, status_code=status.HTTP_201_CREATED)
async def create_workflow(
    workflow_data: WorkflowCreate,
    service: WorkflowService = Depends(get_workflow_service),
):
    """Create a new workflow configuration."""
    return await service.create_workflow(workflow_data)


@router.get("", response_model=List[WorkflowResponse])
async def list_workflows(
    service: WorkflowService = Depends(get_workflow_service),
):
    """List all workflow configurations."""
    return await service.list_workflows()


@router.get("/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(
    workflow_id: str,
    service: WorkflowService = Depends(get_workflow_service),
):
    """Get a workflow configuration by ID."""
    workflow = await service.get_workflow(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return workflow


@router.patch("/{workflow_id}", response_model=WorkflowResponse)
async def update_workflow(
    workflow_id: str,
    workflow_data: WorkflowUpdate,
    service: WorkflowService = Depends(get_workflow_service),
):
    """Update a workflow configuration."""
    workflow = await service.update_workflow(workflow_id, workflow_data)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return workflow


@router.delete("/{workflow_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_workflow(
    workflow_id: str,
    service: WorkflowService = Depends(get_workflow_service),
):
    """Delete a workflow configuration."""
    deleted = await service.delete_workflow(workflow_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Workflow not found")


@router.post("/{workflow_id}/start", response_model=WorkflowResponse)
async def start_workflow(
    workflow_id: str,
    service: WorkflowService = Depends(get_workflow_service),
):
    """Start a workflow."""
    workflow = await service.update_workflow_status(workflow_id, "running")
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return workflow


@router.post("/{workflow_id}/stop", response_model=WorkflowResponse)
async def stop_workflow(
    workflow_id: str,
    service: WorkflowService = Depends(get_workflow_service),
):
    """Stop a workflow."""
    workflow = await service.update_workflow_status(workflow_id, "stopped")
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return workflow
