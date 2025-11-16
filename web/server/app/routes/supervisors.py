"""API routes for supervisor management."""

from typing import List
from fastapi import APIRouter, HTTPException, Depends, status
from app.models import SupervisorCreate, SupervisorUpdate, SupervisorResponse
from app.services import SupervisorService
from app.core.dependencies import get_supervisor_service

router = APIRouter(prefix="/api/supervisors", tags=["supervisors"])


@router.post("", response_model=SupervisorResponse, status_code=status.HTTP_201_CREATED)
async def create_supervisor(
    supervisor_data: SupervisorCreate,
    service: SupervisorService = Depends(get_supervisor_service),
):
    """Create a new supervisor configuration."""
    return await service.create_supervisor(supervisor_data)


@router.get("", response_model=List[SupervisorResponse])
async def list_supervisors(
    service: SupervisorService = Depends(get_supervisor_service),
):
    """List all supervisor configurations."""
    return await service.list_supervisors()


@router.get("/{supervisor_id}", response_model=SupervisorResponse)
async def get_supervisor(
    supervisor_id: str,
    service: SupervisorService = Depends(get_supervisor_service),
):
    """Get a supervisor configuration by ID."""
    supervisor = await service.get_supervisor(supervisor_id)
    if not supervisor:
        raise HTTPException(status_code=404, detail="Supervisor not found")
    return supervisor


@router.patch("/{supervisor_id}", response_model=SupervisorResponse)
async def update_supervisor(
    supervisor_id: str,
    supervisor_data: SupervisorUpdate,
    service: SupervisorService = Depends(get_supervisor_service),
):
    """Update a supervisor configuration."""
    supervisor = await service.update_supervisor(supervisor_id, supervisor_data)
    if not supervisor:
        raise HTTPException(status_code=404, detail="Supervisor not found")
    return supervisor


@router.delete("/{supervisor_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_supervisor(
    supervisor_id: str,
    service: SupervisorService = Depends(get_supervisor_service),
):
    """Delete a supervisor configuration."""
    deleted = await service.delete_supervisor(supervisor_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Supervisor not found")


@router.post("/{supervisor_id}/start", response_model=SupervisorResponse)
async def start_supervisor(
    supervisor_id: str,
    service: SupervisorService = Depends(get_supervisor_service),
):
    """Start a supervisor."""
    supervisor = await service.update_supervisor_status(supervisor_id, "running")
    if not supervisor:
        raise HTTPException(status_code=404, detail="Supervisor not found")
    return supervisor


@router.post("/{supervisor_id}/stop", response_model=SupervisorResponse)
async def stop_supervisor(
    supervisor_id: str,
    service: SupervisorService = Depends(get_supervisor_service),
):
    """Stop a supervisor."""
    supervisor = await service.update_supervisor_status(supervisor_id, "stopped")
    if not supervisor:
        raise HTTPException(status_code=404, detail="Supervisor not found")
    return supervisor
