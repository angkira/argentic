"""API routes for agent management."""

from typing import List
from fastapi import APIRouter, HTTPException, Depends, status
from app.models import AgentCreate, AgentUpdate, AgentResponse
from app.services import AgentService
from app.core.dependencies import get_agent_service

router = APIRouter(prefix="/api/agents", tags=["agents"])


@router.post("", response_model=AgentResponse, status_code=status.HTTP_201_CREATED)
async def create_agent(
    agent_data: AgentCreate,
    service: AgentService = Depends(get_agent_service),
):
    """Create a new agent configuration."""
    return await service.create_agent(agent_data)


@router.get("", response_model=List[AgentResponse])
async def list_agents(
    service: AgentService = Depends(get_agent_service),
):
    """List all agent configurations."""
    return await service.list_agents()


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: str,
    service: AgentService = Depends(get_agent_service),
):
    """Get an agent configuration by ID."""
    agent = await service.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent


@router.patch("/{agent_id}", response_model=AgentResponse)
async def update_agent(
    agent_id: str,
    agent_data: AgentUpdate,
    service: AgentService = Depends(get_agent_service),
):
    """Update an agent configuration."""
    agent = await service.update_agent(agent_id, agent_data)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent


@router.delete("/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent(
    agent_id: str,
    service: AgentService = Depends(get_agent_service),
):
    """Delete an agent configuration."""
    deleted = await service.delete_agent(agent_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Agent not found")


@router.post("/{agent_id}/start", response_model=AgentResponse)
async def start_agent(
    agent_id: str,
    service: AgentService = Depends(get_agent_service),
):
    """Start an agent."""
    agent = await service.update_agent_status(agent_id, "running")
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent


@router.post("/{agent_id}/stop", response_model=AgentResponse)
async def stop_agent(
    agent_id: str,
    service: AgentService = Depends(get_agent_service),
):
    """Stop an agent."""
    agent = await service.update_agent_status(agent_id, "stopped")
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent
