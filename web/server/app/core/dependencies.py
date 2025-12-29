"""Dependency injection for FastAPI routes."""

from functools import lru_cache
from app.services import AgentService, SupervisorService, WorkflowService, RuntimeService


@lru_cache()
def get_agent_service() -> AgentService:
    """Get the agent service singleton."""
    return AgentService()


@lru_cache()
def get_supervisor_service() -> SupervisorService:
    """Get the supervisor service singleton."""
    return SupervisorService()


@lru_cache()
def get_workflow_service() -> WorkflowService:
    """Get the workflow service singleton."""
    return WorkflowService()


@lru_cache()
def get_runtime_service() -> RuntimeService:
    """Get the runtime service singleton."""
    return RuntimeService()
