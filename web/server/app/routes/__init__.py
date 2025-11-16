"""API routes for the Argentic Web Server."""

from .agents import router as agents_router
from .supervisors import router as supervisors_router
from .workflows import router as workflows_router
from .config import router as config_router

__all__ = ["agents_router", "supervisors_router", "workflows_router", "config_router"]
