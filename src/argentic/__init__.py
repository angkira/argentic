"""Argentic - AI Agent Framework"""

# Re-export key classes for simplified imports
from . import core, services
from .cli_client import CliClient
from .core import (
    Agent,
    AskQuestionMessage,
    LLMFactory,
    Messager,
    ModelProvider,
)

# Multi-agent classes are available via:
# from argentic.core.graph.supervisor import Supervisor
# from argentic.core.graph.state import AgentState

__all__ = [
    "Agent",
    "Messager",
    "LLMFactory",
    "AskQuestionMessage",
    "ModelProvider",
    "CliClient",
    "core",
    "services",
]
