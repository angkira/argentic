"""Argentic - AI Agent Framework"""

# Re-export key classes for simplified imports
from .core import (
    Agent,
    Messager,
    LLMFactory,
    AskQuestionMessage,
    ModelProvider,
)
from .cli_client import CliClient
from . import core
from . import services

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
