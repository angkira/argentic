"""Argentic - AI Agent Framework"""

__version__ = "0.3.0"

# Re-export key classes for simplified imports
from .core import (
    Agent,
    Messager,
    LLMFactory,
    AskQuestionMessage,
    ModelProvider,
)

__all__ = [
    "Agent",
    "Messager",
    "LLMFactory",
    "AskQuestionMessage",
    "ModelProvider",
]
