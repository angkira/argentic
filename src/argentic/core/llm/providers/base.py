from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from langchain_core.messages import BaseMessage  # Import BaseMessage


class ModelProvider(ABC):
    """Abstract base class for all model providers."""

    def __init__(self, config: Dict[str, Any], messager: Optional[Any] = None):
        self.config = config
        self.messager = messager  # For logging or other interactions if needed

    @abstractmethod
    def invoke(self, prompt: str, **kwargs: Any) -> BaseMessage:
        """Synchronously invoke the model with a single prompt."""
        pass

    @abstractmethod
    async def ainvoke(self, prompt: str, **kwargs: Any) -> BaseMessage:
        """Asynchronously invoke the model with a single prompt."""
        pass

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> BaseMessage:
        """Synchronously invoke the model with a list of chat messages."""
        pass

    @abstractmethod
    async def achat(
        self, messages: List[Dict[str, str]], tools: Optional[List[Any]] = None, **kwargs: Any
    ) -> BaseMessage:
        """Asynchronously invoke the model with a list of chat messages."""
        pass

    def _get_config_value(self, key: str, default: Optional[Any] = None) -> Optional[Any]:
        """Helper to get a value from the llm sub-dictionary in the config."""
        return self.config.get("llm", {}).get(key, default)

    def _format_chat_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Helper to convert a list of chat messages to a single string prompt.
        This should ideally not be used if the LLM provider handles chat messages directly.
        """
        # Basic implementation, can be overridden by providers if they have specific needs
        # or if the underlying model/API handles chat messages directly.
        return "\n".join(
            [f"{m.get('role', 'user').upper()}: {m.get('content', '')}" for m in messages]
        )

    async def start(self) -> None:
        """Optional method for providers that need to start a server or initialize resources."""
        pass

    async def stop(self) -> None:
        """Optional method for providers that need to stop a server or release resources."""
        pass
