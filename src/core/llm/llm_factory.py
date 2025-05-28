from typing import Any, Dict, Optional

# Import provider classes
from core.llm.providers.base import ModelProvider
from core.llm.providers.ollama import OllamaProvider
from core.llm.providers.llama_cpp_server import LlamaCppServerProvider
from core.llm.providers.llama_cpp_cli import LlamaCppCLIProvider
from core.llm.providers.llama_cpp_langchain import LlamaCppLangchainProvider
from core.llm.providers.google_gemini import GoogleGeminiProvider

from core.logger import get_logger

# Define a mapping from provider names to classes
PROVIDER_MAP: dict[str, type[ModelProvider]] = {
    "ollama": OllamaProvider,
    "llama_cpp_server": LlamaCppServerProvider,
    "llama_cpp_cli": LlamaCppCLIProvider,
    "llama_cpp_langchain": LlamaCppLangchainProvider,
    "google_gemini": GoogleGeminiProvider,
}

# The start_llm_server function might be deprecated or moved if
# LlamaCppServerProvider's auto_start is sufficient.
# For now, I'll leave it but comment it out from factory usage.
# from subprocess import Popen
# import time

logger = get_logger(__name__)

# def start_llm_server(config: Dict[str, Any]) -> None: ... (existing function can remain for now)


class LLMFactory:
    @staticmethod
    def create(config: Dict[str, Any], messager: Optional[Any] = None) -> ModelProvider:
        """
        Creates a ModelProvider instance based on the configuration.

        Args:
            config: The main application configuration dictionary.
            messager: Optional messager instance for logging/communication.

        Returns:
            An instance of a ModelProvider.
        """
        llm_config = config.get("llm", {})
        provider_name = llm_config.get("provider", "ollama").lower()  # Default to ollama

        logger.info(f"Creating LLM provider: {provider_name}")

        try:
            provider_class = PROVIDER_MAP[provider_name]
            logger.debug(f"Found provider class: {provider_class.__name__}")
            return provider_class(config, messager)
        except KeyError as e:
            logger.error(f"Unsupported LLM provider: {provider_name}")
            raise ValueError(
                f"Unsupported LLM provider: {provider_name}. Supported providers are: {list(PROVIDER_MAP.keys())}"
            ) from e
