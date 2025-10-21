from typing import Any, Dict, Optional

# Import provider classes
from argentic.core.llm.providers.base import ModelProvider
from argentic.core.logger import get_logger


# Lazy import provider classes to avoid import-time dependency errors
def _import_provider(name: str):
    if name == "ollama":
        from argentic.core.llm.providers.ollama import OllamaProvider as P

        return P
    if name == "llama_cpp_server":
        from argentic.core.llm.providers.llama_cpp_server import LlamaCppServerProvider as P

        return P
    if name == "llama_cpp_cli":
        from argentic.core.llm.providers.llama_cpp_cli import LlamaCppCLIProvider as P

        return P
    if name == "google_gemini":
        try:
            from argentic.core.llm.providers.google_gemini import GoogleGeminiProvider as P

            return P
        except Exception as e:  # pragma: no cover
            raise ImportError("google-generativeai is required for GoogleGeminiProvider") from e
    raise KeyError(name)


# Define a mapping from provider names to classes
PROVIDER_NAMES = [
    "ollama",
    "llama_cpp_server",
    "llama_cpp_cli",
    "google_gemini",
]

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
            provider_class = _import_provider(provider_name)
            logger.debug(f"Found provider class: {provider_class.__name__}")
            return provider_class(config, messager)
        except KeyError as e:
            logger.error(f"Unsupported LLM provider: {provider_name}")
            raise ValueError(
                f"Unsupported LLM provider: {provider_name}. Supported providers are: {PROVIDER_NAMES}"
            ) from e
