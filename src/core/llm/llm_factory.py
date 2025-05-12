import os
import sys
from typing import Any, Dict, Optional

# Import provider classes
from core.llm.providers.base import ModelProvider
from core.llm.providers.ollama import OllamaProvider
from core.llm.providers.llama_cpp_server import LlamaCppServerProvider
from core.llm.providers.llama_cpp_cli import LlamaCppCLIProvider
from core.llm.providers.google_gemini import GoogleGeminiProvider

from core.logger import get_logger

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

        if provider_name == "ollama":
            return OllamaProvider(config, messager)
        elif provider_name == "llama_cpp_server":
            return LlamaCppServerProvider(config, messager)
        elif provider_name == "llama_cpp_cli":
            return LlamaCppCLIProvider(config, messager)
        elif provider_name == "google_gemini":
            return GoogleGeminiProvider(config, messager)
        else:
            logger.error(f"Unsupported LLM provider: {provider_name}")
            raise ValueError(f"Unsupported LLM provider: {provider_name}")
