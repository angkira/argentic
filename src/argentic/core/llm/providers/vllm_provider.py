"""
vLLM Provider for Argentic - OpenAI-compatible API Server.

This provider connects to a vLLM server running with OpenAI-compatible API.
For direct model loading (no server), use VLLMNativeProvider instead.
"""

from typing import Any, Dict, Optional

from argentic.core.llm.providers.openai_compatible import OpenAICompatibleProvider
from argentic.core.logger import LogLevel


class VLLMProvider(OpenAICompatibleProvider):
    """
    vLLM provider using OpenAI-compatible API server.

    This is a convenience wrapper around OpenAICompatibleProvider specifically
    configured for vLLM servers. For more control or to use other OpenAI-compatible
    services, use OpenAICompatibleProvider directly.

    Setup:
        1. Install vLLM: pip install vllm
        2. Start server: vllm serve google/gemma-3n-E4B-it --port 8000
        3. Configure this provider to point to the server

    For direct model loading without server:
        Use VLLMNativeProvider instead (supports pre-computed embeddings)

    Config Example:
        {
            "llm": {
                "provider": "vllm",
                "vllm_base_url": "http://localhost:8000/v1",
                "vllm_model_name": "google/gemma-3n-E4B-it",
                "vllm_api_key": "dummy",  # vLLM doesn't require real key
                "temperature": 0.7,
                "max_tokens": 2048,
            }
        }

    Note on Embeddings:
        This provider (OpenAI API) does NOT support pre-computed image embeddings.
        For embedding support, use VLLMNativeProvider with enable_mm_embeds=True.
    """

    def __init__(self, config: Dict[str, Any], messager: Optional[Any] = None):
        """Initialize vLLM provider."""
        # Convert vLLM-specific config keys to generic OpenAI config
        openai_config = {
            "base_url": config.get("vllm_base_url", "http://localhost:8000/v1"),
            "api_key": config.get("vllm_api_key", "dummy"),
            "model_name": config.get("vllm_model_name"),
            "temperature": config.get("temperature", 0.7),
            "max_tokens": config.get("max_tokens", 2048),
            "top_p": config.get("top_p", 1.0),
            "timeout": config.get("timeout", 120.0),  # Longer for vision models
            "log_level": config.get("log_level", LogLevel.INFO),
        }

        # Auto-detect model if not provided
        if not openai_config["model_name"]:
            openai_config["model_name"] = self._fetch_first_model(
                openai_config["base_url"]
            )

        # Initialize parent with converted config
        super().__init__(openai_config, messager)

        self.logger.info(
            f"VLLMProvider (OpenAI API) initialized: {self.base_url} (model: {self.model_name})"
        )

    def _fetch_first_model(self, base_url: str) -> str:
        """
        Fetch the first available model from vLLM server.

        Args:
            base_url: The base URL of the vLLM server

        Returns:
            Model name/ID from the server
        """
        import httpx

        try:
            # Make synchronous request to /v1/models endpoint
            models_url = f"{base_url.rstrip('/v1')}/v1/models"
            with httpx.Client(timeout=10.0) as client:
                response = client.get(models_url)
                response.raise_for_status()
                models_data = response.json()

                if models_data.get("data") and len(models_data["data"]) > 0:
                    model_id = models_data["data"][0]["id"]
                    self.logger.info(f"Auto-detected model: {model_id}")
                    return model_id
                else:
                    raise ValueError("No models found on vLLM server")

        except Exception as e:
            # Fall back to default if auto-detection fails
            fallback = "google/gemma-3n-E4B-it"
            self.logger.warning(
                f"Could not auto-detect model from {models_url}: {e}. "
                f"Using fallback: {fallback}"
            )
            return fallback

    def get_model_name(self) -> str:
        """Return the model name with vllm: prefix."""
        return f"vllm:{self.model_name}"
