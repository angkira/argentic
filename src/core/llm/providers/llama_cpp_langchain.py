from typing import Any, Dict, List, Optional

try:
    from langchain_community.llms.llamacpp import LlamaCpp

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from .base import ModelProvider
from core.logger import get_logger


class LlamaCppLangchainProvider(ModelProvider):
    """
    LlamaCpp provider using Langchain's native integration.

    This provider uses Langchain's LlamaCpp class directly for local model inference.
    It supports both CPU and GPU acceleration depending on your llama-cpp-python installation.
    """

    def __init__(self, config: Dict[str, Any], messager: Optional[Any] = None):
        super().__init__(config, messager)
        self.logger = get_logger(self.__class__.__name__)

        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "Langchain LlamaCpp dependencies not available. "
                "Install with: pip install langchain-community llama-cpp-python"
            )

        # Required configuration
        self.model_path = self._get_config_value("llama_cpp_model_path")
        if not self.model_path:
            raise ValueError(
                "llama_cpp_model_path is required for LlamaCppLangchainProvider. "
                "Please specify the path to your GGUF model file in the config."
            )

        # Get advanced parameters from config
        params = self._get_config_value("llama_cpp_langchain_parameters", {}) or {}

        # LlamaCpp specific parameters
        llm_params = {
            "model_path": self.model_path,
            "temperature": params.get("temperature", 0.7),
            "max_tokens": params.get("max_tokens", 256),
            "top_p": params.get("top_p", 0.95),
            "top_k": params.get("top_k", 40),
            "repeat_penalty": params.get("repeat_penalty", 1.1),
            "n_ctx": params.get("n_ctx", 2048),
            "n_batch": params.get("n_batch", 8),
            "n_threads": params.get("n_threads", None),
            "n_gpu_layers": params.get("n_gpu_layers", 0),
            "f16_kv": params.get("f16_kv", True),
            "use_mlock": params.get("use_mlock", False),
            "use_mmap": params.get("use_mmap", True),
            "verbose": params.get("verbose", False),
        }

        # Remove None values to use LlamaCpp defaults
        llm_params = {k: v for k, v in llm_params.items() if v is not None}

        try:
            self.llm = LlamaCpp(**llm_params)
            self.logger.info(f"Initialized LlamaCpp with model: {self.model_path}")

            # Log GPU usage if configured
            if llm_params.get("n_gpu_layers", 0) > 0:
                self.logger.info(f"Using GPU acceleration with {llm_params['n_gpu_layers']} layers")

            # Log key parameters
            self.logger.debug(
                f"Parameters: temperature={llm_params['temperature']}, "
                f"max_tokens={llm_params['max_tokens']}, "
                f"n_ctx={llm_params['n_ctx']}, "
                f"n_gpu_layers={llm_params['n_gpu_layers']}"
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize LlamaCpp: {e}")
            raise

    def _parse_llm_result(self, result: Any) -> str:
        """Parse the result from LlamaCpp."""
        if isinstance(result, str):
            return result
        else:
            self.logger.warning(f"Unexpected result type from LlamaCpp: {type(result)}")
            return str(result)

    def _convert_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to a single prompt string for LlamaCpp."""
        formatted_messages = []
        for msg in messages:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            formatted_messages.append(f"{role}: {content}")
        return "\n".join(formatted_messages)

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        """Synchronously invoke the model with a single prompt."""
        try:
            result = self.llm.invoke(prompt, **kwargs)
            return self._parse_llm_result(result)
        except Exception as e:
            self.logger.error(f"LlamaCpp invoke failed: {e}")
            raise

    async def ainvoke(self, prompt: str, **kwargs: Any) -> str:
        """Asynchronously invoke the model with a single prompt."""
        try:
            result = await self.llm.ainvoke(prompt, **kwargs)
            return self._parse_llm_result(result)
        except Exception as e:
            self.logger.error(f"LlamaCpp ainvoke failed: {e}")
            raise

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """Synchronously invoke the model with chat messages."""
        try:
            prompt = self._convert_messages_to_prompt(messages)
            result = self.llm.invoke(prompt, **kwargs)
            return self._parse_llm_result(result)
        except Exception as e:
            self.logger.error(f"LlamaCpp chat failed: {e}")
            raise

    async def achat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """Asynchronously invoke the model with chat messages."""
        try:
            prompt = self._convert_messages_to_prompt(messages)
            result = await self.llm.ainvoke(prompt, **kwargs)
            return self._parse_llm_result(result)
        except Exception as e:
            self.logger.error(f"LlamaCpp achat failed: {e}")
            raise

    async def start(self) -> None:
        """Optional startup method - LlamaCpp loads model on first use."""
        self.logger.info("LlamaCpp provider ready - model will load on first inference")

    async def stop(self) -> None:
        """Optional cleanup method."""
        self.logger.info("LlamaCpp provider stopped")
