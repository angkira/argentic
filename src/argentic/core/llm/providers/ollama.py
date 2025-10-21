import asyncio
from typing import Any, Dict, List, Optional

import httpx

from argentic.core.logger import get_logger
from argentic.core.protocol.chat_message import AssistantMessage, LLMChatResponse

from .base import ModelProvider

# ---------------------------------
# Helper util for wrapping strings
# ---------------------------------


class OllamaProvider(ModelProvider):
    def __init__(self, config: Dict[str, Any], messager: Optional[Any] = None):
        super().__init__(config, messager)
        self.logger = get_logger(self.__class__.__name__)
        self.model_name = self._get_config_value("ollama_model_name", "gemma3:12b-it-qat")
        self.base_url = self._get_config_value("ollama_base_url", "http://localhost:11434")

        # Get advanced parameters from config
        self.params = self._get_config_value("ollama_parameters", {}) or {}

        # Log key parameters
        self.logger.debug(
            f"Parameters: temperature={self.params.get('temperature', 'default')}, "
            f"top_p={self.params.get('top_p', 'default')}, "
            f"top_k={self.params.get('top_k', 'default')}, "
            f"num_ctx={self.params.get('num_ctx', 'default')}"
        )

    async def _call_api(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        """Internal async call to Ollama /api/chat endpoint using httpx."""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "options": dict(self.params),  # Ensure dict
            "stream": False,
        }
        if "options" in kwargs and isinstance(kwargs["options"], dict):
            payload["options"].update(kwargs["options"])  # type: ignore[arg-type]

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/chat", json=payload, timeout=300.0
                )
                # In tests httpx.Response may not have request set; manually check status
                if response.status_code >= 400:
                    response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                self.logger.error(f"Ollama API error: {e}")
                raise

    def _parse_api_response(self, api_response: Dict[str, Any]) -> LLMChatResponse:
        """Parse Ollama API response to LLMChatResponse."""
        message_data = api_response.get("message", {})
        content = message_data.get("content", "")
        role = message_data.get("role", "assistant")

        assistant = AssistantMessage(role=role, content=content)

        usage = (
            {
                "prompt_tokens": api_response.get("prompt_eval_count", 0),
                "completion_tokens": api_response.get("eval_count", 0),
                "total_tokens": api_response.get("prompt_eval_count", 0)
                + api_response.get("eval_count", 0),
            }
            if "prompt_eval_count" in api_response and "eval_count" in api_response
            else None
        )

        finish_reason = "stop" if api_response.get("done") else None

        return LLMChatResponse(message=assistant, usage=usage, finish_reason=finish_reason)

    def _convert_messages_to_ollama(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Convert dict messages to Ollama format (same as input for now)."""
        return messages  # Ollama uses similar {"role": "...", "content": "..."} format

    # Implement interface methods
    def invoke(self, prompt: str, **kwargs: Any) -> LLMChatResponse:
        # Prefer the asyncio.run path so tests can monkeypatch it easily.
        try:
            api_response = asyncio.run(
                self._call_api([{"role": "user", "content": prompt}], **kwargs)
            )
            return self._parse_api_response(api_response)
        except RuntimeError:
            # Fallback to sync httpx when not in an event loop
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "options": dict(self.params),
                "stream": False,
            }
            if "options" in kwargs and isinstance(kwargs["options"], dict):
                payload["options"].update(kwargs["options"])  # type: ignore[arg-type]

            with httpx.Client() as client:
                response = client.post(f"{self.base_url}/api/chat", json=payload, timeout=300.0)
                if response.status_code >= 400:
                    response.raise_for_status()
                api_response = response.json()
            return self._parse_api_response(api_response)

    async def ainvoke(self, prompt: str, **kwargs: Any) -> LLMChatResponse:
        messages = [{"role": "user", "content": prompt}]
        api_response = await self._call_api(messages, **kwargs)
        return self._parse_api_response(api_response)

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> LLMChatResponse:
        # Prefer asyncio.run path for test monkeypatching
        try:
            api_response = asyncio.run(
                self._call_api(self._convert_messages_to_ollama(messages), **kwargs)
            )
            return self._parse_api_response(api_response)
        except RuntimeError:
            payload = {
                "model": self.model_name,
                "messages": self._convert_messages_to_ollama(messages),
                "options": dict(self.params),
                "stream": False,
            }
            if "options" in kwargs and isinstance(kwargs["options"], dict):
                payload["options"].update(kwargs["options"])  # type: ignore[arg-type]

            with httpx.Client() as client:
                response = client.post(f"{self.base_url}/api/chat", json=payload, timeout=300.0)
                if response.status_code >= 400:
                    response.raise_for_status()
                api_response = response.json()
            return self._parse_api_response(api_response)

    async def achat(
        self, messages: List[Dict[str, str]], tools: Optional[List[Any]] = None, **kwargs: Any
    ) -> LLMChatResponse:
        ollama_messages = self._convert_messages_to_ollama(messages)
        # Note: Ollama doesn't support tools natively; ignore tools param or handle via prompt
        if tools:
            self.logger.warning("Ollama does not support tools; ignoring tools parameter")
        api_response = await self._call_api(ollama_messages, **kwargs)
        return self._parse_api_response(api_response)

    # ------------------------------------------------------------------
    # Unified interface helpers (parity with GoogleGeminiProvider)
    # ------------------------------------------------------------------

    def get_model_name(self) -> str:
        """Return the Ollama model name in use."""
        return str(self.model_name)

    def supports_tools(self) -> bool:
        """Ollama currently does not support tool calling via LangChain bindings."""
        return False

    def supports_streaming(self) -> bool:
        """Return True – Ollama endpoint supports streaming, though not used here."""
        return True

    def get_available_models(self) -> List[str]:
        """Return a static list of commonly available Ollama models (best-effort)."""
        return [
            "gemma3:12b-it-qat",
            "llama3:8b",
            "llama3:70b",
            "phi3:mini",
            "mistral:7b",
        ]

    # ------------------------------------------------------------------
    # Internal retry / circuit-breaker logic (simplified for local HTTP)
    # ------------------------------------------------------------------

    def _is_retryable_error(self, error: Exception) -> bool:
        """Identify retry-worthy transient errors."""
        retryable_types = (ConnectionError, TimeoutError)
        # tenacity wraps some exceptions – inspect __cause__
        if hasattr(error, "__cause__") and error.__cause__:
            return isinstance(error.__cause__, retryable_types)
        return isinstance(error, retryable_types)

    def _create_retry_decorator(self):
        """Create tenacity retry decorator with exponential back-off."""
        from tenacity import (
            retry,
            retry_if_exception_type,
            stop_after_attempt,
            wait_random_exponential,
        )

        return retry(
            stop=stop_after_attempt(3),
            wait=wait_random_exponential(multiplier=1, max=20),
            retry=retry_if_exception_type((ConnectionError, TimeoutError)),
            reraise=True,
        )
