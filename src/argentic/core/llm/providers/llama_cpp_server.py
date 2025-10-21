import asyncio
import os
import subprocess
from subprocess import Popen
from typing import Any, Dict, List, Optional

import httpx  # Using httpx for async requests

from argentic.core.logger import get_logger
from argentic.core.protocol.chat_message import AssistantMessage, LLMChatResponse

from .base import ModelProvider

# Removed LangChain dependency - using our own message types and httpx for HTTP calls




class LlamaCppServerProvider(ModelProvider):
    def __init__(self, config: Dict[str, Any], messager: Optional[Any] = None):
        super().__init__(config, messager)
        self.logger = get_logger(self.__class__.__name__)
        self.server_binary = os.path.expanduser(
            self._get_config_value("llama_cpp_server_binary", "")
        )
        server_args_config = self._get_config_value("llama_cpp_server_args", [])
        self.server_args = server_args_config if server_args_config is not None else []
        self.server_host = self._get_config_value("llama_cpp_server_host", "127.0.0.1")

        # Handle port with proper type checking
        port_value = self._get_config_value("llama_cpp_server_port", 8080)
        self.server_port = int(port_value) if port_value is not None else 8080

        self.auto_start = self._get_config_value("llama_cpp_server_auto_start", False)
        self.server_url = f"http://{self.server_host}:{self.server_port}"
        self._process: Optional[Popen] = None
        self.timeout = 600  # seconds for requests

        # Get advanced parameters from config
        self.server_params = self._get_config_value("llama_cpp_server_parameters", {}) or {}

        # Validate server binary path if auto_start is enabled
        if self.auto_start and not self.server_binary:
            self.logger.warning(
                "llama_cpp_server_auto_start is true, but llama_cpp_server_binary is not set."
            )
        elif self.auto_start and self.server_binary and not os.path.isfile(self.server_binary):
            self.logger.error(
                f"llama_cpp_server_binary not found at: {self.server_binary}. Cannot auto-start."
            )
            self.auto_start = False  # Disable auto_start if binary is missing

    async def start(self) -> None:
        if self.auto_start and self.server_binary and self._process is None:
            cmd = [self.server_binary] + [str(arg) for arg in self.server_args]
            # Expand user paths in arguments
            cmd = [os.path.expanduser(c) if isinstance(c, str) and "~" in c else c for c in cmd]
            self.logger.info(f"Attempting to start llama.cpp server with command: {' '.join(cmd)}")
            try:
                self._process = Popen(cmd)
                # Give server time to start
                await asyncio.sleep(5)  # Increased sleep time
                if self._process.poll() is not None:
                    self.logger.error(
                        f"llama.cpp server process exited immediately with code {self._process.returncode}."
                    )
                    self._process = None
                else:
                    self.logger.info(f"llama.cpp server started with PID {self._process.pid}.")
            except Exception as e:
                self.logger.error(f"Failed to start llama.cpp server: {e}")
                self._process = None

    async def stop(self) -> None:
        if self._process and self._process.poll() is None:
            self.logger.info(f"Stopping llama.cpp server (PID: {self._process.pid})...")
            self._process.terminate()
            try:
                await asyncio.to_thread(self._process.wait, timeout=10)
                self.logger.info("llama.cpp server stopped gracefully.")
            except subprocess.TimeoutExpired:
                self.logger.warning("llama.cpp server did not terminate gracefully, killing...")
                self._process.kill()
                await asyncio.to_thread(self._process.wait)
                self.logger.info("llama.cpp server killed.")
            self._process = None

    async def _make_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.server_url}{endpoint}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                self.logger.debug(f"Sending request to {url} with payload: {payload}")
                response = await client.post(url, json=payload)
                response.raise_for_status()
                return response.json()
            except httpx.RequestError as e:
                self.logger.error(f"Error requesting {url}: {e}")
                raise ConnectionError(f"Failed to connect to Llama.cpp server at {url}: {e}") from e
            except httpx.HTTPStatusError as e:
                self.logger.error(
                    f"Llama.cpp server request failed: {e.response.status_code} - {e.response.text}"
                )
                raise ValueError(
                    f"Llama.cpp server error: {e.response.status_code} - {e.response.text}"
                ) from e

    def _to_ai(self, text: str) -> LLMChatResponse:
        return LLMChatResponse(
            message=AssistantMessage(role="assistant", content=str(text)),
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            finish_reason="stop",
        )

    def invoke(self, prompt: str, **kwargs: Any) -> LLMChatResponse:  # type: ignore[override]

        # Fallback to HTTP server approach
        # llama.cpp server /completion endpoint
        # Start with configured parameters, then override with kwargs
        payload: Dict[str, Any] = dict(self.server_params)  # Copy configured parameters
        payload.update(
            {"prompt": prompt, "n_predict": kwargs.get("n_predict", payload.get("n_predict", 128))}
        )
        payload.update(kwargs)  # Override with any additional kwargs

        # Remove known chat params if any passed via kwargs to avoid issues with /completion
        payload.pop("messages", None)

        # Use asyncio.run to properly handle the async request in a sync context
        # This avoids blocking future.result() calls that can stall the thread pool
        try:
            response_data = asyncio.run(self._make_request("/completion", payload))
        except RuntimeError:
            # If we're already in an event loop, we need to handle it differently
            # This should only happen in edge cases - prefer ainvoke() for async contexts
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                # Run in a separate thread to avoid blocking the current event loop
                def _sync_request():
                    return asyncio.run(self._make_request("/completion", payload))

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_sync_request)
                    response_data = future.result(timeout=self.timeout)
            else:
                response_data = asyncio.run(self._make_request("/completion", payload))
        return self._to_ai(response_data.get("content", ""))

    async def ainvoke(self, prompt: str, **kwargs: Any) -> LLMChatResponse:  # type: ignore[override]

        # Fallback to HTTP server approach
        # Start with configured parameters, then override with kwargs
        payload: Dict[str, Any] = dict(self.server_params)  # Copy configured parameters
        payload.update(
            {"prompt": prompt, "n_predict": kwargs.get("n_predict", payload.get("n_predict", 128))}
        )
        payload.update(kwargs)  # Override with any additional kwargs
        payload.pop("messages", None)

        response_data = await self._make_request("/completion", payload)
        return self._to_ai(response_data.get("content", ""))

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> LLMChatResponse:  # type: ignore[override]

        # Fallback to HTTP server approach
        # llama.cpp server /v1/chat/completions endpoint (OpenAI compatible)
        # Start with configured parameters, then override with kwargs
        payload: Dict[str, Any] = dict(self.server_params)  # Copy configured parameters
        payload.update({"messages": messages})
        payload.update(kwargs)  # Override with any additional kwargs

        # Use asyncio.run to properly handle the async request in a sync context
        # This avoids blocking future.result() calls that can stall the thread pool
        try:
            response_data = asyncio.run(self._make_request("/v1/chat/completions", payload))
        except RuntimeError:
            # If we're already in an event loop, we need to handle it differently
            # This should only happen in edge cases - prefer achat() for async contexts
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                # Run in a separate thread to avoid blocking the current event loop
                def _sync_request():
                    return asyncio.run(self._make_request("/v1/chat/completions", payload))

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_sync_request)
                    response_data = future.result(timeout=self.timeout)
            else:
                response_data = asyncio.run(self._make_request("/v1/chat/completions", payload))

        if response_data.get("choices") and response_data["choices"][0].get("message"):
            return self._to_ai(response_data["choices"][0]["message"].get("content", ""))
        return self._to_ai("")

    async def achat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> LLMChatResponse:  # type: ignore[override]

        # Fallback to HTTP server approach
        # Start with configured parameters, then override with kwargs
        payload: Dict[str, Any] = dict(self.server_params)  # Copy configured parameters
        payload.update({"messages": messages})
        payload.update(kwargs)  # Override with any additional kwargs

        response_data = await self._make_request("/v1/chat/completions", payload)
        if response_data.get("choices") and response_data["choices"][0].get("message"):
            return self._to_ai(response_data["choices"][0]["message"].get("content", ""))
        return self._to_ai("")

    # ------------------------------------------------------------------
    # Unified interface helpers (parity with GoogleGeminiProvider)
    # ------------------------------------------------------------------

    def get_model_name(self) -> str:
        return "llama.cpp-server"

    def supports_tools(self) -> bool:
        return False

    def supports_streaming(self) -> bool:
        # llama.cpp server supports streaming via SSE but not implemented
        return False

    def get_available_models(self) -> List[str]:
        return [self.get_model_name()]

    # ------------------------------------------------------------------
    # End of LlamaCppServerProvider extension
    # ------------------------------------------------------------------
