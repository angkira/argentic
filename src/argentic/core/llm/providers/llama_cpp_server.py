import os
import asyncio
import subprocess
from subprocess import Popen
from typing import Any, Dict, List, Optional

import httpx  # Using httpx for async requests

# Langchain imports for alternative implementation
try:
    from langchain_community.llms.llamacpp import LlamaCpp

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LlamaCpp = None
    LANGCHAIN_AVAILABLE = False

from .base import ModelProvider
from core.logger import get_logger


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

        # Langchain integration option
        self.use_langchain = self._get_config_value("llama_cpp_use_langchain", False)
        self.model_path = self._get_config_value("llama_cpp_model_path", "")

        # Get advanced parameters from config
        self.server_params = self._get_config_value("llama_cpp_server_parameters", {}) or {}

        # Initialize Langchain LLM if requested and available
        self.langchain_llm: Optional[Any] = None
        if self.use_langchain and LANGCHAIN_AVAILABLE and self.model_path and LlamaCpp is not None:
            try:
                # Use langchain parameters if available, otherwise fall back to server params
                langchain_params = (
                    self._get_config_value("llama_cpp_langchain_parameters", {}) or {}
                )

                self.langchain_llm = LlamaCpp(
                    model_path=self.model_path,
                    temperature=langchain_params.get(
                        "temperature", self.server_params.get("temperature", 0.7)
                    ),
                    max_tokens=langchain_params.get(
                        "max_tokens", self.server_params.get("n_predict", 256)
                    ),
                    n_ctx=langchain_params.get("n_ctx", self.server_params.get("n_ctx", 2048)),
                    n_gpu_layers=langchain_params.get(
                        "n_gpu_layers", self.server_params.get("n_gpu_layers", 0)
                    ),
                    verbose=langchain_params.get("verbose", False),
                )
                self.logger.info(f"Initialized Langchain LlamaCpp with model: {self.model_path}")
            except Exception as e:
                self.logger.error(f"Failed to initialize Langchain LlamaCpp: {e}")
                self.use_langchain = False

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

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        # Use Langchain if available and configured
        if self.use_langchain and self.langchain_llm is not None:
            try:
                return self.langchain_llm.invoke(prompt, **kwargs)
            except Exception as e:
                self.logger.error(f"Langchain LlamaCpp invoke failed: {e}, falling back to HTTP")

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

        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If in async context, run sync in executor
            future = asyncio.run_coroutine_threadsafe(
                self._make_request("/completion", payload), loop
            )
            response_data = future.result(timeout=self.timeout)
        else:
            response_data = asyncio.run(self._make_request("/completion", payload))
        return response_data.get("content", "")

    async def ainvoke(self, prompt: str, **kwargs: Any) -> str:
        # Use Langchain if available and configured
        if self.use_langchain and self.langchain_llm is not None:
            try:
                return await self.langchain_llm.ainvoke(prompt, **kwargs)
            except Exception as e:
                self.logger.error(f"Langchain LlamaCpp ainvoke failed: {e}, falling back to HTTP")

        # Fallback to HTTP server approach
        # Start with configured parameters, then override with kwargs
        payload: Dict[str, Any] = dict(self.server_params)  # Copy configured parameters
        payload.update(
            {"prompt": prompt, "n_predict": kwargs.get("n_predict", payload.get("n_predict", 128))}
        )
        payload.update(kwargs)  # Override with any additional kwargs
        payload.pop("messages", None)

        response_data = await self._make_request("/completion", payload)
        return response_data.get("content", "")

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        # Use Langchain if available and configured
        if self.use_langchain and self.langchain_llm is not None:
            try:
                # Convert messages to prompt for Langchain LlamaCpp
                prompt = self._format_chat_messages_to_prompt(messages)
                return self.langchain_llm.invoke(prompt, **kwargs)
            except Exception as e:
                self.logger.error(f"Langchain LlamaCpp chat failed: {e}, falling back to HTTP")

        # Fallback to HTTP server approach
        # llama.cpp server /v1/chat/completions endpoint (OpenAI compatible)
        # Start with configured parameters, then override with kwargs
        payload: Dict[str, Any] = dict(self.server_params)  # Copy configured parameters
        payload.update({"messages": messages})
        payload.update(kwargs)  # Override with any additional kwargs

        loop = asyncio.get_event_loop()
        if loop.is_running():
            future = asyncio.run_coroutine_threadsafe(
                self._make_request("/v1/chat/completions", payload), loop
            )
            response_data = future.result(timeout=self.timeout)
        else:
            response_data = asyncio.run(self._make_request("/v1/chat/completions", payload))

        if response_data.get("choices") and response_data["choices"][0].get("message"):
            return response_data["choices"][0]["message"].get("content", "")
        return ""

    async def achat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        # Use Langchain if available and configured
        if self.use_langchain and self.langchain_llm is not None:
            try:
                # Convert messages to prompt for Langchain LlamaCpp
                prompt = self._format_chat_messages_to_prompt(messages)
                return await self.langchain_llm.ainvoke(prompt, **kwargs)
            except Exception as e:
                self.logger.error(f"Langchain LlamaCpp achat failed: {e}, falling back to HTTP")

        # Fallback to HTTP server approach
        # Start with configured parameters, then override with kwargs
        payload: Dict[str, Any] = dict(self.server_params)  # Copy configured parameters
        payload.update({"messages": messages})
        payload.update(kwargs)  # Override with any additional kwargs

        response_data = await self._make_request("/v1/chat/completions", payload)
        if response_data.get("choices") and response_data["choices"][0].get("message"):
            return response_data["choices"][0]["message"].get("content", "")
        return ""
