import os
import time
import asyncio
import subprocess
from subprocess import Popen
from typing import Any, Dict, List, Optional

import httpx  # Using httpx for async requests

from .base import ModelProvider
from core.logger import get_logger


class LlamaCppServerProvider(ModelProvider):
    def __init__(self, config: Dict[str, Any], messager: Optional[Any] = None):
        super().__init__(config, messager)
        self.logger = get_logger(self.__class__.__name__)

        llm_config = config.get("llm", {})
        server_config_block = llm_config.get("llama_cpp_server", {})
        default_gen_settings = llm_config.get("default_generation_settings", {})

        self.server_binary = os.path.expanduser(
            server_config_block.get("llama_cpp_server_binary", "")
        )
        # Startup arguments for the server process itself
        self.startup_args = server_config_block.get("startup_args", [])
        self.server_host = server_config_block.get("llama_cpp_server_host", "127.0.0.1")
        self.server_port = int(server_config_block.get("llama_cpp_server_port", 8080))
        self.auto_start = server_config_block.get("llama_cpp_server_auto_start", False)
        self.server_url = f"http://{self.server_host}:{self.server_port}"
        self._process: Optional[Popen] = None
        self.timeout = 600  # seconds for requests

        # Prepare and store generation parameters for API calls
        self.generation_params: Dict[str, Any] = {**default_gen_settings}
        specific_gen_settings = server_config_block.get("generation_settings", {})
        self.generation_params.update(specific_gen_settings)

        # Map generic max_tokens to Llama.cpp's n_predict if not already set
        if "n_predict" not in self.generation_params and "max_tokens" in self.generation_params:
            self.generation_params["n_predict"] = self.generation_params.pop("max_tokens")

        # Llama.cpp server specific: rename stop_sequences to stop
        if "stop_sequences" in self.generation_params:
            self.generation_params["stop"] = self.generation_params.pop("stop_sequences")

        self.logger.debug(f"Llama.cpp server generation params set to: {self.generation_params}")

        if self.auto_start and not self.server_binary:
            self.logger.warning(
                "llama_cpp_server_auto_start is true, but llama_cpp_server_binary is not set."
            )
        elif self.auto_start and not os.path.isfile(self.server_binary):
            self.logger.error(
                f"llama_cpp_server_binary not found at: {self.server_binary}. Cannot auto-start."
            )
            self.auto_start = False  # Disable auto_start if binary is missing

    async def start(self) -> None:
        if self.auto_start and self.server_binary and self._process is None:
            cmd = [self.server_binary] + [str(arg) for arg in self.startup_args]
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
        # llama.cpp server /completion endpoint
        # Start with base generation params, then override with call-specific kwargs
        payload = {**self.generation_params, "prompt": prompt, **kwargs}
        # Remove known chat params if any passed via kwargs to avoid issues with /completion
        payload.pop("messages", None)
        # Ensure n_predict has a default if not in merged params somehow
        if "n_predict" not in payload:
            payload["n_predict"] = 128

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
        payload = {**self.generation_params, "prompt": prompt, **kwargs}
        payload.pop("messages", None)
        if "n_predict" not in payload:
            payload["n_predict"] = 128
        response_data = await self._make_request("/completion", payload)
        return response_data.get("content", "")

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        # llama.cpp server /v1/chat/completions endpoint (OpenAI compatible)
        # Start with base generation params, then override with call-specific kwargs
        payload = {**self.generation_params, "messages": messages, **kwargs}
        # Remove prompt if it accidentally got in from default_generation_settings and we are in chat mode
        payload.pop("prompt", None)

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
        payload = {**self.generation_params, "messages": messages, **kwargs}
        payload.pop("prompt", None)
        response_data = await self._make_request("/v1/chat/completions", payload)
        if response_data.get("choices") and response_data["choices"][0].get("message"):
            return response_data["choices"][0]["message"].get("content", "")
        return ""
