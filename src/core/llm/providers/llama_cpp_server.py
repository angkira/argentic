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
        self.server_binary = os.path.expanduser(
            self._get_config_value("llama_cpp_server_binary", "")
        )
        self.server_args = self._get_config_value("llama_cpp_server_args", [])
        self.server_host = self._get_config_value("llama_cpp_server_host", "127.0.0.1")
        self.server_port = int(
            self._get_config_value("llama_cpp_server_port", 8080)
        )  # llama.cpp default
        self.auto_start = self._get_config_value("llama_cpp_server_auto_start", False)
        self.server_url = f"http://{self.server_host}:{self.server_port}"
        self._process: Optional[Popen] = None
        self.timeout = 600  # seconds for requests

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
        # llama.cpp server /completion endpoint
        payload = {"prompt": prompt, "n_predict": kwargs.get("n_predict", 128), **kwargs}
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
        payload = {"prompt": prompt, "n_predict": kwargs.get("n_predict", 128), **kwargs}
        payload.pop("messages", None)
        response_data = await self._make_request("/completion", payload)
        return response_data.get("content", "")

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        # llama.cpp server /v1/chat/completions endpoint (OpenAI compatible)
        payload = {"messages": messages, **kwargs}
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
        payload = {"messages": messages, **kwargs}
        response_data = await self._make_request("/v1/chat/completions", payload)
        if response_data.get("choices") and response_data["choices"][0].get("message"):
            return response_data["choices"][0]["message"].get("content", "")
        return ""
