import os
import subprocess
from subprocess import Popen
from typing import Any, Dict, Optional
import json

import requests

from langchain_ollama import OllamaLLM


class OllamaChatLLM:
    def __init__(self, model: str):
        self.llm = OllamaLLM(model=model)

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        raw = self.llm.invoke(prompt)
        if isinstance(raw, str):
            return raw
        if hasattr(raw, "value"):
            return raw.value
        if hasattr(raw, "content"):
            return raw.content
        if isinstance(raw, dict):
            return next(iter(raw.values()))
        return str(raw)

    def chat(self, messages: list[dict], **kwargs: Any) -> str:
        combined = []
        for m in messages:
            role = m.get("role", "").upper()
            content = m.get("content", "")
            combined.append(f"{role}: {content}")
        prompt = "\n".join(combined)
        return self(prompt, **kwargs)


class LlamaCppLLM:
    def __init__(self, model_path: str, binary_path: str, args: list[str] = None):
        self.model_path = os.path.expanduser(model_path)
        self.binary_path = os.path.expanduser(binary_path)
        self.args = args or []
        if not os.path.isfile(self.binary_path):
            raise FileNotFoundError(f"llama.cpp binary not found at {self.binary_path}")
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        cmd = [self.binary_path, "-m", self.model_path] + self.args + ["-p", prompt]
        try:
            result = subprocess.run(
                cmd,
                input=prompt,
                text=True,
                capture_output=True,
                check=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"llama.cpp subprocess failed: {e.stderr}, cmd: {cmd}") from e


class LlamaServerLLM:
    def __init__(
        self,
        host: str,
        port: int,
        messager: Any,
        binary_path: str = None,
        binary_args: list[str] = None,
        auto_start: bool = False,
    ):
        self.host = host
        self.port = port
        self.messager = messager
        self.binary_path = os.path.expanduser(binary_path) if binary_path else None
        self.binary_args = [os.path.expanduser(arg) for arg in (binary_args or [])]
        self.server_url = f"http://{self.host}:{self.port}"
        self._process = None
        if auto_start and self.binary_path:
            cmd = [self.binary_path] + self.binary_args
            self.messager.log(f"Launching llama-server with command: {cmd}")
            self._process = Popen(cmd)
            import time

            time.sleep(1)

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        url = f"{self.server_url}/completion"
        payload = {"prompt": prompt}
        if "grammar" in kwargs:
            payload["grammar"] = kwargs["grammar"]
        resp = requests.post(url, json=payload, timeout=600)
        resp.raise_for_status()
        data = resp.json()
        return data.get("content", data.get("response", ""))

    def chat(self, messages: list[dict], grammar: Optional[str] = None, **kwargs: Any) -> str:
        url = f"{self.server_url}/v1/chat/completions"
        payload: Dict[str, Any] = {"model": None, "messages": messages}
        if grammar:
            payload["grammar"] = grammar

        self.messager.log(
            f"Sending chat request to {url} with payload: {json.dumps(payload, indent=2)}"
        )
        resp = requests.post(url, json=payload, timeout=600)
        self.messager.log(f"Received response status: {resp.status_code}")

        try:
            resp.raise_for_status()
            data = resp.json()
            self.messager.log(f"Received LLM response data: {json.dumps(data, indent=2)}")
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content", "")
            return content
        except requests.exceptions.HTTPError as http_err:
            self.messager.log(
                f"HTTP error calling LLM: {http_err} - Response: {resp.text}", level="error"
            )
            raise
        except Exception as e:
            self.messager.log(f"Error processing LLM response: {e}", level="error")
            raise


class LLMFactory:
    @staticmethod
    def create(config: Dict[str, Any], messager: Any) -> Any:
        backend = config.get("backend", "ollama").lower()
        if backend == "llamaserver":
            host = config.get("server_host")
            port = config.get("server_port")
            binary = config.get("server_binary")
            args = config.get("server_args", [])
            auto = bool(config.get("auto_start", False))
            if not host or not port:
                raise ValueError(
                    "'server_host' and 'server_port' must be set for llama server backend."
                )
            return LlamaServerLLM(
                host=host,
                port=port,
                messager=messager,
                binary_path=binary,
                binary_args=args,
                auto_start=auto,
            )
        elif backend == "ollama":
            model_name = config.get("model_name")
            if not model_name:
                raise ValueError("'model_name' must be set for Ollama backend.")
            return OllamaChatLLM(model_name)
        elif backend == "llamacpp":
            model_path = config.get("model_path")
            binary = config.get("llama_cpp_binary")
            args = config.get("llama_cpp_args", [])
            if not model_path or not binary:
                raise ValueError(
                    "'model_path' and 'llama_cpp_binary' must be set for llama.cpp backend."
                )
            return LlamaCppLLM(model_path=model_path, binary_path=binary, args=args)
        else:
            raise ValueError(f"Unsupported LLM backend: {backend}")
