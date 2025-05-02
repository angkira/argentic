import subprocess
import shlex
import os
from typing import Dict, Any
import requests  # For HTTP API calls
from subprocess import Popen

# Ollama backend via LangChain
from langchain_ollama import OllamaLLM


class OllamaChatLLM:
    """
    Adapter for OllamaLLM to expose a chat interface.
    """

    def __init__(self, model: str):
        self.llm = OllamaLLM(model=model)

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        # Use invoke() correctly with a string prompt
        raw = self.llm.invoke(prompt)
        # Normalize to a native string
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
        # Combine all message contents into a single prompt
        combined = []
        for m in messages:
            role = m.get("role", "").upper()
            content = m.get("content", "")
            combined.append(f"{role}: {content}")
        prompt = "\n".join(combined)
        # Use our __call__ to leverage invoke() and normalization
        return self(prompt, **kwargs)


class LlamaCppLLM:
    """
    Wrapper that invokes the llama.cpp compiled binary for text generation.
    """

    def __init__(self, model_path: str, binary_path: str, args: list[str] = []):
        self.model_path = os.path.expanduser(model_path)
        self.binary_path = os.path.expanduser(binary_path)
        self.args = args or []
        if not os.path.isfile(self.binary_path):
            raise FileNotFoundError(f"llama.cpp binary not found at {self.binary_path}")
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        # Build command: binary, -m model, additional args, -p prompt
        cmd = [self.binary_path, "-m", self.model_path] + self.args + ["-p", prompt]
        # Run subprocess and capture stdout
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
            raise RuntimeError(f"llama.cpp subprocess failed: {e.stderr}")


class LlamaServerLLM:
    """
    Wrapper to interact with a llama.cpp HTTP server via REST API.
    Optionally auto-starts the server binary if provided.
    """

    def __init__(
        self,
        host: str,
        port: int,
        binary_path: str = None,
        binary_args: list[str] = None,
        auto_start: bool = False,
    ):
        self.host = host
        self.port = port
        self.binary_path = os.path.expanduser(binary_path) if binary_path else None
        # Expand '~' in any server_args entries (paths)
        self.binary_args = [os.path.expanduser(arg) for arg in (binary_args or [])]
        self.server_url = f"http://{self.host}:{self.port}"
        self._process = None
        if auto_start and self.binary_path:
            # Launch the server in background
            cmd = [self.binary_path] + self.binary_args
            print(f"Launching llama-server with command: {cmd}")
            self._process = Popen(cmd)
            # Wait a moment for the server to start
            import time

            time.sleep(1)

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        # Send prompt to llama.cpp HTTP server via the /completion endpoint
        url = f"{self.server_url}/completion"
        payload = {"prompt": prompt}
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        # The completion endpoint returns JSON with a 'content' field
        return data.get("content", data.get("response", ""))

    def chat(self, messages: list[dict], **kwargs: Any) -> str:
        """Call the OpenAI-compatible chat endpoint on llama.cpp-server."""
        url = f"{self.server_url}/v1/chat/completions"
        payload = {"model": None, "messages": messages}
        # omit model field if not required; llama.cpp-server uses default
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        # return the assistant's content from the first choice
        return data["choices"][0]["message"]["content"]


class LLMFactory:
    """
    Factory to create LLM instances based on a configuration dict.
    Supported backends: 'ollama', 'llamacpp', 'llamaserver'.
    """

    @staticmethod
    def create(config: Dict[str, Any]) -> Any:
        backend = config.get("backend", "ollama").lower()
        if backend == "llamaserver":
            host = config.get("server_host")
            port = config.get("server_port")
            binary = config.get("server_binary")
            args = config.get("server_args", [])
            # Respect auto_start flag from config (default: False)
            auto = bool(config.get("auto_start", False))
            if not host or not port:
                raise ValueError(
                    "'server_host' and 'server_port' must be set for llama server backend."
                )
            return LlamaServerLLM(
                host=host,
                port=port,
                binary_path=binary,
                binary_args=args,
                auto_start=auto,
            )
        elif backend == "ollama":
            model_name = config.get("model_name")
            if not model_name:
                raise ValueError("'model_name' must be set for Ollama backend.")
            # Use chat adapter so .chat() is available
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
