import os
import asyncio
import subprocess
from typing import Any, Dict, List, Optional

from .base import ModelProvider
from argentic.core.logger import get_logger

from typing import Optional, Any, Dict, List

from langchain_core.messages import AIMessage, BaseMessage


class LlamaCppCLIProvider(ModelProvider):
    def __init__(self, config: Dict[str, Any], messager: Optional[Any] = None):
        super().__init__(config, messager)
        self.logger = get_logger(self.__class__.__name__)
        self.cli_binary = os.path.expanduser(self._get_config_value("llama_cpp_cli_binary", ""))
        self.model_path = os.path.expanduser(self._get_config_value("llama_cpp_cli_model_path", ""))
        self.cli_args = self._get_config_value("llama_cpp_cli_args", []) or []

        # Get advanced parameters from config
        self.cli_params = self._get_config_value("llama_cpp_cli_parameters", {}) or {}

        if not self.cli_binary or not os.path.isfile(self.cli_binary):
            raise ValueError(f"Llama.cpp CLI binary not found or not specified: {self.cli_binary}")
        if not self.model_path or not os.path.isfile(self.model_path):
            raise ValueError(f"Llama.cpp model path not found or not specified: {self.model_path}")
        self.logger.info(
            f"Initialized LlamaCppCLIProvider with binary: {self.cli_binary}, model: {self.model_path}"
        )

        # Log key parameters
        self.logger.debug(
            f"Parameters: temperature={self.cli_params.get('temperature', 'default')}, "
            f"ctx_size={self.cli_params.get('ctx_size', 'default')}, "
            f"n_gpu_layers={self.cli_params.get('n_gpu_layers', 'default')}"
        )

    def _build_command(self, prompt: str, **kwargs: Any) -> List[str]:
        cmd = [self.cli_binary, "-m", self.model_path, "-p", prompt]
        cmd.extend(
            [str(arg) for arg in self.cli_args if arg is not None]
        )  # Add default args from config

        # Add configured parameters as CLI arguments
        if "temperature" in self.cli_params:
            cmd.extend(["--temp", str(self.cli_params["temperature"])])
        if "top_k" in self.cli_params:
            cmd.extend(["--top-k", str(self.cli_params["top_k"])])
        if "top_p" in self.cli_params:
            cmd.extend(["--top-p", str(self.cli_params["top_p"])])
        if "repeat_penalty" in self.cli_params:
            cmd.extend(["--repeat-penalty", str(self.cli_params["repeat_penalty"])])
        if "ctx_size" in self.cli_params:
            cmd.extend(["--ctx-size", str(self.cli_params["ctx_size"])])
        if "batch_size" in self.cli_params:
            cmd.extend(["--batch-size", str(self.cli_params["batch_size"])])
        if "threads" in self.cli_params and self.cli_params["threads"] != -1:
            cmd.extend(["--threads", str(self.cli_params["threads"])])
        if "n_gpu_layers" in self.cli_params:
            cmd.extend(["--n-gpu-layers", str(self.cli_params["n_gpu_layers"])])
        if "seed" in self.cli_params and self.cli_params["seed"] != -1:
            cmd.extend(["--seed", str(self.cli_params["seed"])])
        if "n_predict" in self.cli_params:
            cmd.extend(["--n-predict", str(self.cli_params["n_predict"])])
        if self.cli_params.get("mlock", False):
            cmd.append("--mlock")
        if self.cli_params.get("no_mmap", False):
            cmd.append("--no-mmap")

        # Override/add specific args from kwargs if needed
        if "n_predict" in kwargs:  # llama.cpp uses --n-predict
            cmd.extend(["--n-predict", str(kwargs["n_predict"])])
        if "temp" in kwargs:
            cmd.extend(["--temp", str(kwargs["temp"])])
        # Add other relevant llama.cpp params from kwargs as needed
        # Cast to satisfy static typing (all elements are stringified)
        from typing import cast, List

        return cast(List[str], cmd)

    def _to_ai(self, text: str) -> "BaseMessage":
        """Utility to wrap plain text into an AIMessage."""
        return AIMessage(content=text)

    # ------------------------------------------------------------------
    # ModelProvider required implementations returning BaseMessage
    # ------------------------------------------------------------------

    def invoke(self, prompt: str, **kwargs: Any) -> BaseMessage:  # type: ignore[override]
        cmd = self._build_command(prompt, **kwargs)
        self.logger.debug(f"Executing Llama.cpp CLI: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, encoding="utf-8"
            )
            # The output of llama.cpp main usually includes the prompt itself.
            # We need to parse the actual completion.
            # A common pattern is that the completion starts after the prompt.
            # This might need adjustment based on the exact llama.cpp version and verbosity.
            output = result.stdout.strip()
            if output.startswith(prompt.strip()):
                # Add one for potential space or newline after prompt in output
                output = output[len(prompt.strip()) :].strip()
            return self._to_ai(output)  # type: ignore[return-value]
        except subprocess.CalledProcessError as e:
            self.logger.error(
                f"Llama.cpp CLI execution failed. CMD: {' '.join(e.cmd)}. Error: {e.stderr}"
            )
            raise RuntimeError(f"Llama.cpp CLI error: {e.stderr}") from e

    async def ainvoke(self, prompt: str, **kwargs: Any) -> BaseMessage:  # type: ignore[override]
        cmd = self._build_command(prompt, **kwargs)
        self.logger.debug(f"Executing Llama.cpp CLI (async): {' '.join(cmd)}")
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                self.logger.error(
                    f"Llama.cpp CLI execution failed. CMD: {' '.join(cmd)}. Error: {stderr.decode(errors='ignore')}"
                )
                raise RuntimeError(f"Llama.cpp CLI error: {stderr.decode(errors='ignore')}")

            output = stdout.decode(errors="ignore").strip()
            if output.startswith(prompt.strip()):
                output = output[len(prompt.strip()) :].strip()
            return self._to_ai(output)
        except Exception as e:
            self.logger.error(f"Async Llama.cpp CLI execution failed: {e}")
            raise RuntimeError(f"Async Llama.cpp CLI error: {e}") from e

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> BaseMessage:  # type: ignore[override]
        prompt = self._format_chat_messages_to_prompt(messages)
        return self.invoke(prompt, **kwargs)

    async def achat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Any]] = None,  # tools ignored for llama.cpp CLI
        **kwargs: Any,
    ) -> BaseMessage:  # type: ignore[override]
        prompt = self._format_chat_messages_to_prompt(messages)
        return await self.ainvoke(prompt, **kwargs)

    # ------------------------------------------------------------------
    # Unified interface helpers (parity with GoogleGeminiProvider)
    # ------------------------------------------------------------------

    def get_model_name(self) -> str:
        """Return the underlying GGUF model file name."""
        import os

        return str(os.path.basename(self.model_path))

    def supports_tools(self) -> bool:
        """llama.cpp CLI does not support structured tool-calling by itself."""
        return False

    def supports_streaming(self) -> bool:
        """Streaming of tokens can be enabled via CLI flags, but is not exposed here."""
        return False

    def get_available_models(self) -> List[str]:
        """Return list containing only the current model; discovery is not available."""
        return [self.get_model_name()]

    # ------------------------------------------------------------------
    # End of LlamaCppCLIProvider extension
    # ------------------------------------------------------------------
