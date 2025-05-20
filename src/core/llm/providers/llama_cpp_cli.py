import os
import asyncio
import subprocess
from typing import Any, Dict, List, Optional

from .base import ModelProvider
from core.logger import get_logger


class LlamaCppCLIProvider(ModelProvider):
    def __init__(self, config: Dict[str, Any], messager: Optional[Any] = None):
        super().__init__(config, messager)
        self.logger = get_logger(self.__class__.__name__)

        llm_config = config.get("llm", {})
        cli_config_block = llm_config.get("llama_cpp_cli", {})
        default_gen_settings = llm_config.get("default_generation_settings", {})

        self.cli_binary = os.path.expanduser(cli_config_block.get("llama_cpp_cli_binary", ""))
        self.model_path = os.path.expanduser(cli_config_block.get("llama_cpp_cli_model_path", ""))
        self.additional_cli_flags = cli_config_block.get("additional_cli_flags", [])

        if not self.cli_binary or not os.path.isfile(self.cli_binary):
            raise ValueError(f"Llama.cpp CLI binary not found or not specified: {self.cli_binary}")
        if not self.model_path or not os.path.isfile(self.model_path):
            raise ValueError(f"Llama.cpp model path not found or not specified: {self.model_path}")

        # Prepare and store generation parameters for CLI command construction
        self.generation_params: Dict[str, Any] = {**default_gen_settings}
        specific_gen_params = cli_config_block.get("generation_parameters", {})
        self.generation_params.update(specific_gen_params)

        self.logger.info(
            f"Initialized LlamaCppCLIProvider with binary: {self.cli_binary}, model: {self.model_path}"
        )
        self.logger.debug(f"Llama.cpp CLI base generation params: {self.generation_params}")
        self.logger.debug(f"Llama.cpp CLI additional flags: {self.additional_cli_flags}")

    def _build_command(self, prompt: str, **kwargs: Any) -> List[str]:
        # Start with base generation params, then override with call-specific kwargs
        current_params = {**self.generation_params, **kwargs}

        cmd = [self.cli_binary, "-m", self.model_path]

        # Map and add parameters from current_params to CLI arguments
        param_to_cli_arg = {
            "temperature": "--temp",
            "n_predict": "--n-predict",
            "max_tokens": "--n-predict",  # Alias for n_predict from default_settings
            "top_k": "--top-k",
            "top_p": "--top-p",
            "repeat_penalty": "--repeat-penalty",
            "seed": "--seed",
            # Add other direct mappings here as needed
        }

        for param_name, cli_flag in param_to_cli_arg.items():
            if param_name in current_params and current_params[param_name] is not None:
                # Special handling for max_tokens if n_predict is also directly provided in kwargs
                # Prefer n_predict if both max_tokens and n_predict are in current_params
                if param_name == "max_tokens" and "n_predict" in current_params:
                    if (
                        current_params.get("n_predict") is not None
                    ):  # n_predict from kwargs takes precedence
                        continue  # Skip adding --n-predict for max_tokens if n_predict is set
                    # else if n_predict is None but max_tokens is not, it will be handled by n_predict mapping later or here

                cmd.extend([cli_flag, str(current_params[param_name])])

        # Handle stop sequences (Llama.cpp CLI uses multiple --stop "sequence" arguments)
        stop_sequences = current_params.get("stop_sequences") or current_params.get(
            "stop"
        )  # Check both keys
        if stop_sequences:
            if isinstance(stop_sequences, list):
                for seq in stop_sequences:
                    cmd.extend(["--stop", str(seq)])
            elif isinstance(stop_sequences, str):  # Handle if a single stop string is provided
                cmd.extend(["--stop", str(stop_sequences)])

        # Add any other fixed/additional CLI flags from config
        cmd.extend([str(arg) for arg in self.additional_cli_flags])

        # Finally, add the prompt
        cmd.extend(["-p", prompt])

        return cmd

    def invoke(self, prompt: str, **kwargs: Any) -> str:
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
            # A simple way to get text after prompt, assuming prompt is at the start of output
            if output.startswith(prompt.strip()):
                # Add one for potential space or newline after prompt in output
                return output[len(prompt.strip()) :].strip()
            return output  # Fallback if prompt not found at start
        except subprocess.CalledProcessError as e:
            self.logger.error(
                f"Llama.cpp CLI execution failed. CMD: {' '.join(e.cmd)}. Error: {e.stderr}"
            )
            raise RuntimeError(f"Llama.cpp CLI error: {e.stderr}") from e

    async def ainvoke(self, prompt: str, **kwargs: Any) -> str:
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
                return output[len(prompt.strip()) :].strip()
            return output
        except Exception as e:
            self.logger.error(f"Async Llama.cpp CLI execution failed: {e}")
            raise RuntimeError(f"Async Llama.cpp CLI error: {e}") from e

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        prompt = self._format_chat_messages_to_prompt(messages)
        return self.invoke(prompt, **kwargs)

    async def achat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        prompt = self._format_chat_messages_to_prompt(messages)
        return await self.ainvoke(prompt, **kwargs)
