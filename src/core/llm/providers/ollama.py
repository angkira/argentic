from typing import Any, Dict, List, Optional, Union

from langchain_ollama import OllamaLLM
from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage

from .base import ModelProvider
from core.logger import get_logger


class OllamaProvider(ModelProvider):
    llm: Union[ChatOllama, OllamaLLM]

    def __init__(self, config: Dict[str, Any], messager: Optional[Any] = None):
        super().__init__(config, messager)
        self.logger = get_logger(self.__class__.__name__)

        llm_config = config.get("llm", {})
        ollama_config_block = llm_config.get("ollama", {})
        default_gen_settings = llm_config.get("default_generation_settings", {})

        self.model_name = ollama_config_block.get("ollama_model_name", "gemma3:12b-it-qat")
        self.use_chat_model = ollama_config_block.get("ollama_use_chat_model", True)
        self.base_url = ollama_config_block.get("ollama_base_url", "http://localhost:11434")

        # Prepare generation parameters
        gen_params: Dict[str, Any] = {**default_gen_settings}
        specific_gen_settings = ollama_config_block.get("generation_settings", {})
        gen_params.update(specific_gen_settings)

        # Map generic max_tokens to Ollama's num_predict if not already set
        if "num_predict" not in gen_params and "max_tokens" in gen_params:
            gen_params["num_predict"] = gen_params.pop("max_tokens")

        # Ollama specific: rename stop_sequences to stop
        if "stop_sequences" in gen_params:
            gen_params["stop"] = gen_params.pop("stop_sequences")

        # Parameters accepted by Langchain's OllamaLLM/ChatOllama constructors
        # (This list might not be exhaustive but covers common ones from config)
        accepted_ollama_params = [
            "mirostat",
            "mirostat_eta",
            "mirostat_tau",
            "num_ctx",
            "num_gpu",
            "num_thread",
            "repeat_last_n",
            "repeat_penalty",
            "temperature",
            "stop",
            "tfs_z",
            "top_k",
            "top_p",
            "seed",
            "num_predict",
            # Format, keep_alive, etc. are handled separately or have defaults
        ]

        final_generation_config = {
            k: v for k, v in gen_params.items() if k in accepted_ollama_params and v is not None
        }
        self.logger.debug(f"Final generation config for Ollama: {final_generation_config}")

        if self.use_chat_model:
            self.llm = ChatOllama(
                model=self.model_name, base_url=self.base_url, **final_generation_config
            )
            self.logger.info(
                f"Initialized ChatOllama with model: {self.model_name} at {self.base_url} with config: {final_generation_config}"
            )
        else:
            self.llm = OllamaLLM(
                model=self.model_name, base_url=self.base_url, **final_generation_config
            )
            self.logger.info(
                f"Initialized OllamaLLM with model: {self.model_name} at {self.base_url} with config: {final_generation_config}"
            )

    def _parse_llm_result(self, result: Any) -> str:
        if isinstance(result, BaseMessage):
            return result.content
        elif isinstance(result, str):
            return result
        self.logger.warning(
            f"Unexpected result type from Ollama: {type(result)}. Converting to string."
        )
        return str(result)

    def _convert_messages_to_langchain(self, messages: List[Dict[str, str]]) -> List[BaseMessage]:
        lc_messages = []
        for msg in messages:
            role = msg.get("role", "user").lower()
            content = msg.get("content", "")
            if role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            elif role == "system":
                lc_messages.append(SystemMessage(content=content))
            else:  # Fallback for unknown roles
                lc_messages.append(HumanMessage(content=f"{role}: {content}"))
        return lc_messages

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        if self.use_chat_model:  # ChatOllama expects list of messages
            result = self.llm.invoke([HumanMessage(content=prompt)], **kwargs)
        else:  # OllamaLLM expects a string
            result = self.llm.invoke(prompt, **kwargs)
        return self._parse_llm_result(result)

    async def ainvoke(self, prompt: str, **kwargs: Any) -> str:
        if self.use_chat_model:
            result = await self.llm.ainvoke([HumanMessage(content=prompt)], **kwargs)
        else:
            result = await self.llm.ainvoke(prompt, **kwargs)
        return self._parse_llm_result(result)

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        if self.use_chat_model:
            lc_messages = self._convert_messages_to_langchain(messages)
            result = self.llm.invoke(lc_messages, **kwargs)
        else:  # Fallback for non-chat model
            prompt = self._format_chat_messages_to_prompt(messages)
            result = self.llm.invoke(prompt, **kwargs)
        return self._parse_llm_result(result)

    async def achat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        if self.use_chat_model:
            lc_messages = self._convert_messages_to_langchain(messages)
            result = await self.llm.ainvoke(lc_messages, **kwargs)
        else:
            prompt = self._format_chat_messages_to_prompt(messages)
            result = await self.llm.ainvoke(prompt, **kwargs)
        return self._parse_llm_result(result)
