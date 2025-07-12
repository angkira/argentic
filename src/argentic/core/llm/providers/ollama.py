from typing import Any, Dict, List, Optional, Union

from langchain_ollama import OllamaLLM
from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage

from .base import ModelProvider
from core.logger import get_logger


class OllamaProvider(ModelProvider):
    def __init__(self, config: Dict[str, Any], messager: Optional[Any] = None):
        super().__init__(config, messager)
        self.logger = get_logger(self.__class__.__name__)
        self.model_name = self._get_config_value("ollama_model_name", "gemma3:12b-it-qat")
        self.use_chat_model = self._get_config_value("ollama_use_chat_model", True)
        self.base_url = self._get_config_value("ollama_base_url", "http://localhost:11434")

        # Get advanced parameters from config
        params = self._get_config_value("ollama_parameters", {}) or {}

        # Build common parameters for both OllamaLLM and ChatOllama
        common_params = {
            "model": self.model_name,
            "base_url": self.base_url,
        }

        # Add sampling parameters if specified
        if "temperature" in params:
            common_params["temperature"] = params["temperature"]
        if "top_p" in params:
            common_params["top_p"] = params["top_p"]
        if "top_k" in params:
            common_params["top_k"] = params["top_k"]
        if "num_predict" in params:
            common_params["num_predict"] = params["num_predict"]
        if "repeat_penalty" in params:
            common_params["repeat_penalty"] = params["repeat_penalty"]
        if "repeat_last_n" in params:
            common_params["repeat_last_n"] = params["repeat_last_n"]
        if "tfs_z" in params:
            common_params["tfs_z"] = params["tfs_z"]
        if "typical_p" in params:
            common_params["typical_p"] = params["typical_p"]
        if "presence_penalty" in params:
            common_params["presence_penalty"] = params["presence_penalty"]
        if "frequency_penalty" in params:
            common_params["frequency_penalty"] = params["frequency_penalty"]
        if "num_ctx" in params:
            common_params["num_ctx"] = params["num_ctx"]
        if "num_batch" in params:
            common_params["num_batch"] = params["num_batch"]
        if "num_gpu" in params:
            common_params["num_gpu"] = params["num_gpu"]
        if "main_gpu" in params:
            common_params["main_gpu"] = params["main_gpu"]
        if "num_thread" in params:
            common_params["num_thread"] = params["num_thread"]
        if "seed" in params:
            common_params["seed"] = params["seed"]
        if "stop" in params:
            common_params["stop"] = params["stop"]
        if "numa" in params:
            common_params["numa"] = params["numa"]
        if "use_mmap" in params:
            common_params["use_mmap"] = params["use_mmap"]
        if "use_mlock" in params:
            common_params["use_mlock"] = params["use_mlock"]

        if self.use_chat_model:
            self.llm: Union[ChatOllama, OllamaLLM] = ChatOllama(**common_params)
            self.logger.info(
                f"Initialized ChatOllama with model: {self.model_name} at {self.base_url}"
            )
        else:
            self.llm = OllamaLLM(**common_params)
            self.logger.info(
                f"Initialized OllamaLLM with model: {self.model_name} at {self.base_url}"
            )

        # Log key parameters
        self.logger.debug(
            f"Parameters: temperature={params.get('temperature', 'default')}, "
            f"top_p={params.get('top_p', 'default')}, "
            f"top_k={params.get('top_k', 'default')}, "
            f"num_ctx={params.get('num_ctx', 'default')}"
        )

    def _parse_llm_result(self, result: Any) -> str:
        if isinstance(result, BaseMessage):
            content = result.content
        elif isinstance(result, str):
            content = result
        else:
            self.logger.warning(
                f"Unexpected result type from Ollama: {type(result)}. Converting to string."
            )
            content = str(result)

        # Ensure content is a string
        if not isinstance(content, str):
            content = str(content)

        return content

    def _convert_messages_to_langchain(self, messages: List[Dict[str, str]]) -> List[BaseMessage]:
        lc_messages: List[BaseMessage] = []
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
            # Type assertion to help with Union type
            assert isinstance(self.llm, ChatOllama)
            result = await self.llm.ainvoke([HumanMessage(content=prompt)], **kwargs)
        else:
            # Type assertion to help with Union type
            assert isinstance(self.llm, OllamaLLM)
            result = await self.llm.ainvoke(prompt, **kwargs)
        return self._parse_llm_result(result)

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        if self.use_chat_model:
            lc_messages = self._convert_messages_to_langchain(messages)
            # Type assertion to help with Union type
            assert isinstance(self.llm, ChatOllama)
            result = self.llm.invoke(lc_messages, **kwargs)
        else:  # Fallback for non-chat model
            prompt = self._format_chat_messages_to_prompt(messages)
            assert isinstance(self.llm, OllamaLLM)
            result = self.llm.invoke(prompt, **kwargs)
        return self._parse_llm_result(result)

    async def achat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        if self.use_chat_model:
            lc_messages = self._convert_messages_to_langchain(messages)
            # Type assertion to help with Union type
            assert isinstance(self.llm, ChatOllama)
            result = await self.llm.ainvoke(lc_messages, **kwargs)
        else:
            prompt = self._format_chat_messages_to_prompt(messages)
            assert isinstance(self.llm, OllamaLLM)
            result = await self.llm.ainvoke(prompt, **kwargs)
        return self._parse_llm_result(result)
