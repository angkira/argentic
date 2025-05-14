from typing import Any, Dict, List, Optional

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

        if self.use_chat_model:
            self.llm = ChatOllama(model=self.model_name, base_url=self.base_url)
            self.logger.info(
                f"Initialized ChatOllama with model: {self.model_name} at {self.base_url}"
            )
        else:
            self.llm = OllamaLLM(model=self.model_name, base_url=self.base_url)
            self.logger.info(
                f"Initialized OllamaLLM with model: {self.model_name} at {self.base_url}"
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
