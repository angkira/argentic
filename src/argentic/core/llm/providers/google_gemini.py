import os
import json
from typing import Any, Dict, List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage

from .base import ModelProvider
from core.logger import get_logger


class GoogleGeminiProvider(ModelProvider):
    def __init__(self, config: Dict[str, Any], messager: Optional[Any] = None):
        super().__init__(config, messager)
        self.logger = get_logger(self.__class__.__name__)
        self.api_key = os.getenv("GEMINI_API_KEY") or self._get_config_value(
            "google_gemini_api_key"
        )
        self.model_name = self._get_config_value("google_gemini_model_name", "gemini-2.0-flash")

        if not self.api_key:
            raise ValueError(
                "Google Gemini API key not found. Set GEMINI_API_KEY environment variable or google_gemini_api_key in config."
            )

        # Get advanced parameters from config
        params = self._get_config_value("google_gemini_parameters", {}) or {}

        # Build LLM initialization parameters
        llm_params = {
            "model": self.model_name,
            "google_api_key": self.api_key,
            "temperature": params.get("temperature", 0.7),
            "top_p": params.get("top_p", 0.95),
            "top_k": params.get("top_k", 40),
            "max_output_tokens": params.get("max_output_tokens", 2048),
        }

        # Build model_kwargs for parameters that need to be passed to the underlying API
        model_kwargs = {}

        # candidate_count should be in model_kwargs to avoid LangChain warnings
        candidate_count = params.get("candidate_count", 1)
        if candidate_count != 1:  # Only add if different from default
            model_kwargs["candidate_count"] = candidate_count

        # Add optional parameters if specified
        if params.get("stop_sequences"):
            llm_params["stop_sequences"] = params["stop_sequences"]

        if params.get("safety_settings"):
            llm_params["safety_settings"] = params["safety_settings"]

        if params.get("response_mime_type"):
            llm_params["response_mime_type"] = params["response_mime_type"]

        if params.get("response_schema"):
            llm_params["response_schema"] = params["response_schema"]

        # Add model_kwargs if any parameters were set
        if model_kwargs:
            llm_params["model_kwargs"] = model_kwargs

        # Initialize with configured parameters
        self.llm = ChatGoogleGenerativeAI(**llm_params)

        self.logger.info(f"Initialized GoogleGeminiProvider with model: {self.model_name}")

        # Log key parameters including candidate_count
        log_msg = (
            f"Parameters: temperature={llm_params['temperature']}, "
            f"top_p={llm_params['top_p']}, top_k={llm_params['top_k']}, "
            f"max_output_tokens={llm_params['max_output_tokens']}"
        )
        if candidate_count != 1:
            log_msg += f", candidate_count={candidate_count}"
        self.logger.debug(log_msg)

    def _parse_llm_result(self, result: Any) -> str:
        if isinstance(result, BaseMessage):
            content = result.content
        elif isinstance(result, str):
            content = result
        else:
            self.logger.warning(
                f"Unexpected result type from Google Gemini: {type(result)}. Converting to string."
            )
            content = str(result)

        # Ensure content is a string
        if not isinstance(content, str):
            content = str(content)

        # Try to parse the content as JSON
        try:
            parsed = json.loads(content)

            # Handle nested tool calls (when Gemini wraps them in a respond function)
            if isinstance(parsed, dict) and "tool_calls" in parsed:
                tool_calls = parsed["tool_calls"]
                if len(tool_calls) == 1 and "function" in tool_calls[0]:
                    # Extract the inner tool call from the respond function
                    inner_content = tool_calls[0]["function"]["arguments"]
                    try:
                        inner_parsed = json.loads(inner_content)
                        if isinstance(inner_parsed, dict) and "content" in inner_parsed:
                            # Extract the actual tool call from the content
                            content_str = inner_parsed["content"]
                            # Remove markdown code block if present
                            if content_str.startswith("```json"):
                                content_str = content_str[7:]
                            if content_str.endswith("```"):
                                content_str = content_str[:-3]
                            content_str = content_str.strip()
                            # Parse the actual tool call
                            actual_tool_call = json.loads(content_str)
                            return json.dumps(actual_tool_call)
                    except json.JSONDecodeError:
                        pass

            # If it's a simple content response, return it directly without wrapping in tool_calls
            if isinstance(parsed, dict) and "content" in parsed and "tool_calls" not in parsed:
                return str(parsed["content"])
            return content
        except json.JSONDecodeError:
            # If it's not JSON, return the content directly
            return content

    def _convert_messages_to_langchain(self, messages: List[Dict[str, str]]) -> List[BaseMessage]:
        lc_messages = []
        for msg in messages:
            role = msg.get("role", "user").lower()
            content = msg.get("content", "")
            if role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant" or role == "model":
                lc_messages.append(AIMessage(content=content))
            elif role == "system":
                # For Gemini, we'll prepend system messages to the first user message
                if lc_messages and isinstance(lc_messages[0], HumanMessage):
                    lc_messages[0] = HumanMessage(
                        content=f"System Instructions: {content}\n\n{lc_messages[0].content}"
                    )
                else:
                    lc_messages.append(HumanMessage(content=f"System Instructions: {content}"))
            else:
                lc_messages.append(HumanMessage(content=f"{role}: {content}"))
        return lc_messages

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        # Single message invocation
        result = self.llm.invoke([HumanMessage(content=prompt)], **kwargs)
        return self._parse_llm_result(result)

    async def ainvoke(self, prompt: str, **kwargs: Any) -> str:
        result = await self.llm.ainvoke([HumanMessage(content=prompt)], **kwargs)
        return self._parse_llm_result(result)

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        lc_messages = self._convert_messages_to_langchain(messages)
        result = self.llm.invoke(lc_messages, **kwargs)
        return self._parse_llm_result(result)

    async def achat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        lc_messages = self._convert_messages_to_langchain(messages)
        result = await self.llm.ainvoke(lc_messages, **kwargs)
        return self._parse_llm_result(result)
