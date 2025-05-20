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

        llm_config = config.get("llm", {})
        gemini_config = llm_config.get("google_gemini", {})
        default_gen_settings = llm_config.get("default_generation_settings", {})

        self.api_key = os.getenv("GEMINI_API_KEY") or gemini_config.get("google_gemini_api_key")
        self.model_name = gemini_config.get("google_gemini_model_name", "gemini-2.0-flash")

        if not self.api_key:
            raise ValueError(
                "Google Gemini API key not found. Set GEMINI_API_KEY environment variable or google_gemini_api_key in config."
            )

        # Prepare generation_config for ChatGoogleGenerativeAI
        # Start with defaults, then override with Gemini-specific settings
        gen_params: Dict[str, Any] = {**default_gen_settings}
        specific_gen_settings = gemini_config.get("generation_settings", {})
        gen_params.update(specific_gen_settings)

        # Handle max_tokens from default if not overridden by max_output_tokens
        if "max_output_tokens" not in gen_params and "max_tokens" in gen_params:
            gen_params["max_output_tokens"] = gen_params.pop("max_tokens")

        # Rename stop_sequences to stop for Langchain if present
        if "stop_sequences" in gen_params:
            gen_params["stop"] = gen_params.pop("stop_sequences")

        # Filter out any parameters not accepted by ChatGoogleGenerativeAI constructor
        # or not relevant for this provider. Also filter out None values.
        accepted_llm_params = [
            "temperature",
            "top_p",
            "top_k",
            "max_output_tokens",
            "stop",  # Langchain's ChatGoogleGenerativeAI uses 'stop'
            "candidate_count",
        ]

        final_generation_config = {
            k: v for k, v in gen_params.items() if k in accepted_llm_params and v is not None
        }

        self.logger.debug(f"Final generation config for Gemini: {final_generation_config}")

        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name, google_api_key=self.api_key, **final_generation_config
        )
        self.logger.info(
            f"Initialized GoogleGeminiProvider with model: {self.model_name} and config: {final_generation_config}"
        )

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
                return parsed["content"]
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
