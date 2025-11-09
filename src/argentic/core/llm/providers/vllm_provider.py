"""vLLM Provider for Argentic - OpenAI-compatible API."""

import asyncio
import base64
from io import BytesIO
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from PIL import Image

from argentic.core.llm.providers.base import ModelProvider
from argentic.core.logger import LogLevel, get_logger
from argentic.core.protocol.chat_message import (
    AssistantMessage,
    ChatMessage,
    LLMChatResponse,
    SystemMessage,
    UserMessage,
)


class VLLMProvider(ModelProvider):
    """
    vLLM provider using OpenAI-compatible API.

    vLLM serves models with OpenAI-compatible endpoints, supporting:
    - Text generation (chat completions)
    - Multimodal models (text + images)
    - Tool calling
    - Streaming

    Setup:
        1. Install vLLM: pip install vllm
        2. Start server: vllm serve google/gemma-3n-E4B-it --port 8000
        3. Configure provider with base_url pointing to server

    Config Example:
        {
            "llm": {
                "provider": "vllm",
                "vllm_base_url": "http://localhost:8000/v1",
                "vllm_model_name": "google/gemma-3n-E4B-it",
                "vllm_api_key": "dummy",  # vLLM doesn't require real key
                "temperature": 0.7,
                "max_tokens": 2048,
            }
        }
    """

    def __init__(self, config: Dict[str, Any], messager: Optional[Any] = None):
        """Initialize vLLM provider with OpenAI client."""
        super().__init__(config, messager)

        self.base_url = config.get("vllm_base_url", "http://localhost:8000/v1")
        self.api_key = config.get("vllm_api_key", "dummy")  # vLLM doesn't check

        # Create async OpenAI client with longer timeout for vision models
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=120.0,  # 2 minutes for vision models
        )

        # Auto-detect model name if not provided
        self.model_name = config.get("vllm_model_name")
        if not self.model_name:
            self.model_name = self._fetch_first_model()

        # Generation parameters
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 2048)
        self.top_p = config.get("top_p", 1.0)
        self.frequency_penalty = config.get("frequency_penalty", 0.0)
        self.presence_penalty = config.get("presence_penalty", 0.0)

        self.logger = get_logger("vllm", LogLevel.INFO)
        self.logger.info(f"VLLMProvider initialized: {self.base_url} (model: {self.model_name})")

    def _fetch_first_model(self) -> str:
        """
        Fetch the first available model from vLLM server.

        Returns:
            Model name/ID from the server
        """
        import httpx

        try:
            # Make synchronous request to /v1/models endpoint
            models_url = f"{self.base_url.rstrip('/v1')}/v1/models"
            with httpx.Client(timeout=10.0) as client:
                response = client.get(models_url)
                response.raise_for_status()
                models_data = response.json()

                if models_data.get("data") and len(models_data["data"]) > 0:
                    model_id = models_data["data"][0]["id"]
                    return model_id
                else:
                    raise ValueError("No models found on vLLM server")

        except Exception as e:
            # Fall back to default if auto-detection fails
            fallback = "google/gemma-3n-E4B-it"
            print(f"Warning: Could not auto-detect model from {models_url}: {e}")
            print(f"Using fallback model: {fallback}")
            return fallback

    def get_model_name(self) -> str:
        """Return the model name."""
        return f"vllm:{self.model_name}"

    def invoke(self, prompt: str, **kwargs: Any) -> LLMChatResponse:
        """Synchronously invoke the model with a single prompt."""
        return asyncio.run(self.ainvoke(prompt, **kwargs))

    async def ainvoke(self, prompt: str, **kwargs: Any) -> LLMChatResponse:
        """Asynchronously invoke the model with a single prompt."""
        messages = [UserMessage(content=prompt)]
        return await self.achat(messages, **kwargs)

    def chat(self, messages: List, **kwargs: Any) -> LLMChatResponse:
        """Synchronously invoke the model with a list of chat messages."""
        return asyncio.run(self.achat(messages, **kwargs))

    def _prepare_messages(self, messages: List) -> List[Dict[str, Any]]:
        """
        Convert argentic ChatMessages (or dicts) to OpenAI format.

        Handles multimodal content (text + images).
        """
        import numpy as np

        openai_messages = []

        for msg in messages:
            # If it's a dict, check if it has multimodal content
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content")

                # Check if content is a multimodal dict with text and images
                if isinstance(content, dict) and ("images" in content or "text" in content):
                    # Multimodal message
                    text = content.get("text", "")
                    images = content.get("images", [])

                    # Build content parts
                    content_parts = [{"type": "text", "text": text}]

                    for img in images:
                        # Convert numpy array to PIL Image if needed
                        if isinstance(img, np.ndarray):
                            img = Image.fromarray(img)

                        # Convert PIL Image to base64
                        if isinstance(img, Image.Image):
                            buffered = BytesIO()
                            img.save(buffered, format="PNG")
                            img_b64 = base64.b64encode(buffered.getvalue()).decode()
                            img_url = f"data:image/png;base64,{img_b64}"
                        else:
                            # Assume it's already a URL or base64 string
                            img_url = str(img)

                        content_parts.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": img_url},
                            }
                        )

                    openai_messages.append(
                        {
                            "role": role,
                            "content": content_parts,
                        }
                    )
                else:
                    # Simple dict message, use as-is
                    openai_messages.append(msg)
                continue

            if isinstance(msg, SystemMessage):
                openai_messages.append(
                    {
                        "role": "system",
                        "content": msg.content,
                    }
                )
            elif isinstance(msg, UserMessage):
                # Check for multimodal content
                if hasattr(msg, "images") and msg.images:
                    # Multimodal: text + images
                    content_parts = [{"type": "text", "text": msg.content}]

                    for img in msg.images:
                        # Convert PIL Image to base64
                        if isinstance(img, Image.Image):
                            buffered = BytesIO()
                            img.save(buffered, format="PNG")
                            img_b64 = base64.b64encode(buffered.getvalue()).decode()
                            img_url = f"data:image/png;base64,{img_b64}"
                        else:
                            # Assume it's already a URL or base64 string
                            img_url = str(img)

                        content_parts.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": img_url},
                            }
                        )

                    openai_messages.append(
                        {
                            "role": "user",
                            "content": content_parts,
                        }
                    )
                else:
                    # Text only
                    openai_messages.append(
                        {
                            "role": "user",
                            "content": msg.content,
                        }
                    )
            elif isinstance(msg, AssistantMessage):
                openai_messages.append(
                    {
                        "role": "assistant",
                        "content": msg.content,
                    }
                )

        return openai_messages

    async def achat(
        self,
        messages: List,  # Accept both ChatMessage and dict
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> LLMChatResponse:
        """
        Generate chat completion via vLLM OpenAI API.

        Args:
            messages: List of chat messages
            tools: Optional tool definitions (function calling)
            **kwargs: Additional generation parameters

        Returns:
            LLMChatResponse with generated text
        """
        try:
            # Prepare messages
            openai_messages = self._prepare_messages(messages)

            # Merge generation parameters
            gen_params = {
                "model": self.model_name,
                "messages": openai_messages,
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "top_p": kwargs.get("top_p", self.top_p),
                "frequency_penalty": kwargs.get("frequency_penalty", self.frequency_penalty),
                "presence_penalty": kwargs.get("presence_penalty", self.presence_penalty),
            }

            # Debug logging
            self.logger.debug(f"Sending to vLLM: {len(openai_messages)} messages, max_tokens={gen_params['max_tokens']}")

            # Add tools if provided
            if tools:
                gen_params["tools"] = tools

            # Call vLLM via OpenAI API
            response = await self.client.chat.completions.create(**gen_params)

            # Extract response
            choice = response.choices[0]
            content = choice.message.content or ""

            # Extract token usage if available
            usage_dict = None
            if hasattr(response, "usage") and response.usage:
                usage_dict = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
                self.logger.debug(f"Token usage: {usage_dict}")

            # Debug logging
            self.logger.debug(f"vLLM raw response - content: '{content}', finish_reason: {choice.finish_reason}")
            if not content:
                self.logger.warning(f"Empty response from vLLM. Full choice: {choice}")

            # Handle tool calls if present
            tool_calls = None
            if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
                tool_calls = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in choice.message.tool_calls
                ]

            return LLMChatResponse(
                message=AssistantMessage(role="assistant", content=content),
                usage=usage_dict,
                finish_reason=choice.finish_reason,
            )

        except Exception as e:
            self.logger.error(f"Error during vLLM generation: {e}")
            raise

    async def astream_chat(
        self,
        messages: List,  # Accept both ChatMessage and dict
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ):
        """
        Stream chat completion via vLLM OpenAI API.

        Yields:
            LLMChatResponse chunks
        """
        try:
            # Prepare messages
            openai_messages = self._prepare_messages(messages)

            # Merge generation parameters
            gen_params = {
                "model": self.model_name,
                "messages": openai_messages,
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "top_p": kwargs.get("top_p", self.top_p),
                "frequency_penalty": kwargs.get("frequency_penalty", self.frequency_penalty),
                "presence_penalty": kwargs.get("presence_penalty", self.presence_penalty),
                "stream": True,
            }

            # Add tools if provided
            if tools:
                gen_params["tools"] = tools

            # Stream from vLLM
            stream = await self.client.chat.completions.create(**gen_params)

            async for chunk in stream:
                delta = chunk.choices[0].delta

                if delta.content:
                    yield LLMChatResponse(
                        message=AssistantMessage(role="assistant", content=delta.content),
                        finish_reason=chunk.choices[0].finish_reason,
                    )

                # Handle tool calls in streaming
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    tool_calls = [
                        {
                            "id": tc.id if hasattr(tc, "id") else None,
                            "type": tc.type if hasattr(tc, "type") else "function",
                            "function": {
                                "name": tc.function.name if hasattr(tc.function, "name") else "",
                                "arguments": (
                                    tc.function.arguments
                                    if hasattr(tc.function, "arguments")
                                    else ""
                                ),
                            },
                        }
                        for tc in delta.tool_calls
                    ]

                    yield LLMChatResponse(
                        message=AssistantMessage(role="assistant", content=""),
                        finish_reason=chunk.choices[0].finish_reason,
                    )

        except Exception as e:
            self.logger.error(f"Error during vLLM streaming: {e}")
            raise
