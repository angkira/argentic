"""
Generic OpenAI-Compatible API Provider

Works with any service that implements the OpenAI API:
- OpenAI (official)
- vLLM (OpenAI-compatible server)
- OpenRouter
- Together AI
- Anyscale
- And other OpenAI-compatible services
"""

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
    LLMChatResponse,
    SystemMessage,
    UserMessage,
)


class OpenAICompatibleProvider(ModelProvider):
    """
    Generic provider for OpenAI-compatible APIs.

    This provider can connect to any service that implements the OpenAI Chat Completions API:
    - Official OpenAI API (api.openai.com)
    - vLLM server (http://localhost:8000/v1)
    - OpenRouter (https://openrouter.ai/api/v1)
    - Together AI (https://api.together.xyz/v1)
    - And many others

    Config Example:
        {
            "llm": {
                "provider": "openai_compatible",
                "base_url": "http://localhost:8000/v1",  # Required
                "api_key": "your-api-key",  # Required (use "dummy" for vLLM)
                "model_name": "gpt-4",  # Required
                "temperature": 0.7,
                "max_tokens": 2048,
                "timeout": 60.0,
            }
        }

    Note on Visual Embeddings:
        Most OpenAI-compatible APIs do NOT support pre-computed image embeddings.
        They accept:
        - Raw images (base64 or URLs)
        - Text
        But NOT pre-computed visual embeddings (those require native library APIs)
    """

    def __init__(self, config: Dict[str, Any], messager: Optional[Any] = None):
        """Initialize OpenAI-compatible provider."""
        super().__init__(config, messager)

        # Required configuration
        self.base_url = config.get("base_url")
        if not self.base_url:
            raise ValueError(
                "base_url is required for OpenAI-compatible provider. "
                "Examples: 'https://api.openai.com/v1', 'http://localhost:8000/v1', "
                "'https://openrouter.ai/api/v1'"
            )

        self.api_key = config.get("api_key")
        if not self.api_key:
            raise ValueError(
                "api_key is required. Use 'dummy' for services that don't require authentication."
            )

        self.model_name = config.get("model_name")
        if not self.model_name:
            raise ValueError("model_name is required (e.g., 'gpt-4', 'llama-2-7b')")

        # Optional configuration
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 2048)
        self.top_p = config.get("top_p", 1.0)
        self.timeout = config.get("timeout", 60.0)

        # Create async OpenAI client
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )

        self.logger = get_logger(
            __name__, level=config.get("log_level", LogLevel.INFO)
        )
        self.logger.info(
            f"Initialized OpenAI-compatible provider: {self.base_url}, model={self.model_name}"
        )

    def _prepare_messages(self, messages: List) -> List[Dict[str, Any]]:
        """
        Convert Argentic ChatMessages to OpenAI format.

        Handles:
        - Text messages
        - Multimodal messages (text + images)
        - System messages

        Raises:
            NotImplementedError: If pre-computed image_embeddings are detected
        """
        import numpy as np

        openai_messages = []

        for msg in messages:
            # Handle dict messages
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content")

                # Check for pre-computed embeddings (not supported via OpenAI API)
                if isinstance(content, dict) and "image_embeddings" in content:
                    raise NotImplementedError(
                        f"OpenAI-compatible API does not support pre-computed image embeddings. "
                        f"Service: {self.base_url}\n"
                        f"To use image embeddings:\n"
                        f"1. Use the native library API (if available)\n"
                        f"2. Or use raw images instead of embeddings (remove embedding_function)"
                    )

                # Handle multimodal content
                if isinstance(content, dict) and ("images" in content or "text" in content):
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
                    # Simple message
                    openai_messages.append(msg)
                continue

            # Handle ChatMessage objects
            if isinstance(msg, SystemMessage):
                openai_messages.append(
                    {
                        "role": "system",
                        "content": msg.content,
                    }
                )
            elif isinstance(msg, UserMessage):
                # Check for multimodal content
                if isinstance(msg.content, dict):
                    # Check for embeddings
                    if "image_embeddings" in msg.content:
                        raise NotImplementedError(
                            f"OpenAI-compatible API does not support pre-computed image embeddings. "
                            f"Service: {self.base_url}"
                        )

                    # Handle images
                    if "images" in msg.content:
                        text = msg.content.get("text", "")
                        images = msg.content.get("images", [])

                        content_parts = [{"type": "text", "text": text}]

                        for img in images:
                            if isinstance(img, Image.Image):
                                buffered = BytesIO()
                                img.save(buffered, format="PNG")
                                img_b64 = base64.b64encode(buffered.getvalue()).decode()
                                img_url = f"data:image/png;base64,{img_b64}"
                            else:
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
                        # Dict without images
                        openai_messages.append(
                            {
                                "role": "user",
                                "content": msg.content.get("text", str(msg.content)),
                            }
                        )
                else:
                    # Simple text
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
        messages: List,
        system: Optional[str] = None,
        **kwargs,
    ) -> LLMChatResponse:
        """
        Call the OpenAI-compatible API with chat messages.

        Args:
            messages: List of chat messages
            system: Optional system message
            **kwargs: Additional parameters to pass to the API

        Returns:
            LLMChatResponse with generated content

        Raises:
            NotImplementedError: If pre-computed embeddings are detected
        """
        # Add system message if provided
        if system:
            messages = [{"role": "system", "content": system}] + messages

        # Prepare messages in OpenAI format
        openai_messages = self._prepare_messages(messages)

        # Call API
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=openai_messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                top_p=kwargs.get("top_p", self.top_p),
                **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens", "top_p"]},
            )

            # Extract response
            content = response.choices[0].message.content
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

            return LLMChatResponse(
                message=AssistantMessage(content=content),
                usage=usage,
                model=response.model,
            )

        except Exception as e:
            self.logger.error(f"API call failed: {e}")
            raise

    async def ainvoke(
        self, prompt: str, system: Optional[str] = None, **kwargs
    ) -> LLMChatResponse:
        """
        Invoke model with a simple text prompt.

        Args:
            prompt: Text prompt
            system: Optional system message
            **kwargs: Additional parameters

        Returns:
            LLMChatResponse with generated content
        """
        messages = [{"role": "user", "content": prompt}]
        return await self.achat(messages, system=system, **kwargs)

    def chat(self, messages: List, **kwargs: Any) -> LLMChatResponse:
        """Synchronous chat (runs async in event loop)."""
        return asyncio.run(self.achat(messages, **kwargs))

    def get_model_name(self) -> str:
        """Get the model name."""
        return self.model_name

    def supports_multimodal(self) -> bool:
        """
        Check if multimodal is supported.

        Note: This returns True for image support, but NOT for pre-computed embeddings.
        OpenAI-compatible APIs typically support images but not pre-computed embeddings.
        """
        return True
