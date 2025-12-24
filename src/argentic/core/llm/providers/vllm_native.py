"""
Native vLLM Provider using vLLM's Python API

This provider uses vLLM's LLM class directly (not the OpenAI-compatible server).
Supports:
- Multimodal inputs (images + text)
- Pre-computed image embeddings (with enable_mm_embeds=True)
- Offline inference without HTTP server
- Direct GPU access

For OpenAI-compatible API server, use VLLMProvider instead.
"""

import asyncio
from typing import Any, Dict, List, Optional

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None

try:
    from vllm import LLM, SamplingParams

    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None

import numpy as np
from PIL import Image

from argentic.core.llm.providers.base import ModelProvider
from argentic.core.logger import LogLevel, get_logger
from argentic.core.protocol.chat_message import (
    AssistantMessage,
    LLMChatResponse,
    SystemMessage,
    UserMessage,
)


class VLLMNativeProvider(ModelProvider):
    """
    Native vLLM provider using vLLM's Python API (LLM class).

    This provider loads models directly into memory and runs inference locally.
    Supports pre-computed image embeddings via multi_modal_data parameter.

    Requirements:
        - vllm library: pip install vllm
        - torch library: pip install torch

    Config Example:
        {
            "llm": {
                "provider": "vllm_native",
                "model_name": "llava-hf/llava-1.5-7b-hf",
                "tensor_parallel_size": 1,
                "enable_mm_embeds": true,  # Enable image embedding support
                "temperature": 0.7,
                "max_tokens": 2048,
                "gpu_memory_utilization": 0.9,
            }
        }

    Features:
        - Direct model loading (no HTTP server needed)
        - Pre-computed image embeddings support
        - Raw image input support
        - Batch inference
        - GPU memory management
    """

    def __init__(self, config: Dict[str, Any], messager: Optional[Any] = None):
        """Initialize native vLLM provider."""
        super().__init__(config, messager)

        if not _VLLM_AVAILABLE:
            raise ImportError(
                "vllm is required for VLLMNativeProvider. "
                "Install with: pip install vllm"
            )

        if not _TORCH_AVAILABLE:
            raise ImportError(
                "torch is required for VLLMNativeProvider. "
                "Install with: pip install torch"
            )

        self.model_name = config.get("model_name")
        if not self.model_name:
            raise ValueError("model_name is required")

        # vLLM-specific configuration
        self.tensor_parallel_size = config.get("tensor_parallel_size", 1)
        self.gpu_memory_utilization = config.get("gpu_memory_utilization", 0.9)
        self.enable_mm_embeds = config.get("enable_mm_embeds", False)
        self.trust_remote_code = config.get("trust_remote_code", False)

        # Sampling parameters
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 2048)
        self.top_p = config.get("top_p", 1.0)
        self.top_k = config.get("top_k", -1)

        self.logger = get_logger(
            __name__, level=config.get("log_level", LogLevel.INFO)
        )

        # Initialize LLM (this loads the model)
        self.logger.info(f"Loading vLLM model: {self.model_name}...")
        self.logger.info(
            f"Config: tensor_parallel_size={self.tensor_parallel_size}, "
            f"gpu_memory_utilization={self.gpu_memory_utilization}, "
            f"enable_mm_embeds={self.enable_mm_embeds}"
        )

        try:
            self.llm = LLM(
                model=self.model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                trust_remote_code=self.trust_remote_code,
            )
            self.logger.info("✓ vLLM model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load vLLM model: {e}")
            raise

    def _prepare_vllm_inputs(
        self, messages: List
    ) -> List[Dict[str, Any]]:
        """
        Prepare inputs for vLLM's native API.

        Returns list of dicts with 'prompt' and optionally 'multi_modal_data'.

        Args:
            messages: Argentic chat messages

        Returns:
            List of vLLM input dicts

        Each dict can have:
            - prompt: str - the text prompt
            - multi_modal_data: dict - multimodal inputs
                - image: PIL.Image | torch.Tensor | List[PIL.Image]
        """
        # Extract system and user messages
        system_prompt = None
        user_messages = []

        for msg in messages:
            if isinstance(msg, dict):
                if msg.get("role") == "system":
                    system_prompt = msg.get("content")
                elif msg.get("role") == "user":
                    user_messages.append(msg.get("content"))
            elif isinstance(msg, SystemMessage):
                system_prompt = msg.content
            elif isinstance(msg, UserMessage):
                user_messages.append(msg.content)

        if not user_messages:
            return []

        # Process last user message (vLLM typically uses last message)
        last_msg = user_messages[-1]

        # Build vLLM input
        vllm_input = {}

        # Handle multimodal content
        if isinstance(last_msg, dict):
            text = last_msg.get("text", "")

            # Add system prompt
            if system_prompt:
                text = f"{system_prompt}\n\n{text}"

            vllm_input["prompt"] = text

            # Handle images
            if "images" in last_msg:
                images = []
                for img in last_msg["images"]:
                    if isinstance(img, np.ndarray):
                        # Convert numpy to PIL
                        img = Image.fromarray(img.astype("uint8"))
                    images.append(img)

                if images:
                    vllm_input["multi_modal_data"] = {"image": images if len(images) > 1 else images[0]}

            # Handle pre-computed embeddings
            elif "image_embeddings" in last_msg:
                if not self.enable_mm_embeds:
                    raise ValueError(
                        "Image embeddings provided but enable_mm_embeds=False. "
                        "Set enable_mm_embeds=True in config to use pre-computed embeddings."
                    )

                embeddings = last_msg["image_embeddings"]

                # Convert to torch tensor if needed
                if isinstance(embeddings, np.ndarray):
                    embeddings = torch.from_numpy(embeddings)

                # Ensure correct shape: (num_items, feature_size, hidden_size)
                if len(embeddings.shape) == 2:
                    # Add batch dimension if missing
                    embeddings = embeddings.unsqueeze(0)

                vllm_input["multi_modal_data"] = {"image": embeddings}
                self.logger.debug(f"Using image embeddings: shape={embeddings.shape}")

        else:
            # Simple text message
            text = str(last_msg)
            if system_prompt:
                text = f"{system_prompt}\n\n{text}"
            vllm_input["prompt"] = text

        return [vllm_input]

    async def achat(
        self,
        messages: List,
        system: Optional[str] = None,
        **kwargs,
    ) -> LLMChatResponse:
        """
        Generate response using vLLM's native API.

        Args:
            messages: Chat messages
            system: Optional system message
            **kwargs: Additional sampling parameters

        Returns:
            LLMChatResponse with generated content
        """
        # Add system message if provided
        if system:
            messages = [{"role": "system", "content": system}] + messages

        # Prepare vLLM inputs
        vllm_inputs = self._prepare_vllm_inputs(messages)

        if not vllm_inputs:
            raise ValueError("No valid inputs after message processing")

        # Create sampling params
        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            top_p=kwargs.get("top_p", self.top_p),
            top_k=kwargs.get("top_k", self.top_k),
        )

        # Run inference in thread pool (vLLM is sync)
        def _generate():
            return self.llm.generate(vllm_inputs, sampling_params=sampling_params)

        try:
            outputs = await asyncio.to_thread(_generate)

            # Extract response
            generated_text = outputs[0].outputs[0].text

            # vLLM doesn't provide token usage in the same way
            # We can estimate or return 0
            usage = {
                "prompt_tokens": 0,  # vLLM doesn't expose this easily
                "completion_tokens": 0,
                "total_tokens": 0,
            }

            return LLMChatResponse(
                message=AssistantMessage(content=generated_text),
                usage=usage,
                model=self.model_name,
            )

        except Exception as e:
            self.logger.error(f"vLLM generation failed: {e}", exc_info=True)
            raise

    async def ainvoke(
        self, prompt: str, system: Optional[str] = None, **kwargs
    ) -> LLMChatResponse:
        """
        Invoke with a simple text prompt.

        Args:
            prompt: Text prompt
            system: Optional system message
            **kwargs: Additional parameters

        Returns:
            LLMChatResponse
        """
        messages = [{"role": "user", "content": prompt}]
        return await self.achat(messages, system=system, **kwargs)

    def chat(self, messages: List, **kwargs: Any) -> LLMChatResponse:
        """Synchronous chat."""
        return asyncio.run(self.achat(messages, **kwargs))

    def get_model_name(self) -> str:
        """Get model name."""
        return self.model_name

    def supports_multimodal(self) -> bool:
        """This provider supports multimodal inputs including embeddings."""
        return True

    def supports_embeddings(self) -> bool:
        """Check if this provider supports pre-computed embeddings."""
        return self.enable_mm_embeds
