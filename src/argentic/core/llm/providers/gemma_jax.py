"""
Gemma JAX Provider - Official Google DeepMind implementation.

Uses official `gemma` library with JAX for multimodal inference.
Supports Gemma 3 models (including 3n E4B) with vision.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

try:
    from gemma import gm
    import jax
    from PIL import Image
    import numpy as np

    _GEMMA_AVAILABLE = True
except ImportError:
    _GEMMA_AVAILABLE = False
    gm = None
    jax = None

from argentic.core.llm.providers.base import ModelProvider
from argentic.core.logger import LogLevel, get_logger
from argentic.core.protocol.chat_message import (
    AssistantMessage,
    ChatMessage,
    LLMChatResponse,
    SystemMessage,
    UserMessage,
)


class GemmaJAXProvider(ModelProvider):
    """Official Google DeepMind Gemma with JAX - multimodal support."""

    def __init__(self, config: Dict[str, Any], messager: Optional[Any] = None):
        super().__init__(config, messager)
        self.logger = get_logger("gemma_jax", LogLevel.INFO)

        if not _GEMMA_AVAILABLE:
            raise ImportError(
                "gemma library not installed. Install: pip install 'jax[cuda12_local]' gemma"
            )

        # Model config
        self.model_size = config.get("gemma_model_size", "4B")
        self.checkpoint_path = config.get("gemma_checkpoint_path")

        # Generation params
        self.multi_turn = config.get("gemma_multi_turn", True)
        self.max_output_tokens = config.get("max_output_tokens", 512)
        self.temperature = config.get("temperature", 0.7)
        self.top_p = config.get("top_p", 0.9)
        self.top_k = config.get("top_k", 40)

        # Thread pool for JAX (blocking operations)
        self._inference_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gemma_jax")

        # Lazy loading
        self._model = None
        self._params = None
        self._sampler = None
        self._initialized = False

        self.logger.info(f"GemmaJAXProvider initialized (size: {self.model_size})")

    async def _initialize_model(self):
        """Lazy load model, params, sampler."""
        if self._initialized:
            return

        self.logger.info("Loading Gemma model with JAX...")

        devices = jax.devices()
        self.logger.info(f"JAX devices: {devices}")

        def _load():
            # Load model architecture
            # Note: Gemma3n uses same architecture as Gemma3, different weights
            if self.model_size in ("1B",):
                self._model = gm.nn.Gemma3_1B()
            elif self.model_size in ("4B", "E4B"):
                self._model = gm.nn.Gemma3_4B()
            elif self.model_size in ("12B",):
                self._model = gm.nn.Gemma3_12B()
            elif self.model_size in ("27B",):
                self._model = gm.nn.Gemma3_27B()
            elif self.model_size in ("E2B",):
                # E2B uses 2B architecture (but no standalone 2B in Gemma 3)
                # Try to use the smallest available
                self._model = gm.nn.Gemma3_1B()  # Fallback
            else:
                raise ValueError(f"Unsupported model size: {self.model_size}")

            # Load params
            if self.checkpoint_path:
                self._params = gm.ckpts.load_params(self.checkpoint_path)
            else:
                # Auto-download via CheckpointPath enum
                # Gemma 3: 1B, 4B, 12B, 27B, 270M (text-only)
                # Gemma 3n: E2B, E4B (multimodal)
                checkpoint_map = {
                    "1B": gm.ckpts.CheckpointPath.GEMMA3_1B_IT,
                    "4B": gm.ckpts.CheckpointPath.GEMMA3_4B_IT,
                    "12B": gm.ckpts.CheckpointPath.GEMMA3_12B_IT,
                    "27B": gm.ckpts.CheckpointPath.GEMMA3_27B_IT,
                    "E2B": gm.ckpts.CheckpointPath.GEMMA3N_E2B_IT,
                    "E4B": gm.ckpts.CheckpointPath.GEMMA3N_E4B_IT,
                }
                checkpoint = checkpoint_map.get(self.model_size)
                if not checkpoint:
                    raise ValueError(f"No default checkpoint for {self.model_size}")

                self.logger.info(f"Auto-downloading checkpoint: {checkpoint}")
                self._params = gm.ckpts.load_params(checkpoint)

            # Create sampler
            self._sampler = gm.text.ChatSampler(
                model=self._model,
                params=self._params,
                multi_turn=self.multi_turn,
            )

            self.logger.info("✓ Gemma model loaded")

        await asyncio.get_event_loop().run_in_executor(self._inference_pool, _load)
        self._initialized = True

    def _prepare_multimodal_input(
        self, messages: List[ChatMessage]
    ) -> tuple[str, Optional[List[Image.Image]]]:
        """
        Prepare multimodal input from chat messages.

        Returns (prompt, images):
            - prompt: Text with <start_of_image> placeholders
            - images: List of PIL Images or None
        """
        system_prompt = None
        user_messages = []

        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_prompt = msg.content
            elif isinstance(msg, UserMessage):
                user_messages.append(msg)

        if not user_messages:
            return system_prompt or "", None

        # Process last user message
        last_msg = user_messages[-1]

        # Handle multimodal content
        if isinstance(last_msg.content, dict):
            text = last_msg.content.get("text", "")
            images = []

            # Extract images
            if last_msg.content.get("images"):
                for img in last_msg.content["images"]:
                    if isinstance(img, Image.Image):
                        images.append(img)
                    elif isinstance(img, np.ndarray):
                        pil_img = Image.fromarray(img.astype("uint8"))
                        images.append(pil_img)

                # Format prompt with image placeholders
                if images:
                    image_prompts = "\n".join(
                        [f"Image {i+1}: <start_of_image>" for i in range(len(images))]
                    )
                    text = f"{text}\n\n{image_prompts}"

            # Add system prompt
            if system_prompt:
                text = f"{system_prompt}\n\n{text}"

            return text, images if images else None

        else:
            # Simple text
            text = last_msg.content
            if system_prompt:
                text = f"{system_prompt}\n\n{text}"
            return text, None

    async def _generate(self, prompt: str, images: Optional[List[Image.Image]] = None) -> str:
        """Generate response using Gemma sampler."""
        await self._initialize_model()

        def _inference():
            if images:
                response = self._sampler.chat(prompt, images=images)
            else:
                response = self._sampler.chat(prompt)
            return response

        response = await asyncio.get_event_loop().run_in_executor(self._inference_pool, _inference)

        return response

    async def ainvoke(self, prompt: str, **kwargs: Any) -> LLMChatResponse:
        """Async invoke with single prompt."""
        messages = [UserMessage(role="user", content=prompt)]
        return await self.achat(messages, **kwargs)

    def invoke(self, prompt: str, **kwargs: Any) -> LLMChatResponse:
        """Sync invoke with single prompt."""
        return asyncio.run(self.ainvoke(prompt, **kwargs))

    async def achat(
        self, messages: List, tools: Optional[List[Any]] = None, **kwargs: Any
    ) -> LLMChatResponse:
        """Async invoke with chat messages."""
        # Convert dicts to ChatMessage
        chat_messages: List[ChatMessage] = []
        for m in messages:
            if isinstance(m, (SystemMessage, UserMessage, AssistantMessage)):
                chat_messages.append(m)
            elif isinstance(m, dict):
                role = m.get("role", "user")
                content = m.get("content", "")
                if role == "system":
                    chat_messages.append(SystemMessage(role="system", content=content))
                elif role == "user":
                    chat_messages.append(UserMessage(role="user", content=content))
                elif role == "assistant":
                    chat_messages.append(AssistantMessage(role="assistant", content=content))
            else:
                raise TypeError(f"Unsupported message type: {type(m)}")

        # Prepare multimodal input
        prompt, images = self._prepare_multimodal_input(chat_messages)

        # Generate
        try:
            response_text = await self._generate(prompt, images)

            return LLMChatResponse(
                message=AssistantMessage(role="assistant", content=response_text),
                usage=None,
            )
        except Exception as e:
            self.logger.error(f"Error during generation: {e}")
            raise

    def chat(
        self, messages: List, tools: Optional[List[Any]] = None, **kwargs: Any
    ) -> LLMChatResponse:
        """Sync invoke with chat messages."""
        return asyncio.run(self.achat(messages, tools=tools, **kwargs))

    def supports_tools(self) -> bool:
        """Check if model supports tool calling."""
        return False

    def supports_multimodal(self) -> bool:
        """Check if model supports multimodal input."""
        return True

    def get_model_name(self) -> str:
        """Get model name."""
        return f"gemma-3-{self.model_size.lower()}-it"

    def _encode_image(self, image: np.ndarray) -> Image.Image:
        """
        Encode numpy image array to PIL Image format for Gemma.

        Args:
            image: Numpy array representing an image (H, W, C) or (H, W)

        Returns:
            PIL Image object
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(image)}")

        # Handle grayscale images (H, W) -> (H, W, 1)
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)

        # Ensure uint8 type
        if image.dtype != np.uint8:
            # Normalize to 0-255 range if needed
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        # Convert to PIL Image
        pil_image = Image.fromarray(image)
        return pil_image

    def _encode_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Encode audio numpy array for Gemma processing.

        Args:
            audio: Numpy array representing audio samples

        Returns:
            Processed numpy array suitable for Gemma
        """
        if not isinstance(audio, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(audio)}")

        # Ensure float32 type for audio processing
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Normalize audio to [-1, 1] range if needed
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val

        return audio

    async def stop(self):
        """Stop provider and cleanup."""
        self._initialized = False
        self._model = None
        self._params = None
        self._sampler = None
        self.logger.info("GemmaJAXProvider stopped")

    def __del__(self):
        """Cleanup thread pool."""
        if hasattr(self, "_inference_pool"):
            self._inference_pool.shutdown(wait=False)
