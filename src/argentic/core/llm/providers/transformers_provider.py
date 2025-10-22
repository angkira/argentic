import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

try:
    from transformers import AutoModelForCausalLM, AutoProcessor
    import torch

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False
    AutoModelForCausalLM = None
    AutoProcessor = None
    torch = None

try:
    import numpy as np
    from PIL import Image

    _NUMPY_AVAILABLE = True
    _PIL_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False
    _PIL_AVAILABLE = False
    np = None
    Image = None

from argentic.core.llm.providers.base import ModelProvider
from argentic.core.logger import LogLevel, get_logger
from argentic.core.protocol.chat_message import (
    AssistantMessage,
    ChatMessage,
    LLMChatResponse,
    SystemMessage,
    UserMessage,
)


class TransformersProvider(ModelProvider):
    """
    Universal Hugging Face Transformers provider with multimodal support.

    Supports any model from transformers library including:
    - Text models (GPT, LLaMA, Mistral, etc.)
    - Multimodal models (Gemma 3n, LLaVA, Qwen-VL, etc.)
    - Vision-language models with AutoProcessor

    Uses PyTorch with proper threading to avoid blocking.

    Threading strategy:
    - Model loading in thread pool (heavyweight operation)
    - Inference in thread pool (CPU/GPU bound)
    - All async operations properly awaited

    Config parameters:
        - hf_model_id: HF model ID (e.g., "google/gemma-3n-E4B-it")
        - hf_model_path: Optional local path to downloaded model
        - hf_device: Device to run on ("cuda", "cpu", "auto")
        - hf_torch_dtype: torch dtype ("float16", "bfloat16", "float32", "auto")
        - hf_trust_remote_code: Whether to trust remote code (default: True)
        - max_new_tokens: Maximum tokens to generate (default: 512)
        - temperature: Sampling temperature (default: 0.7)
        - top_p: Nucleus sampling (default: 0.9)
        - do_sample: Whether to use sampling (default: True)
    """

    def __init__(self, config: Dict[str, Any], messager: Optional[Any] = None):
        super().__init__(config, messager)
        self.logger = get_logger("transformers_provider", LogLevel.INFO)

        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library is not installed. "
                "Install with: pip install transformers torch"
            )

        # Model configuration
        self.model_id = config.get("hf_model_id") or config.get(
            "gemma_model_id", "google/gemma-3n-E4B-it"
        )
        self.model_path = config.get("hf_model_path") or config.get(
            "gemma_model_path"
        )  # Optional local path
        self.device = config.get("hf_device") or config.get("gemma_device", "auto")

        # Parse torch dtype
        dtype_str = config.get("hf_torch_dtype") or config.get("gemma_torch_dtype", "float16")
        if dtype_str == "auto":
            self.torch_dtype = "auto"
        else:
            self.torch_dtype = getattr(torch, dtype_str, torch.float16)

        self.trust_remote_code = config.get("hf_trust_remote_code", True)

        # Generation parameters
        self.max_new_tokens = config.get("max_new_tokens", 512)
        self.temperature = config.get("temperature", 0.7)
        self.top_p = config.get("top_p", 0.9)
        self.do_sample = config.get("do_sample", True)

        # Thread pool for model operations
        self._inference_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gemma")

        # Model and processor (loaded lazily)
        self._model = None
        self._processor = None
        self._initialized = False

        self.logger.info(f"TransformersProvider initialized with model: {self.model_id}")

    async def _initialize_model(self):
        """Lazy load model and processor in thread pool."""
        if self._initialized:
            return

        self.logger.info(f"Loading model: {self.model_id}...")

        # Check CUDA availability
        if torch.cuda.is_available():
            self.logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            self.logger.info(
                f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
            )
        else:
            self.logger.warning("CUDA not available, using CPU (will be slow!)")

        def _load():
            # Determine model path - validate if local path provided
            if self.model_path:
                from pathlib import Path

                model_path_obj = Path(self.model_path)
                # Check if local path exists and is valid
                if model_path_obj.exists() and any(model_path_obj.iterdir()):
                    model_path = str(model_path_obj)
                    self.logger.info(f"Using local model path: {model_path}")
                else:
                    self.logger.warning(
                        f"Local path {self.model_path} empty/invalid, using model_id from HF"
                    )
                    model_path = self.model_id
            else:
                model_path = self.model_id

            # Try to load processor (for multimodal models)
            try:
                self._processor = AutoProcessor.from_pretrained(
                    model_path, trust_remote_code=self.trust_remote_code
                )
                self.logger.info("✓ Loaded AutoProcessor (multimodal support)")
            except Exception as e:
                self.logger.debug(f"AutoProcessor failed: {e}, trying AutoTokenizer...")
                # Fall back to tokenizer for text-only models
                from transformers import AutoTokenizer

                self._processor = AutoTokenizer.from_pretrained(
                    model_path, trust_remote_code=self.trust_remote_code
                )
                self.logger.info("✓ Loaded AutoTokenizer (text-only)")

            # Load model (FP16/BF16 for Gemma 3n - designed for edge devices)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=self.torch_dtype,
                device_map="auto",
                trust_remote_code=self.trust_remote_code,
            )

            # Set to eval mode
            self._model.eval()

            # Log device info
            device_info = str(self._model.device)
            if hasattr(self._model, "hf_device_map"):
                device_info = f"{device_info} (device_map: {self._model.hf_device_map})"
            self.logger.info(f"✓ Model loaded on: {device_info}")

        await asyncio.get_event_loop().run_in_executor(self._inference_pool, _load)
        self._initialized = True
        self.logger.info("Model ready for inference")

    def _prepare_inputs(self, messages: List[ChatMessage]) -> Dict[str, Any]:
        """
        Prepare inputs for transformers model from chat messages.

        Returns dict with 'text' and optionally 'images' keys.
        """
        # Extract system prompt
        system_prompt = None
        user_messages = []

        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_prompt = msg.content
            elif isinstance(msg, UserMessage):
                user_messages.append(msg)

        # For now, process last user message (can be extended for multi-turn)
        if not user_messages:
            return {"text": system_prompt or ""}

        last_msg = user_messages[-1]

        # Handle multimodal content
        if isinstance(last_msg.content, dict):
            result = {"text": last_msg.content.get("text", "")}

            # Add system prompt if present
            if system_prompt:
                result["text"] = f"{system_prompt}\n\n{result['text']}"

            # Add images if present
            if last_msg.content.get("images"):
                images = []
                for img in last_msg.content["images"]:
                    if isinstance(img, Image.Image):
                        images.append(img)
                    elif isinstance(img, np.ndarray):
                        # Convert numpy to PIL
                        pil_img = Image.fromarray(img.astype("uint8"))
                        images.append(pil_img)
                result["images"] = images

            # TODO: Add audio support when available

            return result
        else:
            # Simple text
            text = last_msg.content
            if system_prompt:
                text = f"{system_prompt}\n\n{text}"
            return {"text": text}

    async def _generate(self, inputs: Dict[str, Any]) -> str:
        """Generate response using the model."""
        await self._initialize_model()

        def _inference():
            # Prepare inputs using processor
            if "images" in inputs:
                # Multimodal: text + images
                model_inputs = self._processor(
                    text=inputs["text"], images=inputs["images"], return_tensors="pt"
                ).to(self._model.device)
            else:
                # Text only
                model_inputs = self._processor(text=inputs["text"], return_tensors="pt").to(
                    self._model.device
                )

            # Generate
            with torch.no_grad():
                outputs = self._model.generate(
                    **model_inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature if self.do_sample else None,
                    top_p=self.top_p if self.do_sample else None,
                    do_sample=self.do_sample,
                )

            # Decode (skip input tokens)
            input_length = model_inputs["input_ids"].shape[1]
            generated_ids = outputs[0][input_length:]
            generated_text = self._processor.decode(generated_ids, skip_special_tokens=True)

            return generated_text.strip()

        response = await asyncio.get_event_loop().run_in_executor(self._inference_pool, _inference)

        return response

    async def ainvoke(self, prompt: str, **kwargs: Any) -> LLMChatResponse:
        """Asynchronously invoke the model with a single prompt."""
        messages = [UserMessage(role="user", content=prompt)]
        return await self.achat(messages, **kwargs)

    def invoke(self, prompt: str, **kwargs: Any) -> LLMChatResponse:
        """Synchronously invoke the model with a single prompt."""
        return asyncio.run(self.ainvoke(prompt, **kwargs))

    async def achat(
        self, messages: List, tools: Optional[List[Any]] = None, **kwargs: Any
    ) -> LLMChatResponse:
        """Asynchronously invoke the model with chat messages."""
        # Convert dicts to ChatMessage if needed
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

        # Prepare inputs
        inputs = self._prepare_inputs(chat_messages)

        # Generate
        try:
            response_text = await self._generate(inputs)

            return LLMChatResponse(
                message=AssistantMessage(role="assistant", content=response_text),
                usage=None,  # TODO: Track token usage
            )
        except Exception as e:
            self.logger.error(f"Error during generation: {e}")
            raise

    def chat(
        self, messages: List, tools: Optional[List[Any]] = None, **kwargs: Any
    ) -> LLMChatResponse:
        """Synchronously invoke the model with chat messages."""
        return asyncio.run(self.achat(messages, tools=tools, **kwargs))

    def supports_tools(self) -> bool:
        """Check if the model supports tool calling."""
        return False  # Most HF models don't support tool calling natively

    def get_model_name(self) -> str:
        """Get the model name."""
        return self.model_id

    def __del__(self):
        """Cleanup thread pool on deletion."""
        if hasattr(self, "_inference_pool"):
            self._inference_pool.shutdown(wait=False)
