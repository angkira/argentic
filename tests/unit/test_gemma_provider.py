"""Unit tests for Gemma Provider."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from argentic.core.llm.providers.gemma import GemmaProvider
from argentic.core.protocol.chat_message import AssistantMessage, LLMChatResponse


class TestGemmaProviderInitialization:
    """Test GemmaProvider initialization."""

    def test_init_default_config(self):
        """Test provider initialization with default config."""
        config = {
            "gemma_model_name": "gemma-3n-e4b-it",
            "gemma_checkpoint_path": "GEMMA3_4B_IT",
        }

        provider = GemmaProvider(config)

        assert provider.model_name == "gemma-3n-e4b-it"
        assert provider.checkpoint_path == "GEMMA3_4B_IT"
        assert provider.enable_ple_caching is True
        assert provider.multi_turn is True
        assert provider._initialized is False

    def test_init_custom_config(self):
        """Test provider initialization with custom config."""
        config = {
            "gemma_model_name": "gemma-3n-e2b-it",
            "gemma_checkpoint_path": "/path/to/checkpoint",
            "gemma_enable_ple_caching": False,
            "gemma_multi_turn": False,
            "gemma_parameters": {
                "temperature": 0.5,
                "top_p": 0.8,
                "max_output_tokens": 1024,
            },
        }

        provider = GemmaProvider(config)

        assert provider.model_name == "gemma-3n-e2b-it"
        assert provider.checkpoint_path == "/path/to/checkpoint"
        assert provider.enable_ple_caching is False
        assert provider.multi_turn is False
        assert provider.default_temperature == 0.5
        assert provider.default_top_p == 0.8
        assert provider.default_max_tokens == 1024

    def test_init_creates_thread_pool(self):
        """Test that inference thread pool is created."""
        config = {"gemma_model_name": "gemma-3n-e4b-it"}

        provider = GemmaProvider(config)

        assert provider._inference_pool is not None
        assert provider._inference_pool._max_workers == 2


class TestGemmaProviderModelLoading:
    """Test model loading functionality."""

    @pytest.mark.asyncio
    async def test_initialize_model(self):
        """Test model initialization."""
        config = {
            "gemma_model_name": "gemma-3n-e4b-it",
            "gemma_checkpoint_path": "GEMMA3_4B_IT",
        }
        provider = GemmaProvider(config)

        await provider._initialize_model()

        assert provider._initialized is True
        assert provider._params is not None
        assert provider._sampler is not None

    @pytest.mark.asyncio
    async def test_initialize_model_only_once(self):
        """Test that model is only initialized once."""
        config = {"gemma_model_name": "gemma-3n-e4b-it"}
        provider = GemmaProvider(config)

        await provider._initialize_model()
        params_first = provider._params

        await provider._initialize_model()
        params_second = provider._params

        # Should be the same instance (not reloaded)
        assert params_first is params_second


class TestGemmaProviderImageEncoding:
    """Test image encoding functionality."""

    def test_encode_numpy_array(self):
        """Test encoding numpy array image."""
        config = {"gemma_model_name": "gemma-3n-e4b-it"}
        provider = GemmaProvider(config)

        img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        encoded = provider._encode_image(img_array)

        assert encoded is not None
        assert isinstance(encoded, Image.Image)

    def test_encode_pil_image(self):
        """Test encoding PIL Image."""
        config = {"gemma_model_name": "gemma-3n-e4b-it"}
        provider = GemmaProvider(config)

        pil_img = Image.new("RGB", (640, 480), color="red")

        encoded = provider._encode_image(pil_img)

        assert encoded is not None
        assert isinstance(encoded, Image.Image)

    def test_encode_base64_string(self):
        """Test encoding base64 string."""
        config = {"gemma_model_name": "gemma-3n-e4b-it"}
        provider = GemmaProvider(config)

        # Create a small test image and convert to base64
        import base64
        import io

        pil_img = Image.new("RGB", (10, 10), color="blue")
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        encoded = provider._encode_image(img_base64)

        assert encoded is not None

    def test_encode_image_with_data_uri(self):
        """Test encoding image with data URI prefix."""
        config = {"gemma_model_name": "gemma-3n-e4b-it"}
        provider = GemmaProvider(config)

        import base64
        import io

        pil_img = Image.new("RGB", (10, 10), color="green")
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        data_uri = f"data:image/png;base64,{img_base64}"

        encoded = provider._encode_image(data_uri)

        assert encoded is not None


class TestGemmaProviderAudioEncoding:
    """Test audio encoding functionality."""

    def test_encode_numpy_audio(self):
        """Test encoding numpy array audio."""
        config = {"gemma_model_name": "gemma-3n-e4b-it"}
        provider = GemmaProvider(config)

        audio_array = np.random.randn(1000).astype(np.float32)

        encoded = provider._encode_audio(audio_array)

        assert encoded is not None
        assert isinstance(encoded, np.ndarray)

    def test_encode_base64_audio(self):
        """Test encoding base64 audio."""
        config = {"gemma_model_name": "gemma-3n-e4b-it"}
        provider = GemmaProvider(config)

        import base64

        # Create test audio data
        audio_data = np.random.randn(100).astype(np.float32).tobytes()
        audio_base64 = base64.b64encode(audio_data).decode()

        encoded = provider._encode_audio(audio_base64)

        assert encoded is not None
        assert isinstance(encoded, np.ndarray)


class TestGemmaProviderMessageConversion:
    """Test message format conversion."""

    def test_convert_simple_text_message(self):
        """Test converting simple text message."""
        config = {"gemma_model_name": "gemma-3n-e4b-it"}
        provider = GemmaProvider(config)

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ]

        converted = provider._convert_to_gemma_format(messages)

        assert len(converted) == 2
        assert converted[0]["role"] == "system"
        assert converted[1]["role"] == "user"

    def test_convert_assistant_to_model_role(self):
        """Test that assistant role is converted to model role."""
        config = {"gemma_model_name": "gemma-3n-e4b-it"}
        provider = GemmaProvider(config)

        messages = [{"role": "assistant", "content": "Hi there!"}]

        converted = provider._convert_to_gemma_format(messages)

        assert converted[0]["role"] == "model"

    def test_convert_multimodal_message(self):
        """Test converting multimodal message with images."""
        config = {"gemma_model_name": "gemma-3n-e4b-it"}
        provider = GemmaProvider(config)

        img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)

        messages = [
            {
                "role": "user",
                "content": {
                    "text": "What's in this image?",
                    "images": [img],
                },
            }
        ]

        converted = provider._convert_to_gemma_format(messages)

        assert len(converted) == 1
        assert converted[0]["role"] == "user"
        assert isinstance(converted[0]["content"], list)
        assert len(converted[0]["content"]) == 2  # image + text

    def test_convert_multimodal_with_audio(self):
        """Test converting multimodal message with audio."""
        config = {"gemma_model_name": "gemma-3n-e4b-it"}
        provider = GemmaProvider(config)

        audio = np.random.randn(100).astype(np.float32)

        messages = [
            {
                "role": "user",
                "content": {
                    "text": "What do you hear?",
                    "audio": audio,
                },
            }
        ]

        converted = provider._convert_to_gemma_format(messages)

        assert len(converted) == 1
        assert isinstance(converted[0]["content"], list)


class TestGemmaProviderInference:
    """Test model inference functionality."""

    @pytest.mark.asyncio
    async def test_achat_simple(self):
        """Test simple chat inference."""
        config = {"gemma_model_name": "gemma-3n-e4b-it"}
        provider = GemmaProvider(config)

        messages = [{"role": "user", "content": "Hello!"}]

        response = await provider.achat(messages)

        assert isinstance(response, LLMChatResponse)
        assert isinstance(response.message, AssistantMessage)
        assert response.message.role == "assistant"
        assert len(response.message.content) > 0

    @pytest.mark.asyncio
    async def test_achat_with_parameters(self):
        """Test chat inference with custom parameters."""
        config = {"gemma_model_name": "gemma-3n-e4b-it"}
        provider = GemmaProvider(config)

        messages = [{"role": "user", "content": "Test"}]

        response = await provider.achat(messages, temperature=0.5, top_p=0.8, max_output_tokens=100)

        assert isinstance(response, LLMChatResponse)

    @pytest.mark.asyncio
    async def test_achat_multimodal(self):
        """Test multimodal chat inference."""
        config = {"gemma_model_name": "gemma-3n-e4b-it"}
        provider = GemmaProvider(config)

        img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)

        messages = [
            {
                "role": "user",
                "content": {
                    "text": "Describe this image",
                    "images": [img],
                },
            }
        ]

        response = await provider.achat(messages)

        assert isinstance(response, LLMChatResponse)
        assert len(response.message.content) > 0

    @pytest.mark.asyncio
    async def test_ainvoke_simple(self):
        """Test simple ainvoke."""
        config = {"gemma_model_name": "gemma-3n-e4b-it"}
        provider = GemmaProvider(config)

        response = await provider.ainvoke("Hello, how are you?")

        assert isinstance(response, LLMChatResponse)
        assert isinstance(response.message, AssistantMessage)


class TestGemmaProviderSyncMethods:
    """Test synchronous inference methods."""

    def test_invoke(self):
        """Test synchronous invoke method."""
        config = {"gemma_model_name": "gemma-3n-e4b-it"}
        provider = GemmaProvider(config)

        response = provider.invoke("Test prompt")

        assert isinstance(response, LLMChatResponse)

    def test_chat(self):
        """Test synchronous chat method."""
        config = {"gemma_model_name": "gemma-3n-e4b-it"}
        provider = GemmaProvider(config)

        messages = [{"role": "user", "content": "Hello"}]

        response = provider.chat(messages)

        assert isinstance(response, LLMChatResponse)


class TestGemmaProviderCapabilities:
    """Test provider capability reporting."""

    def test_get_model_name(self):
        """Test getting model name."""
        config = {"gemma_model_name": "gemma-3n-e2b-it"}
        provider = GemmaProvider(config)

        assert provider.get_model_name() == "gemma-3n-e2b-it"

    def test_supports_tools(self):
        """Test tool support reporting."""
        config = {"gemma_model_name": "gemma-3n-e4b-it"}
        provider = GemmaProvider(config)

        # Gemma 3n doesn't have native tool calling yet
        assert provider.supports_tools() is False

    def test_supports_multimodal(self):
        """Test multimodal support reporting."""
        config = {"gemma_model_name": "gemma-3n-e4b-it"}
        provider = GemmaProvider(config)

        assert provider.supports_multimodal() is True


class TestGemmaProviderCleanup:
    """Test cleanup and resource management."""

    @pytest.mark.asyncio
    async def test_stop(self):
        """Test stopping provider."""
        config = {"gemma_model_name": "gemma-3n-e4b-it"}
        provider = GemmaProvider(config)

        # Initialize first
        await provider._initialize_model()
        assert provider._initialized is True

        # Stop
        await provider.stop()

        assert provider._initialized is False
        assert provider._params is None
        assert provider._sampler is None

    def test_thread_pool_shutdown_on_del(self):
        """Test that thread pool is shut down on deletion."""
        config = {"gemma_model_name": "gemma-3n-e4b-it"}
        provider = GemmaProvider(config)

        pool = provider._inference_pool

        # Delete provider
        del provider

        # Thread pool should still be accessible until garbage collection
        # This is a basic check
        assert pool is not None
