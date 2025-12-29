"""
Visual Agent with Embedding Function Example

Demonstrates how to use a custom embedding function with VisualAgent
to pre-process visual data before passing it to the LLM.

This example shows:
1. How to define an async embedding function
2. How to integrate it with VisualAgent
3. Provider compatibility (most providers don't support pre-computed embeddings)
"""

import asyncio
import logging
from typing import Any, Dict, List

import numpy as np

from argentic.core.agent.visual_agent import VisualAgent
from argentic.core.drivers.webrtc_driver import WebRTCDriver
from argentic.core.llm.providers.base import LLMChatResponse, ModelProvider
from argentic.core.messager.messager import Messager
from argentic.core.protocol.message import AssistantMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Example 1: Simple embedding function
async def simple_embedding_function(frames: List[np.ndarray]) -> np.ndarray:
    """
    Simple embedding function that creates random embeddings.

    In a real implementation, this would:
    - Load a vision model (CLIP, ViT, etc.)
    - Process frames through the model
    - Return the visual embeddings

    Args:
        frames: List of numpy arrays (H, W, C)

    Returns:
        Embeddings array of shape (num_frames, embedding_dim)
    """
    logger.info(f"Encoding {len(frames)} frames to embeddings...")

    # Simulate encoding: create random embeddings (512-dim)
    embedding_dim = 512
    embeddings = np.random.randn(len(frames), embedding_dim).astype(np.float32)

    # Normalize (common in vision encoders)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    logger.info(f"Generated embeddings with shape: {embeddings.shape}")
    return embeddings


# Example 2: Embedding function with metadata
async def embedding_with_metadata(frames: List[np.ndarray]) -> Dict[str, Any]:
    """
    Embedding function that returns metadata along with embeddings.

    Returns:
        Dict with 'embeddings' key and additional metadata
    """
    import time

    start_time = time.time()

    # Generate embeddings
    embedding_dim = 768
    embeddings = np.random.randn(len(frames), embedding_dim).astype(np.float32)

    elapsed = time.time() - start_time

    # Return dict with metadata
    return {
        "embeddings": embeddings,
        "num_frames": len(frames),
        "embedding_dim": embedding_dim,
        "processing_time_ms": elapsed * 1000,
        "model_version": "example-v1.0",
    }


# Example 3: Dummy LLM Provider for testing
class DummyLLMProvider(ModelProvider):
    """
    Dummy LLM provider that demonstrates embedding handling.

    NOTE: Most real providers (Gemma, Gemini, vLLM OpenAI API) will raise
    NotImplementedError when they receive pre-computed embeddings.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        self.model_name = "dummy-model"

    async def achat(
        self, messages: List[Dict[str, Any]], system: str = None, **kwargs
    ) -> LLMChatResponse:
        """Process messages (demonstrating embedding detection)."""
        # Check if we received embeddings
        for msg in messages:
            if isinstance(msg, dict) and isinstance(msg.get("content"), dict):
                content = msg["content"]
                if "image_embeddings" in content:
                    # In reality, most providers would raise NotImplementedError here
                    embeddings = content["image_embeddings"]
                    logger.info(
                        f"Received image embeddings: shape={embeddings.shape if hasattr(embeddings, 'shape') else 'unknown'}"
                    )
                    return LLMChatResponse(
                        message=AssistantMessage(
                            content=f"Processed {len(embeddings)} embedding vectors. "
                            f"[This is a dummy response - real providers would process the embeddings]"
                        ),
                        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                        model=self.model_name,
                    )
                elif "images" in content:
                    images = content["images"]
                    logger.info(f"Received {len(images)} raw image frames")
                    return LLMChatResponse(
                        message=AssistantMessage(
                            content=f"Processed {len(images)} raw frames. "
                            f"[This is a dummy response - real providers would process the frames]"
                        ),
                        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                        model=self.model_name,
                    )

        # Text-only message
        return LLMChatResponse(
            message=AssistantMessage(content="Dummy text response"),
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            model=self.model_name,
        )

    async def ainvoke(self, prompt: str, system: str = None, **kwargs) -> LLMChatResponse:
        """Invoke with a simple prompt."""
        messages = [{"role": "user", "content": prompt}]
        return await self.achat(messages, system=system, **kwargs)

    def get_model_name(self) -> str:
        """Get model name."""
        return self.model_name

    def supports_multimodal(self) -> bool:
        """This provider supports multimodal input."""
        return True


async def example_with_embeddings():
    """Example: Visual Agent with embedding function."""
    logger.info("=== Example 1: Visual Agent with Embedding Function ===")

    # Setup messager
    messager = Messager(broker_address="localhost", port=1883)
    try:
        await messager.connect()
    except Exception as e:
        logger.warning(f"Could not connect to MQTT broker: {e}. Continuing anyway...")

    # Setup WebRTC driver
    driver = WebRTCDriver(
        video_buffer_size=20,
        frame_rate=10,
        resize_frames=(224, 224),
        enable_audio=False,
    )

    # Create dummy LLM provider
    llm = DummyLLMProvider()

    # Create visual agent WITH embedding function
    VisualAgent(
        llm=llm,
        messager=messager,
        webrtc_driver=driver,
        embedding_function=simple_embedding_function,  # Pass the embedding function
        enable_auto_processing=False,
        system_prompt="You are a visual AI assistant.",
    )

    logger.info("Visual Agent initialized with embedding function")

    # Simulate some frames (since we're not actually capturing video)
    logger.info("Simulating video frames...")
    dummy_frames = [
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(5)
    ]

    # Manually test the embedding function
    logger.info("Testing embedding function directly...")
    embeddings = await simple_embedding_function(dummy_frames)
    logger.info(f"Direct embedding test successful: {embeddings.shape}")

    logger.info("Example completed successfully!")


async def example_without_embeddings():
    """Example: Visual Agent without embedding function (default mode)."""
    logger.info("\n=== Example 2: Visual Agent WITHOUT Embedding Function ===")

    messager = Messager(broker_address="localhost", port=1883)
    try:
        await messager.connect()
    except Exception as e:
        logger.warning(f"Could not connect to MQTT broker: {e}. Continuing anyway...")

    driver = WebRTCDriver(
        video_buffer_size=20,
        frame_rate=10,
        resize_frames=(640, 480),
        enable_audio=False,
    )

    llm = DummyLLMProvider()

    # Create visual agent WITHOUT embedding function (uses raw frames)
    VisualAgent(
        llm=llm,
        messager=messager,
        webrtc_driver=driver,
        # No embedding_function - will use raw frames
        enable_auto_processing=False,
        system_prompt="You are a visual AI assistant.",
    )

    logger.info("Visual Agent initialized in raw frames mode (no embedding function)")
    logger.info("Example completed successfully!")


async def main():
    """Run all examples."""
    logger.info("Visual Agent Embedding Function Examples\n")

    try:
        # Example 1: With embeddings
        await example_with_embeddings()

        # Example 2: Without embeddings (default)
        await example_without_embeddings()

        logger.info("\n" + "=" * 60)
        logger.info("All examples completed successfully!")
        logger.info("=" * 60)
        logger.info("\nKey Takeaways:")
        logger.info("1. embedding_function is optional - agent defaults to raw frames")
        logger.info("2. Embedding function must be async and accept List[np.ndarray]")
        logger.info("3. Most providers (Gemma, Gemini, vLLM) don't support pre-computed embeddings")
        logger.info("4. Use embedding function for custom providers with native vLLM API")

    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
