"""
Visual Agent with Different Frame Sources

Demonstrates how to use VisualAgent with various frame sources:
- WebRTC (real-time streaming)
- Video files (pre-recorded)
- Static images (for testing)
- Custom functional source

This showcases the flexibility of the frame source abstraction.
"""

import asyncio
import logging
from pathlib import Path
from typing import List

import numpy as np

from argentic.core.agent.frame_source import (
    FunctionalFrameSource,
    StaticFrameSource,
)
from argentic.core.agent.visual_agent import VisualAgent
from argentic.core.llm.providers.base import LLMChatResponse, ModelProvider
from argentic.core.messager.messager import Messager
from argentic.core.protocol.message import AssistantMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Dummy LLM for testing
class DummyLLM(ModelProvider):
    """Dummy LLM that just echoes frame count."""

    async def achat(self, messages, **kwargs) -> LLMChatResponse:
        # Extract frame count from messages
        for msg in messages:
            if isinstance(msg, dict) and isinstance(msg.get("content"), dict):
                images = msg["content"].get("images", [])
                if images:
                    return LLMChatResponse(
                        message=AssistantMessage(
                            content=f"Processed {len(images)} frames"
                        ),
                        usage={
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                        },
                        model="dummy",
                    )
        return LLMChatResponse(
            message=AssistantMessage(content="No frames received"),
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            model="dummy",
        )

    def get_model_name(self) -> str:
        return "dummy"


async def example_static_frames():
    """Example 1: Using static frames (useful for testing)."""
    logger.info("=== Example 1: Static Frame Source ===")

    # Create some dummy frames
    frames = [
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(5)
    ]

    # Create static frame source
    source = StaticFrameSource(frames)

    # Create messager (won't actually connect in this example)
    messager = Messager(broker_address="localhost", port=1883)

    # Create agent
    agent = VisualAgent(
        llm=DummyLLM(),
        messager=messager,
        frame_source=source,
        enable_auto_processing=False,  # Manual control
    )

    # Query
    await agent.async_init()
    response = await agent.query_with_video("What do you see?")
    logger.info(f"Response: {response}")

    await agent.stop()
    logger.info("Static frame example complete\n")


async def example_functional_source():
    """Example 2: Using functional frame source with custom logic."""
    logger.info("=== Example 2: Functional Frame Source ===")

    # Frame counter for dynamic generation
    frame_count = {"value": 0}

    async def get_dynamic_frames() -> List[np.ndarray]:
        """Generate frames dynamically."""
        # Simulate generating new frames each time
        frames = []
        for i in range(3):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            # Add frame number as visual indicator (top-left corner)
            frame[0:50, 0:100] = frame_count["value"] % 255
            frames.append(frame)
            frame_count["value"] += 1
        logger.info(
            f"Generated {len(frames)} frames (total count: {frame_count['value']})"
        )
        return frames

    # Create functional frame source
    source = FunctionalFrameSource(get_frames_fn=get_dynamic_frames)

    messager = Messager(broker_address="localhost", port=1883)

    agent = VisualAgent(
        llm=DummyLLM(),
        messager=messager,
        frame_source=source,
        enable_auto_processing=False,
    )

    await agent.async_init()

    # Query multiple times - should get different frames each time
    for i in range(3):
        response = await agent.query_with_video(f"Query {i + 1}: What do you see?")
        logger.info(f"Query {i + 1} Response: {response}")

    await agent.stop()
    logger.info("Functional source example complete\n")


async def example_webrtc_source():
    """Example 3: Using WebRTC frame source (requires running WebRTC server)."""
    logger.info("=== Example 3: WebRTC Frame Source ===")
    logger.info("Note: This example requires a WebRTC server to be running")

    try:
        from argentic.core.agent.frame_sources.webrtc_source import WebRTCFrameSource
        from argentic.core.drivers.webrtc_driver import WebRTCDriver

        # Create WebRTC driver
        driver = WebRTCDriver(
            video_buffer_size=30,
            frame_rate=10,
            resize_frames=(640, 480),
            enable_audio=False,
        )

        # Wrap in frame source
        source = WebRTCFrameSource(driver)

        messager = Messager(broker_address="localhost", port=1883)

        VisualAgent(
            llm=DummyLLM(),
            messager=messager,
            frame_source=source,
            enable_auto_processing=False,
        )

        logger.info("WebRTC example configured (would need actual WebRTC connection)")
        logger.info(f"Frame source info: {source.get_info()}")

    except Exception as e:
        logger.warning(f"WebRTC example setup failed (expected): {e}")

    logger.info("WebRTC source example complete\n")


async def example_video_file_source():
    """Example 4: Using video file source (requires video file)."""
    logger.info("=== Example 4: Video File Frame Source ===")
    logger.info("Note: This example requires a video file to exist")

    try:
        from argentic.core.agent.frame_sources.video_file_source import (
            VideoFileFrameSource,
        )

        # Example video path (would need to exist)
        video_path = "/path/to/your/video.mp4"

        if Path(video_path).exists():
            source = VideoFileFrameSource(
                video_path=video_path,
                buffer_size=30,
                fps=10,  # Subsample to 10 FPS
                loop=False,
            )

            messager = Messager(broker_address="localhost", port=1883)

            agent = VisualAgent(
                llm=DummyLLM(),
                messager=messager,
                frame_source=source,
                enable_auto_processing=False,
            )

            await agent.async_init()
            logger.info(f"Video source info: {source.get_info()}")

            response = await agent.query_with_video("What's in this video?")
            logger.info(f"Response: {response}")

            await agent.stop()
        else:
            logger.info(f"Video file not found: {video_path} (skipping)")

    except ImportError as e:
        logger.warning(f"Video file source not available: {e}")

    logger.info("Video file source example complete\n")


async def main():
    """Run all examples."""
    logger.info("Visual Agent Frame Sources Examples\n")

    # Example 1: Static frames (always works)
    await example_static_frames()

    # Example 2: Functional source (always works)
    await example_functional_source()

    # Example 3: WebRTC (requires setup)
    await example_webrtc_source()

    # Example 4: Video file (requires file)
    await example_video_file_source()

    logger.info("=" * 60)
    logger.info("All examples complete!")
    logger.info("=" * 60)
    logger.info("\nKey Takeaways:")
    logger.info("1. VisualAgent works with ANY FrameSource implementation")
    logger.info("2. StaticFrameSource: Easy testing with fixed images")
    logger.info("3. FunctionalFrameSource: Custom logic via callbacks")
    logger.info("4. WebRTCFrameSource: Real-time streaming")
    logger.info("5. VideoFileFrameSource: Pre-recorded videos")
    logger.info("6. Easy to create custom sources for cameras, screens, etc.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
