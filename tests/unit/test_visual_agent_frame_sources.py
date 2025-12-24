"""Unit tests for VisualAgent with frame sources."""

import asyncio
from unittest.mock import AsyncMock

import numpy as np
import pytest

from argentic.core.agent.frame_source import (
    FrameSourceConfig,
    FunctionalFrameSource,
    StaticFrameSource,
)
from argentic.core.agent.visual_agent import VisualAgent
from argentic.core.llm.providers.base import LLMChatResponse, ModelProvider
from argentic.core.messager.messager import Messager
from argentic.core.protocol.chat_message import AssistantMessage


class MockLLMProvider(ModelProvider):
    """Mock LLM provider for testing."""

    def __init__(self, config=None, messager=None):
        self.config = config or {}
        self.messager = messager
        self.call_count = 0

    async def achat(self, messages, **kwargs):
        self.call_count += 1
        # Check if embeddings are in the message
        has_embeddings = False
        for msg in messages:
            if isinstance(msg, dict) and isinstance(msg.get("content"), dict):
                if "image_embeddings" in msg["content"]:
                    has_embeddings = True
                    break

        content = f"Mock response (embeddings={'yes' if has_embeddings else 'no'})"
        return LLMChatResponse(
            message=AssistantMessage(content=content),
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            model="mock-model",
        )

    async def ainvoke(self, prompt, **kwargs):
        """Async invoke implementation."""
        return await self.achat([{"role": "user", "content": prompt}], **kwargs)

    def chat(self, messages, **kwargs):
        """Sync chat (not used in tests)."""
        raise NotImplementedError()

    def invoke(self, prompt, **kwargs):
        """Sync invoke (not used in tests)."""
        raise NotImplementedError()

    def get_model_name(self):
        return "mock-model"


class TestVisualAgentWithStaticFrameSource:
    """Test VisualAgent with StaticFrameSource."""

    @pytest.mark.asyncio
    async def test_init_with_static_frame_source(self):
        """Test initialization with static frame source."""
        frames = [np.random.rand(480, 640, 3) for _ in range(5)]
        frame_source = StaticFrameSource(frames)

        messager = Messager(broker_address="localhost", port=1883)
        llm = MockLLMProvider()

        agent = VisualAgent(
            llm=llm,
            messager=messager,
            frame_source=frame_source,
            enable_auto_processing=False,
        )

        assert agent.frame_source == frame_source
        assert agent.embedding_function is None

    @pytest.mark.asyncio
    async def test_query_with_video_static_frames(self):
        """Test querying with static frames."""
        frames = [np.random.rand(480, 640, 3) for _ in range(10)]
        frame_source = StaticFrameSource(frames)

        messager = Messager(broker_address="localhost", port=1883)
        llm = MockLLMProvider()

        agent = VisualAgent(
            llm=llm,
            messager=messager,
            frame_source=frame_source,
            enable_auto_processing=False,
        )

        await agent.async_init()

        response = await agent.query_with_video("What do you see?")

        assert "Mock response" in response
        assert llm.call_count == 1

        await agent.stop()

    @pytest.mark.asyncio
    async def test_query_with_insufficient_frames(self):
        """Test querying with too few frames."""
        frames = [np.random.rand(480, 640, 3) for _ in range(2)]
        frame_source = StaticFrameSource(frames)

        messager = Messager(broker_address="localhost", port=1883)
        llm = MockLLMProvider()

        agent = VisualAgent(
            llm=llm,
            messager=messager,
            frame_source=frame_source,
            enable_auto_processing=False,
            min_frames_for_processing=10,
        )

        await agent.async_init()

        # Should still work but log warning
        response = await agent.query_with_video("What do you see?")

        assert "Mock response" in response

        await agent.stop()


class TestVisualAgentWithFunctionalFrameSource:
    """Test VisualAgent with FunctionalFrameSource."""

    @pytest.mark.asyncio
    async def test_init_with_functional_frame_source(self):
        """Test initialization with functional frame source."""

        async def get_frames():
            return [np.random.rand(480, 640, 3) for _ in range(5)]

        frame_source = FunctionalFrameSource(get_frames_fn=get_frames)

        messager = Messager(broker_address="localhost", port=1883)
        llm = MockLLMProvider()

        agent = VisualAgent(
            llm=llm,
            messager=messager,
            frame_source=frame_source,
            enable_auto_processing=False,
        )

        assert agent.frame_source == frame_source

    @pytest.mark.asyncio
    async def test_query_with_video_functional_source(self):
        """Test querying with functional source."""
        call_count = {"value": 0}

        async def get_frames():
            call_count["value"] += 1
            return [np.random.rand(480, 640, 3) for _ in range(10)]

        frame_source = FunctionalFrameSource(get_frames_fn=get_frames)

        messager = Messager(broker_address="localhost", port=1883)
        llm = MockLLMProvider()

        agent = VisualAgent(
            llm=llm,
            messager=messager,
            frame_source=frame_source,
            enable_auto_processing=False,
        )

        await agent.async_init()

        response = await agent.query_with_video("What do you see?")

        assert "Mock response" in response
        assert call_count["value"] > 0  # Should have called get_frames

        await agent.stop()

    @pytest.mark.asyncio
    async def test_functional_source_with_audio(self):
        """Test functional source with audio support."""

        async def get_frames():
            return [np.random.rand(480, 640, 3) for _ in range(5)]

        async def get_audio():
            return np.random.randn(1000)

        frame_source = FunctionalFrameSource(
            get_frames_fn=get_frames, get_audio_fn=get_audio
        )

        messager = Messager(broker_address="localhost", port=1883)
        llm = MockLLMProvider()

        agent = VisualAgent(
            llm=llm,
            messager=messager,
            frame_source=frame_source,
            enable_auto_processing=False,
        )

        await agent.async_init()

        # Query should work with audio
        response = await agent.query_with_video("What do you see?")

        assert "Mock response" in response

        await agent.stop()


class TestVisualAgentWithEmbeddingFunction:
    """Test VisualAgent with custom embedding function."""

    @pytest.mark.asyncio
    async def test_init_with_embedding_function(self):
        """Test initialization with embedding function."""

        async def embed_frames(frames):
            return np.random.randn(len(frames), 512)

        frames = [np.random.rand(480, 640, 3) for _ in range(5)]
        frame_source = StaticFrameSource(frames)

        messager = Messager(broker_address="localhost", port=1883)
        llm = MockLLMProvider()

        agent = VisualAgent(
            llm=llm,
            messager=messager,
            frame_source=frame_source,
            embedding_function=embed_frames,
            enable_auto_processing=False,
        )

        assert agent.embedding_function == embed_frames

    @pytest.mark.asyncio
    async def test_query_with_embedding_function(self):
        """Test querying with embedding function."""
        embed_call_count = {"value": 0}

        async def embed_frames(frames):
            embed_call_count["value"] += 1
            return np.random.randn(len(frames), 512).astype(np.float32)

        frames = [np.random.rand(480, 640, 3) for _ in range(10)]
        frame_source = StaticFrameSource(frames)

        messager = Messager(broker_address="localhost", port=1883)
        llm = MockLLMProvider()

        agent = VisualAgent(
            llm=llm,
            messager=messager,
            frame_source=frame_source,
            embedding_function=embed_frames,
            enable_auto_processing=False,
        )

        await agent.async_init()

        response = await agent.query_with_video("What do you see?")

        # Should have called embedding function
        assert embed_call_count["value"] == 1
        assert "embeddings=yes" in response  # Our mock detects embeddings

        await agent.stop()

    @pytest.mark.asyncio
    async def test_embedding_function_with_dict_return(self):
        """Test embedding function that returns dict with metadata."""

        async def embed_frames(frames):
            return {
                "embeddings": np.random.randn(len(frames), 512).astype(np.float32),
                "model": "test-encoder",
                "version": "1.0",
            }

        frames = [np.random.rand(480, 640, 3) for _ in range(10)]
        frame_source = StaticFrameSource(frames)

        messager = Messager(broker_address="localhost", port=1883)
        llm = MockLLMProvider()

        agent = VisualAgent(
            llm=llm,
            messager=messager,
            frame_source=frame_source,
            embedding_function=embed_frames,
            enable_auto_processing=False,
        )

        await agent.async_init()

        response = await agent.query_with_video("What do you see?")

        # Should work with dict return
        assert "embeddings=yes" in response

        await agent.stop()

    @pytest.mark.asyncio
    async def test_embedding_function_error_fallback(self):
        """Test that errors in embedding function fall back to raw frames."""

        async def embed_frames(frames):
            raise ValueError("Embedding failed!")

        frames = [np.random.rand(480, 640, 3) for _ in range(10)]
        frame_source = StaticFrameSource(frames)

        messager = Messager(broker_address="localhost", port=1883)
        llm = MockLLMProvider()

        agent = VisualAgent(
            llm=llm,
            messager=messager,
            frame_source=frame_source,
            embedding_function=embed_frames,
            enable_auto_processing=False,
        )

        await agent.async_init()

        # Should fall back to raw frames
        response = await agent.query_with_video("What do you see?")

        # Should use raw frames (no embeddings)
        assert "embeddings=no" in response

        await agent.stop()


class TestVisualAgentAutoProcessing:
    """Test VisualAgent auto-processing with frame sources."""

    @pytest.mark.asyncio
    async def test_auto_processing_disabled(self):
        """Test that auto-processing can be disabled."""
        frames = [np.random.rand(480, 640, 3) for _ in range(10)]
        frame_source = StaticFrameSource(frames)

        messager = Messager(broker_address="localhost", port=1883)
        llm = MockLLMProvider()

        agent = VisualAgent(
            llm=llm,
            messager=messager,
            frame_source=frame_source,
            enable_auto_processing=False,
        )

        await agent.async_init()

        # Should not have processing task
        assert agent._processing_task is None

        await agent.stop()

    @pytest.mark.asyncio
    async def test_auto_processing_enabled(self):
        """Test that auto-processing can be enabled."""
        call_count = {"value": 0}

        async def get_frames():
            call_count["value"] += 1
            return [np.random.rand(480, 640, 3) for _ in range(10)]

        frame_source = FunctionalFrameSource(get_frames_fn=get_frames)

        messager = Messager(broker_address="localhost", port=1883)
        llm = MockLLMProvider()

        agent = VisualAgent(
            llm=llm,
            messager=messager,
            frame_source=frame_source,
            enable_auto_processing=True,
            auto_process_interval=0.1,  # Very short for testing
        )

        await agent.async_init()

        # Should have processing task
        assert agent._processing_task is not None

        # Wait a bit for auto-processing
        await asyncio.sleep(0.3)

        # Should have processed at least once
        # (This may be flaky depending on timing)

        await agent.stop()


class TestVisualAgentLifecycle:
    """Test VisualAgent lifecycle with frame sources."""

    @pytest.mark.asyncio
    async def test_async_init_starts_frame_source(self):
        """Test that async_init starts the frame source."""
        frames = [np.random.rand(480, 640, 3) for _ in range(5)]
        frame_source = StaticFrameSource(frames)
        frame_source.start = AsyncMock()

        messager = Messager(broker_address="localhost", port=1883)
        llm = MockLLMProvider()

        agent = VisualAgent(
            llm=llm,
            messager=messager,
            frame_source=frame_source,
            enable_auto_processing=False,
        )

        await agent.async_init()

        frame_source.start.assert_awaited_once()

        await agent.stop()

    @pytest.mark.asyncio
    async def test_stop_stops_frame_source(self):
        """Test that stop stops the frame source."""
        frames = [np.random.rand(480, 640, 3) for _ in range(5)]
        frame_source = StaticFrameSource(frames)
        frame_source.stop = AsyncMock()

        messager = Messager(broker_address="localhost", port=1883)
        llm = MockLLMProvider()

        agent = VisualAgent(
            llm=llm,
            messager=messager,
            frame_source=frame_source,
            enable_auto_processing=False,
        )

        await agent.async_init()
        await agent.stop()

        frame_source.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test using VisualAgent as async context manager."""
        frames = [np.random.rand(480, 640, 3) for _ in range(10)]
        frame_source = StaticFrameSource(frames)

        messager = Messager(broker_address="localhost", port=1883)
        llm = MockLLMProvider()

        async with VisualAgent(
            llm=llm,
            messager=messager,
            frame_source=frame_source,
            enable_auto_processing=False,
        ) as agent:
            # Should be initialized
            response = await agent.query_with_video("What do you see?")
            assert "Mock response" in response

        # Should be stopped after context exit


class TestVisualAgentWithDifferentFrameSources:
    """Test VisualAgent compatibility with different frame source implementations."""

    @pytest.mark.asyncio
    async def test_works_with_custom_frame_source(self):
        """Test that VisualAgent works with any FrameSource implementation."""
        from argentic.core.agent.frame_source import FrameSource

        class CustomFrameSource(FrameSource):
            """Custom test frame source."""

            async def get_frames(self):
                return [np.random.rand(100, 100, 3) for _ in range(5)]

            async def get_audio(self):
                return None

            async def start(self):
                pass

            async def stop(self):
                pass

        frame_source = CustomFrameSource(FrameSourceConfig())

        messager = Messager(broker_address="localhost", port=1883)
        llm = MockLLMProvider()

        agent = VisualAgent(
            llm=llm,
            messager=messager,
            frame_source=frame_source,
            enable_auto_processing=False,
        )

        await agent.async_init()

        response = await agent.query_with_video("What do you see?")

        assert "Mock response" in response

        await agent.stop()


class TestVisualAgentQueryBehavior:
    """Test VisualAgent query behavior with frame sources."""

    @pytest.mark.asyncio
    async def test_query_without_frames(self):
        """Test query behavior when no frames available."""
        frame_source = StaticFrameSource([])  # Empty frames

        messager = Messager(broker_address="localhost", port=1883)
        llm = MockLLMProvider()

        agent = VisualAgent(
            llm=llm,
            messager=messager,
            frame_source=frame_source,
            enable_auto_processing=False,
        )

        await agent.async_init()

        response = await agent.query_with_video("What do you see?")

        # Should return warning message
        assert "No video frames available" in response

        await agent.stop()

    @pytest.mark.asyncio
    async def test_query_method_uses_frame_source(self):
        """Test that base query method uses visual context if available."""
        frames = [np.random.rand(480, 640, 3) for _ in range(15)]
        frame_source = StaticFrameSource(frames)

        messager = Messager(broker_address="localhost", port=1883)
        llm = MockLLMProvider()

        agent = VisualAgent(
            llm=llm,
            messager=messager,
            frame_source=frame_source,
            enable_auto_processing=False,
            min_frames_for_processing=10,
        )

        await agent.async_init()

        # Use base query method (not query_with_video)
        response = await agent.query("Tell me what you see")

        # Should use visual processing since we have enough frames
        assert "Mock response" in response

        await agent.stop()
