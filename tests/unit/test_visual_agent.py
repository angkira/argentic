"""Unit tests for Visual Agent."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from argentic.core.agent.visual_agent import VisualAgent
from argentic.core.drivers.webrtc_driver import WebRTCDriver
from argentic.core.llm.providers.gemma import GemmaProvider
from argentic.core.messager.messager import Messager
from argentic.core.protocol.chat_message import LLMChatResponse, AssistantMessage


@pytest.fixture
async def mock_messager():
    """Create a mock messager."""
    messager = AsyncMock(spec=Messager)
    messager.connect = AsyncMock()
    messager.disconnect = AsyncMock()
    messager.publish = AsyncMock()
    messager.subscribe = AsyncMock()
    return messager


@pytest.fixture
async def mock_driver():
    """Create a mock WebRTC driver."""
    driver = AsyncMock(spec=WebRTCDriver)
    driver.connect = AsyncMock()
    driver.disconnect = AsyncMock()
    driver.start_capture = AsyncMock()
    driver.stop_capture = AsyncMock()
    driver.get_frame_buffer = AsyncMock(return_value=[])
    driver.get_audio_buffer = AsyncMock(return_value=None)
    driver.clear_buffers = MagicMock()
    return driver


@pytest.fixture
def mock_llm():
    """Create a mock LLM provider."""
    llm = MagicMock(spec=GemmaProvider)
    return llm


class TestVisualAgentInitialization:
    """Test VisualAgent initialization."""

    def test_init_default_params(self, mock_llm, mock_messager, mock_driver):
        """Test agent initialization with default parameters."""
        agent = VisualAgent(
            llm=mock_llm,
            messager=mock_messager,
            webrtc_driver=mock_driver,
        )

        assert agent.driver is mock_driver
        assert agent.auto_process_interval == 5.0
        assert agent.enable_auto_processing is True
        assert agent.min_frames_for_processing == 10
        assert agent._processing_active is False

    def test_init_custom_params(self, mock_llm, mock_messager, mock_driver):
        """Test agent initialization with custom parameters."""
        agent = VisualAgent(
            llm=mock_llm,
            messager=mock_messager,
            webrtc_driver=mock_driver,
            auto_process_interval=10.0,
            enable_auto_processing=False,
            min_frames_for_processing=20,
            visual_response_topic="custom/topic",
        )

        assert agent.auto_process_interval == 10.0
        assert agent.enable_auto_processing is False
        assert agent.min_frames_for_processing == 20
        assert agent.visual_response_topic == "custom/topic"


class TestVisualAgentAsyncInit:
    """Test async initialization."""

    @pytest.mark.asyncio
    async def test_async_init_connects_driver(self, mock_llm, mock_messager, mock_driver):
        """Test that async_init connects the driver."""
        agent = VisualAgent(
            llm=mock_llm,
            messager=mock_messager,
            webrtc_driver=mock_driver,
            enable_auto_processing=False,  # Disable to avoid task creation
        )

        with patch.object(agent, "_tool_manager", AsyncMock()):
            await agent.async_init()

        mock_driver.connect.assert_called_once()
        mock_driver.start_capture.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_init_starts_auto_processing(self, mock_llm, mock_messager, mock_driver):
        """Test that auto-processing is started when enabled."""
        agent = VisualAgent(
            llm=mock_llm,
            messager=mock_messager,
            webrtc_driver=mock_driver,
            enable_auto_processing=True,
        )

        with patch.object(agent, "_tool_manager", AsyncMock()):
            await agent.async_init()

        assert agent._processing_task is not None
        assert not agent._processing_task.done()

        # Cleanup
        await agent.stop()

    @pytest.mark.asyncio
    async def test_async_init_no_auto_processing(self, mock_llm, mock_messager, mock_driver):
        """Test initialization without auto-processing."""
        agent = VisualAgent(
            llm=mock_llm,
            messager=mock_messager,
            webrtc_driver=mock_driver,
            enable_auto_processing=False,
        )

        with patch.object(agent, "_tool_manager", AsyncMock()):
            await agent.async_init()

        assert agent._processing_task is None


class TestVisualAgentVideoProcessing:
    """Test video processing functionality."""

    @pytest.mark.asyncio
    async def test_process_visual_input_text_only(self, mock_llm, mock_messager, mock_driver):
        """Test processing visual input with text only."""
        agent = VisualAgent(
            llm=mock_llm,
            messager=mock_messager,
            webrtc_driver=mock_driver,
        )

        # Mock LLM response
        mock_response = LLMChatResponse(
            message=AssistantMessage(role="assistant", content="I see a test image")
        )

        with patch.object(agent, "_call_llm", new=AsyncMock(return_value=mock_response)):
            frames = [np.random.rand(480, 640, 3) for _ in range(5)]

            result = await agent._process_visual_input(frames, None, "What do you see?")

        assert result == "I see a test image"

    @pytest.mark.asyncio
    async def test_process_visual_input_with_audio(self, mock_llm, mock_messager, mock_driver):
        """Test processing visual input with audio."""
        agent = VisualAgent(
            llm=mock_llm,
            messager=mock_messager,
            webrtc_driver=mock_driver,
        )

        mock_response = LLMChatResponse(
            message=AssistantMessage(role="assistant", content="I see and hear something")
        )

        with patch.object(agent, "_call_llm", new=AsyncMock(return_value=mock_response)):
            frames = [np.random.rand(480, 640, 3) for _ in range(5)]
            audio = np.random.randn(1000).astype(np.float32)

            result = await agent._process_visual_input(frames, audio, "What do you see and hear?")

        assert "see and hear" in result


class TestVisualAgentQueryWithVideo:
    """Test query_with_video functionality."""

    @pytest.mark.asyncio
    async def test_query_with_video_no_frames(self, mock_llm, mock_messager, mock_driver):
        """Test query when no frames available."""
        agent = VisualAgent(
            llm=mock_llm,
            messager=mock_messager,
            webrtc_driver=mock_driver,
        )

        # Mock empty buffer
        mock_driver.get_frame_buffer = AsyncMock(return_value=[])

        result = await agent.query_with_video("What do you see?")

        assert "No video frames available" in result

    @pytest.mark.asyncio
    async def test_query_with_video_success(self, mock_llm, mock_messager, mock_driver):
        """Test successful query with video."""
        agent = VisualAgent(
            llm=mock_llm,
            messager=mock_messager,
            webrtc_driver=mock_driver,
        )

        # Mock frames
        frames = [np.random.rand(480, 640, 3) for _ in range(15)]
        mock_driver.get_frame_buffer = AsyncMock(return_value=frames)

        mock_response = LLMChatResponse(
            message=AssistantMessage(role="assistant", content="I see a video")
        )

        with patch.object(agent, "_call_llm", new=AsyncMock(return_value=mock_response)):
            result = await agent.query_with_video("Describe the video")

        assert result == "I see a video"
        mock_messager.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_with_video_few_frames_warning(self, mock_llm, mock_messager, mock_driver):
        """Test query with fewer frames than minimum."""
        agent = VisualAgent(
            llm=mock_llm,
            messager=mock_messager,
            webrtc_driver=mock_driver,
            min_frames_for_processing=10,
        )

        # Only 5 frames
        frames = [np.random.rand(480, 640, 3) for _ in range(5)]
        mock_driver.get_frame_buffer = AsyncMock(return_value=frames)

        mock_response = LLMChatResponse(
            message=AssistantMessage(role="assistant", content="Limited frames")
        )

        with patch.object(agent, "_call_llm", new=AsyncMock(return_value=mock_response)):
            result = await agent.query_with_video("What's there?")

        # Should still process, just with a warning
        assert result == "Limited frames"


class TestVisualAgentQuery:
    """Test query method (override from base Agent)."""

    @pytest.mark.asyncio
    async def test_query_with_visual_context(self, mock_llm, mock_messager, mock_driver):
        """Test query uses visual context when available."""
        agent = VisualAgent(
            llm=mock_llm,
            messager=mock_messager,
            webrtc_driver=mock_driver,
            min_frames_for_processing=5,
        )

        # Mock frames available
        frames = [np.random.rand(480, 640, 3) for _ in range(10)]
        mock_driver.get_frame_buffer = AsyncMock(return_value=frames)

        mock_response = LLMChatResponse(
            message=AssistantMessage(role="assistant", content="Visual context used")
        )

        with patch.object(agent, "_call_llm", new=AsyncMock(return_value=mock_response)):
            result = await agent.query("What's happening?")

        assert result == "Visual context used"

    @pytest.mark.asyncio
    async def test_query_fallback_to_text_only(self, mock_llm, mock_messager, mock_driver):
        """Test query falls back to text-only when no frames."""
        agent = VisualAgent(
            llm=mock_llm,
            messager=mock_messager,
            webrtc_driver=mock_driver,
        )

        # No frames available
        mock_driver.get_frame_buffer = AsyncMock(return_value=[])

        # Mock base agent query
        with patch(
            "argentic.core.agent.agent.Agent.query",
            new=AsyncMock(return_value="Text-only response"),
        ):
            result = await agent.query("Hello")

        assert result == "Text-only response"


class TestVisualAgentAutoProcessing:
    """Test auto-processing loop functionality."""

    @pytest.mark.asyncio
    async def test_auto_process_loop_runs(self, mock_llm, mock_messager, mock_driver):
        """Test that auto-process loop runs."""
        agent = VisualAgent(
            llm=mock_llm,
            messager=mock_messager,
            webrtc_driver=mock_driver,
            auto_process_interval=0.1,  # Short interval for testing
            enable_auto_processing=True,
            min_frames_for_processing=2,
        )

        # Mock frames
        frames = [np.random.rand(480, 640, 3) for _ in range(5)]
        mock_driver.get_frame_buffer = AsyncMock(return_value=frames)

        mock_response = LLMChatResponse(
            message=AssistantMessage(role="assistant", content="Auto-processed")
        )

        with patch.object(agent, "_tool_manager", AsyncMock()):
            with patch.object(agent, "_call_llm", new=AsyncMock(return_value=mock_response)):
                await agent.async_init()

                # Wait for a couple of processing cycles
                await asyncio.sleep(0.3)

        # Cleanup
        await agent.stop()

        # Should have published at least once
        assert mock_messager.publish.call_count >= 1

    @pytest.mark.asyncio
    async def test_auto_process_skips_insufficient_frames(
        self, mock_llm, mock_messager, mock_driver
    ):
        """Test that auto-process skips when insufficient frames."""
        agent = VisualAgent(
            llm=mock_llm,
            messager=mock_messager,
            webrtc_driver=mock_driver,
            auto_process_interval=0.1,
            min_frames_for_processing=10,
        )

        # Only 3 frames (less than minimum)
        frames = [np.random.rand(480, 640, 3) for _ in range(3)]
        mock_driver.get_frame_buffer = AsyncMock(return_value=frames)

        with patch.object(agent, "_tool_manager", AsyncMock()):
            await agent.async_init()

            # Wait for processing cycles
            await asyncio.sleep(0.3)

        # Cleanup
        await agent.stop()

        # Should not have published
        mock_messager.publish.assert_not_called()


class TestVisualAgentPublishing:
    """Test response publishing functionality."""

    @pytest.mark.asyncio
    async def test_publish_visual_response(self, mock_llm, mock_messager, mock_driver):
        """Test publishing visual response."""
        agent = VisualAgent(
            llm=mock_llm,
            messager=mock_messager,
            webrtc_driver=mock_driver,
            visual_response_topic="test/visual/response",
        )

        await agent._publish_visual_response("Test response", "Test question")

        mock_messager.publish.assert_called_once()
        call_args = mock_messager.publish.call_args

        assert call_args[0][0] == "test/visual/response"
        message = call_args[0][1]
        assert message.answer == "Test response"
        assert message.question == "Test question"


class TestVisualAgentControls:
    """Test pause/resume controls."""

    @pytest.mark.asyncio
    async def test_pause_auto_processing(self, mock_llm, mock_messager, mock_driver):
        """Test pausing auto-processing."""
        agent = VisualAgent(
            llm=mock_llm,
            messager=mock_messager,
            webrtc_driver=mock_driver,
            enable_auto_processing=True,
        )

        with patch.object(agent, "_tool_manager", AsyncMock()):
            await agent.async_init()

        agent.pause_auto_processing()

        assert agent._processing_active is False

        # Cleanup
        await agent.stop()

    @pytest.mark.asyncio
    async def test_resume_auto_processing(self, mock_llm, mock_messager, mock_driver):
        """Test resuming auto-processing."""
        agent = VisualAgent(
            llm=mock_llm,
            messager=mock_messager,
            webrtc_driver=mock_driver,
            enable_auto_processing=True,
        )

        with patch.object(agent, "_tool_manager", AsyncMock()):
            await agent.async_init()

        agent.pause_auto_processing()
        agent.resume_auto_processing()

        assert agent._processing_active is True

        # Cleanup
        await agent.stop()


class TestVisualAgentStop:
    """Test agent stopping and cleanup."""

    @pytest.mark.asyncio
    async def test_stop_cleans_up_resources(self, mock_llm, mock_messager, mock_driver):
        """Test that stop properly cleans up resources."""
        agent = VisualAgent(
            llm=mock_llm,
            messager=mock_messager,
            webrtc_driver=mock_driver,
            enable_auto_processing=True,
        )

        with patch.object(agent, "_tool_manager", AsyncMock()):
            await agent.async_init()

        await agent.stop()

        assert agent._processing_active is False
        assert agent._processing_task is None
        mock_driver.stop_capture.assert_called_once()
        mock_driver.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_cancels_processing_task(self, mock_llm, mock_messager, mock_driver):
        """Test that stop cancels the processing task."""
        agent = VisualAgent(
            llm=mock_llm,
            messager=mock_messager,
            webrtc_driver=mock_driver,
            enable_auto_processing=True,
        )

        with patch.object(agent, "_tool_manager", AsyncMock()):
            await agent.async_init()

        task = agent._processing_task
        assert task is not None
        assert not task.done()

        await agent.stop()

        assert task.cancelled() or task.done()


class TestVisualAgentContextManager:
    """Test async context manager functionality."""

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_llm, mock_messager, mock_driver):
        """Test using VisualAgent as async context manager."""
        agent = VisualAgent(
            llm=mock_llm,
            messager=mock_messager,
            webrtc_driver=mock_driver,
            enable_auto_processing=False,
        )

        with patch.object(agent, "_tool_manager", AsyncMock()):
            async with agent:
                # Should be initialized
                pass

        # After exit, should be stopped
        mock_driver.disconnect.assert_called()
