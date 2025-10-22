"""Integration tests for Visual Agent components."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from argentic.core.agent.visual_agent import VisualAgent
from argentic.core.drivers.webrtc_driver import WebRTCDriver
from argentic.core.llm.providers.gemma import GemmaProvider
from argentic.core.messager.messager import Messager
from argentic.core.protocol.chat_message import AssistantMessage, LLMChatResponse


@pytest.fixture
async def integration_messager():
    """Create a mock messager for integration tests."""
    messager = AsyncMock(spec=Messager)
    messager.connect = AsyncMock()
    messager.disconnect = AsyncMock()
    messager.publish = AsyncMock()
    messager.subscribe = AsyncMock()
    messager.is_connected = MagicMock(return_value=True)
    return messager


class TestVisualAgentDriverIntegration:
    """Test integration between VisualAgent and WebRTC Driver."""

    @pytest.mark.asyncio
    async def test_agent_initializes_driver(self, integration_messager):
        """Test that agent properly initializes the driver."""
        driver = WebRTCDriver()

        config = {
            "gemma_model_name": "gemma-3n-e4b-it",
            "gemma_checkpoint_path": "test_checkpoint",
        }
        llm = GemmaProvider(config)

        agent = VisualAgent(
            llm=llm,
            messager=integration_messager,
            webrtc_driver=driver,
            enable_auto_processing=False,
        )

        with patch.object(agent, "_tool_manager", AsyncMock()):
            await agent.async_init()

        # Driver should be connected and capturing
        assert driver._connected is True

        await agent.stop()

        # Driver should be disconnected
        assert driver._connected is False

    @pytest.mark.asyncio
    async def test_agent_reads_driver_buffers(self, integration_messager):
        """Test that agent can read frames from driver buffers."""
        driver = WebRTCDriver(video_buffer_size=5)

        # Manually add frames to driver buffer
        test_frames = [np.random.rand(480, 640, 3) for _ in range(5)]
        driver._frame_buffer.extend(test_frames)

        config = {"gemma_model_name": "gemma-3n-e4b-it"}
        llm = GemmaProvider(config)

        agent = VisualAgent(
            llm=llm,
            messager=integration_messager,
            webrtc_driver=driver,
            enable_auto_processing=False,
        )

        # Get frames through agent
        frames = await driver.get_frame_buffer()

        assert len(frames) == 5
        assert all(isinstance(f, np.ndarray) for f in frames)

    @pytest.mark.asyncio
    async def test_agent_processes_driver_frames(self, integration_messager):
        """Test agent processing frames from driver."""
        driver = WebRTCDriver(video_buffer_size=10)

        # Add test frames
        test_frames = [np.random.rand(480, 640, 3) for _ in range(10)]
        driver._frame_buffer.extend(test_frames)

        config = {"gemma_model_name": "gemma-3n-e4b-it"}
        llm = GemmaProvider(config)

        agent = VisualAgent(
            llm=llm,
            messager=integration_messager,
            webrtc_driver=driver,
            enable_auto_processing=False,
        )

        mock_response = LLMChatResponse(
            message=AssistantMessage(role="assistant", content="Processed frames")
        )

        with patch.object(agent, "_call_llm", new=AsyncMock(return_value=mock_response)):
            result = await agent.query_with_video("What do you see?")

        assert result == "Processed frames"


class TestGemmaProviderIntegration:
    """Test integration with Gemma Provider."""

    @pytest.mark.asyncio
    async def test_gemma_initialization_in_agent(self, integration_messager):
        """Test Gemma provider initialization through agent."""
        driver = WebRTCDriver()

        config = {
            "gemma_model_name": "gemma-3n-e4b-it",
            "gemma_checkpoint_path": "test_checkpoint",
        }
        llm = GemmaProvider(config)

        agent = VisualAgent(
            llm=llm,
            messager=integration_messager,
            webrtc_driver=driver,
            enable_auto_processing=False,
        )

        # Initialize model
        await llm._initialize_model()

        assert llm._initialized is True
        assert llm._sampler is not None

    @pytest.mark.asyncio
    async def test_gemma_processes_multimodal_input(self, integration_messager):
        """Test Gemma processing multimodal input from agent."""
        driver = WebRTCDriver()

        config = {"gemma_model_name": "gemma-3n-e4b-it"}
        llm = GemmaProvider(config)

        agent = VisualAgent(
            llm=llm,
            messager=integration_messager,
            webrtc_driver=driver,
            enable_auto_processing=False,
        )

        # Create multimodal message
        frames = [np.random.rand(480, 640, 3) for _ in range(5)]
        audio = np.random.randn(1000).astype(np.float32)

        # Test LLM can handle it
        response = await llm.achat(
            [
                {
                    "role": "user",
                    "content": {
                        "text": "Describe this",
                        "images": frames,
                        "audio": audio,
                    },
                }
            ]
        )

        assert isinstance(response, LLMChatResponse)
        assert len(response.message.content) > 0


class TestFullWorkflowIntegration:
    """Test full workflow integration."""

    @pytest.mark.asyncio
    async def test_complete_visual_query_workflow(self, integration_messager):
        """Test complete workflow: driver → agent → llm → messager."""
        # Setup components
        driver = WebRTCDriver(video_buffer_size=15, enable_audio=True)

        config = {"gemma_model_name": "gemma-3n-e4b-it"}
        llm = GemmaProvider(config)

        agent = VisualAgent(
            llm=llm,
            messager=integration_messager,
            webrtc_driver=driver,
            enable_auto_processing=False,
            min_frames_for_processing=10,
        )

        # Add test data to driver
        test_frames = [np.random.rand(480, 640, 3) for _ in range(15)]
        test_audio = np.random.randn(5000).astype(np.float32)
        driver._frame_buffer.extend(test_frames)
        driver._audio_buffer.extend(test_audio)

        # Mock LLM response
        mock_response = LLMChatResponse(
            message=AssistantMessage(
                role="assistant",
                content="I see a scene with various objects and hear ambient sounds.",
            )
        )

        with patch.object(agent, "_call_llm", new=AsyncMock(return_value=mock_response)):
            # Execute query
            result = await agent.query_with_video("Describe what you see and hear")

        # Verify workflow
        assert "see a scene" in result
        assert "hear ambient" in result

        # Verify messager was called to publish
        integration_messager.publish.assert_called_once()

        # Verify published message
        call_args = integration_messager.publish.call_args
        assert call_args[0][0] == agent.visual_response_topic
        message = call_args[0][1]
        assert "see a scene" in message.answer

    @pytest.mark.asyncio
    async def test_auto_processing_workflow(self, integration_messager):
        """Test auto-processing workflow."""
        driver = WebRTCDriver(video_buffer_size=20)

        config = {"gemma_model_name": "gemma-3n-e4b-it"}
        llm = GemmaProvider(config)

        agent = VisualAgent(
            llm=llm,
            messager=integration_messager,
            webrtc_driver=driver,
            auto_process_interval=0.2,  # Fast for testing
            enable_auto_processing=True,
            min_frames_for_processing=5,
        )

        # Add frames
        test_frames = [np.random.rand(480, 640, 3) for _ in range(10)]
        driver._frame_buffer.extend(test_frames)

        mock_response = LLMChatResponse(
            message=AssistantMessage(role="assistant", content="Auto-processed content")
        )

        with patch.object(agent, "_tool_manager", AsyncMock()):
            with patch.object(agent, "_call_llm", new=AsyncMock(return_value=mock_response)):
                await agent.async_init()

                # Wait for auto-processing
                await asyncio.sleep(0.5)

        # Cleanup
        await agent.stop()

        # Should have published
        assert integration_messager.publish.call_count >= 1

    @pytest.mark.asyncio
    async def test_error_handling_in_workflow(self, integration_messager):
        """Test error handling throughout the workflow."""
        driver = WebRTCDriver()

        config = {"gemma_model_name": "gemma-3n-e4b-it"}
        llm = GemmaProvider(config)

        agent = VisualAgent(
            llm=llm,
            messager=integration_messager,
            webrtc_driver=driver,
            enable_auto_processing=False,
        )

        # Add frames
        test_frames = [np.random.rand(480, 640, 3) for _ in range(10)]
        driver._frame_buffer.extend(test_frames)

        # Mock LLM to raise error
        with patch.object(agent, "_call_llm", new=AsyncMock(side_effect=Exception("LLM error"))):
            with pytest.raises(Exception, match="LLM error"):
                await agent.query_with_video("Test query")


class TestBufferManagementIntegration:
    """Test buffer management across components."""

    @pytest.mark.asyncio
    async def test_buffer_overflow_handling(self, integration_messager):
        """Test that buffer overflow is handled properly."""
        driver = WebRTCDriver(video_buffer_size=10)

        config = {"gemma_model_name": "gemma-3n-e4b-it"}
        llm = GemmaProvider(config)

        agent = VisualAgent(
            llm=llm,
            messager=integration_messager,
            webrtc_driver=driver,
            enable_auto_processing=False,
        )

        # Add more frames than buffer size
        for i in range(20):
            driver._frame_buffer.append(np.random.rand(480, 640, 3))

        frames = await driver.get_frame_buffer()

        # Should only have buffer_size frames (deque auto-removes oldest)
        assert len(frames) == 10

    @pytest.mark.asyncio
    async def test_buffer_clearing(self, integration_messager):
        """Test buffer clearing functionality."""
        driver = WebRTCDriver(video_buffer_size=10)

        config = {"gemma_model_name": "gemma-3n-e4b-it"}
        llm = GemmaProvider(config)

        agent = VisualAgent(
            llm=llm,
            messager=integration_messager,
            webrtc_driver=driver,
            enable_auto_processing=False,
        )

        # Add frames
        test_frames = [np.random.rand(480, 640, 3) for _ in range(10)]
        driver._frame_buffer.extend(test_frames)

        # Clear buffers
        driver.clear_buffers()

        frames = await driver.get_frame_buffer()
        assert len(frames) == 0


class TestMultimodalContentIntegration:
    """Test multimodal content handling across components."""

    @pytest.mark.asyncio
    async def test_image_encoding_pipeline(self, integration_messager):
        """Test image encoding from numpy to Gemma format."""
        config = {"gemma_model_name": "gemma-3n-e4b-it"}
        llm = GemmaProvider(config)

        # Test different image formats
        numpy_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        encoded = llm._encode_image(numpy_img)

        assert encoded is not None

    @pytest.mark.asyncio
    async def test_audio_encoding_pipeline(self, integration_messager):
        """Test audio encoding from numpy to Gemma format."""
        config = {"gemma_model_name": "gemma-3n-e4b-it"}
        llm = GemmaProvider(config)

        # Test audio format
        audio_array = np.random.randn(1000).astype(np.float32)

        encoded = llm._encode_audio(audio_array)

        assert encoded is not None
        assert isinstance(encoded, np.ndarray)
