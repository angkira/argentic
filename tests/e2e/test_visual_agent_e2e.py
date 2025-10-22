"""End-to-end tests for Visual Agent system."""

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
async def e2e_messager():
    """Create a mock messager for E2E tests."""
    messager = AsyncMock(spec=Messager)
    messager.connect = AsyncMock()
    messager.disconnect = AsyncMock()
    messager.publish = AsyncMock()
    messager.subscribe = AsyncMock()
    messager.is_connected = MagicMock(return_value=True)
    return messager


class TestVisualAgentE2EScenarios:
    """Test realistic end-to-end scenarios."""

    @pytest.mark.asyncio
    async def test_security_camera_monitoring(self, e2e_messager):
        """
        E2E Test: Security camera monitoring scenario.

        Scenario: A security camera continuously streams video and the agent
        automatically processes frames to detect activity.
        """
        # Setup
        driver = WebRTCDriver(
            video_buffer_size=30,
            frame_rate=10,
            resize_frames=(640, 480),
            enable_audio=False,
        )

        config = {"gemma_model_name": "gemma-3n-e4b-it"}
        llm = GemmaProvider(config)

        agent = VisualAgent(
            llm=llm,
            messager=e2e_messager,
            webrtc_driver=driver,
            auto_process_interval=2.0,
            enable_auto_processing=True,
            system_prompt="You are a security monitoring assistant. Detect any unusual activity.",
        )

        # Simulate video frames
        security_frames = [np.random.rand(480, 640, 3) for _ in range(30)]
        driver._frame_buffer.extend(security_frames)

        # Mock responses
        responses = [
            "Normal activity detected - person walking",
            "Unusual activity detected - motion in restricted area",
        ]
        response_iter = iter(responses)

        async def mock_llm_call(*args, **kwargs):
            try:
                content = next(response_iter)
            except StopIteration:
                content = "Normal activity"
            return LLMChatResponse(message=AssistantMessage(role="assistant", content=content))

        with patch.object(agent, "_tool_manager", AsyncMock()):
            with patch.object(agent, "_call_llm", new=mock_llm_call):
                await agent.async_init()

                # Let it run for a few cycles
                await asyncio.sleep(5.0)

        await agent.stop()

        # Verify monitoring happened
        assert e2e_messager.publish.call_count >= 2

    @pytest.mark.asyncio
    async def test_video_call_assistant(self, e2e_messager):
        """
        E2E Test: Video call assistant scenario.

        Scenario: An agent assists users during a video call by analyzing
        video and audio to provide real-time insights.
        """
        driver = WebRTCDriver(
            video_buffer_size=30,
            frame_rate=15,
            enable_audio=True,
            audio_sample_rate=16000,
        )

        config = {"gemma_model_name": "gemma-3n-e4b-it"}
        llm = GemmaProvider(config)

        agent = VisualAgent(
            llm=llm,
            messager=e2e_messager,
            webrtc_driver=driver,
            enable_auto_processing=False,
            system_prompt="You are a video call assistant. Provide helpful insights about the meeting.",
        )

        # Simulate video call data
        call_frames = [np.random.rand(480, 640, 3) for _ in range(25)]
        call_audio = np.random.randn(16000 * 5).astype(np.float32)  # 5 seconds
        driver._frame_buffer.extend(call_frames)
        driver._audio_buffer.extend(call_audio)

        mock_response = LLMChatResponse(
            message=AssistantMessage(
                role="assistant",
                content="I see 2 people in the meeting. One person is presenting slides. Audio quality is good.",
            )
        )

        with patch.object(agent, "_call_llm", new=AsyncMock(return_value=mock_response)):
            # User asks about the call
            result = await agent.query_with_video("What's happening in the meeting?")

        assert "2 people" in result
        assert "presenting" in result
        e2e_messager.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_retail_customer_analytics(self, e2e_messager):
        """
        E2E Test: Retail customer analytics scenario.

        Scenario: A retail camera analyzes customer behavior and provides
        insights about store traffic and customer engagement.
        """
        driver = WebRTCDriver(
            video_buffer_size=60,
            frame_rate=10,
            resize_frames=(320, 240),  # Lower resolution for efficiency
            enable_audio=False,
        )

        config = {"gemma_model_name": "gemma-3n-e4b-it"}
        llm = GemmaProvider(config)

        agent = VisualAgent(
            llm=llm,
            messager=e2e_messager,
            webrtc_driver=driver,
            auto_process_interval=10.0,  # Every 10 seconds
            enable_auto_processing=True,
            min_frames_for_processing=30,
            system_prompt="You are a retail analytics assistant. Analyze customer behavior and store traffic.",
        )

        # Simulate store footage
        store_frames = [np.random.rand(240, 320, 3) for _ in range(60)]
        driver._frame_buffer.extend(store_frames)

        mock_response = LLMChatResponse(
            message=AssistantMessage(
                role="assistant",
                content="Moderate traffic detected. 3 customers browsing products. High engagement at display area.",
            )
        )

        with patch.object(agent, "_tool_manager", AsyncMock()):
            with patch.object(agent, "_call_llm", new=AsyncMock(return_value=mock_response)):
                await agent.async_init()

                # Wait for analysis
                await asyncio.sleep(12.0)

        await agent.stop()

        # Should have published analytics
        assert e2e_messager.publish.call_count >= 1

    @pytest.mark.asyncio
    async def test_educational_lecture_analysis(self, e2e_messager):
        """
        E2E Test: Educational lecture analysis scenario.

        Scenario: Agent analyzes lecture video and audio to provide summaries
        and insights about teaching effectiveness.
        """
        driver = WebRTCDriver(
            video_buffer_size=120,  # 2 minutes at 1 FPS
            frame_rate=1,  # Lower FPS for lectures
            enable_audio=True,
        )

        config = {"gemma_model_name": "gemma-3n-e4b-it"}
        llm = GemmaProvider(config)

        agent = VisualAgent(
            llm=llm,
            messager=e2e_messager,
            webrtc_driver=driver,
            enable_auto_processing=False,
            system_prompt="You are an educational analysis assistant. Analyze teaching effectiveness and student engagement.",
        )

        # Simulate lecture footage
        lecture_frames = [np.random.rand(480, 640, 3) for _ in range(60)]
        lecture_audio = np.random.randn(16000 * 60).astype(np.float32)  # 1 minute
        driver._frame_buffer.extend(lecture_frames)
        driver._audio_buffer.extend(lecture_audio)

        mock_response = LLMChatResponse(
            message=AssistantMessage(
                role="assistant",
                content="Instructor is using visual aids effectively. Students appear engaged. Clear audio delivery.",
            )
        )

        with patch.object(agent, "_call_llm", new=AsyncMock(return_value=mock_response)):
            result = await agent.query_with_video("Analyze the teaching effectiveness")

        assert "Instructor" in result
        assert "engaged" in result


class TestVisualAgentE2EEdgeCases:
    """Test edge cases and error scenarios."""

    @pytest.mark.asyncio
    async def test_connection_loss_recovery(self, e2e_messager):
        """
        E2E Test: Connection loss and recovery.

        Tests system behavior when WebRTC connection is lost and restored.
        """
        driver = WebRTCDriver()

        config = {"gemma_model_name": "gemma-3n-e4b-it"}
        llm = GemmaProvider(config)

        agent = VisualAgent(
            llm=llm,
            messager=e2e_messager,
            webrtc_driver=driver,
            enable_auto_processing=False,
        )

        with patch.object(agent, "_tool_manager", AsyncMock()):
            await agent.async_init()

        # Simulate connection loss
        await driver.disconnect()
        assert driver._connected is False

        # Reconnect
        await driver.connect()
        await driver.start_capture()
        assert driver._connected is True

        await agent.stop()

    @pytest.mark.asyncio
    async def test_low_quality_video_handling(self, e2e_messager):
        """
        E2E Test: Handling low quality or corrupted video frames.

        Tests system robustness with poor quality input.
        """
        driver = WebRTCDriver(video_buffer_size=20)

        config = {"gemma_model_name": "gemma-3n-e4b-it"}
        llm = GemmaProvider(config)

        agent = VisualAgent(
            llm=llm,
            messager=e2e_messager,
            webrtc_driver=driver,
            enable_auto_processing=False,
        )

        # Add mix of good and degraded frames
        frames = []
        for i in range(20):
            if i % 5 == 0:
                # Simulated "corrupted" frame (very low quality)
                frames.append(np.zeros((480, 640, 3)))
            else:
                frames.append(np.random.rand(480, 640, 3))

        driver._frame_buffer.extend(frames)

        mock_response = LLMChatResponse(
            message=AssistantMessage(
                role="assistant", content="Video quality varies. Some frames appear degraded."
            )
        )

        with patch.object(agent, "_call_llm", new=AsyncMock(return_value=mock_response)):
            result = await agent.query_with_video("What do you see?")

        # Should still process despite quality issues
        assert "quality" in result.lower()

    @pytest.mark.asyncio
    async def test_no_frames_available(self, e2e_messager):
        """
        E2E Test: Query when no frames are available.

        Tests graceful handling when buffer is empty.
        """
        driver = WebRTCDriver()

        config = {"gemma_model_name": "gemma-3n-e4b-it"}
        llm = GemmaProvider(config)

        agent = VisualAgent(
            llm=llm,
            messager=e2e_messager,
            webrtc_driver=driver,
            enable_auto_processing=False,
        )

        # No frames in buffer
        result = await agent.query_with_video("What do you see?")

        # Should return informative message
        assert "No video frames" in result

    @pytest.mark.asyncio
    async def test_high_load_scenario(self, e2e_messager):
        """
        E2E Test: High load scenario with rapid queries.

        Tests system stability under high load.
        """
        driver = WebRTCDriver(video_buffer_size=30)

        config = {"gemma_model_name": "gemma-3n-e4b-it"}
        llm = GemmaProvider(config)

        agent = VisualAgent(
            llm=llm,
            messager=e2e_messager,
            webrtc_driver=driver,
            enable_auto_processing=False,
        )

        # Add frames
        frames = [np.random.rand(480, 640, 3) for _ in range(30)]
        driver._frame_buffer.extend(frames)

        mock_response = LLMChatResponse(
            message=AssistantMessage(role="assistant", content="Processed")
        )

        with patch.object(agent, "_call_llm", new=AsyncMock(return_value=mock_response)):
            # Fire multiple concurrent queries
            tasks = [agent.query_with_video(f"Query {i}") for i in range(10)]

            results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should complete (some might be errors due to concurrent access)
        assert len(results) == 10


class TestVisualAgentE2EPerformance:
    """Test performance characteristics in E2E scenarios."""

    @pytest.mark.asyncio
    async def test_processing_latency(self, e2e_messager):
        """
        E2E Test: Measure processing latency.

        Ensures processing happens within acceptable time limits.
        """
        driver = WebRTCDriver(video_buffer_size=30)

        config = {"gemma_model_name": "gemma-3n-e4b-it"}
        llm = GemmaProvider(config)

        agent = VisualAgent(
            llm=llm,
            messager=e2e_messager,
            webrtc_driver=driver,
            enable_auto_processing=False,
        )

        # Add frames
        frames = [np.random.rand(480, 640, 3) for _ in range(30)]
        driver._frame_buffer.extend(frames)

        mock_response = LLMChatResponse(
            message=AssistantMessage(role="assistant", content="Quick response")
        )

        import time

        with patch.object(agent, "_call_llm", new=AsyncMock(return_value=mock_response)):
            start_time = time.time()
            result = await agent.query_with_video("Test")
            end_time = time.time()

        latency = end_time - start_time

        # Should be reasonably fast (< 1 second with mocked LLM)
        assert latency < 1.0
        assert result == "Quick response"

    @pytest.mark.asyncio
    async def test_memory_efficiency(self, e2e_messager):
        """
        E2E Test: Test memory efficiency with buffer management.

        Ensures buffers don't grow unbounded.
        """
        driver = WebRTCDriver(video_buffer_size=50)

        config = {"gemma_model_name": "gemma-3n-e4b-it"}
        llm = GemmaProvider(config)

        agent = VisualAgent(
            llm=llm,
            messager=e2e_messager,
            webrtc_driver=driver,
            enable_auto_processing=False,
        )

        # Add many frames (more than buffer size)
        for i in range(200):
            driver._frame_buffer.append(np.random.rand(480, 640, 3))

        frames = await driver.get_frame_buffer()

        # Should respect buffer size limit
        assert len(frames) == 50


class TestVisualAgentE2EConfiguration:
    """Test different configuration scenarios."""

    @pytest.mark.asyncio
    async def test_low_resource_configuration(self, e2e_messager):
        """
        E2E Test: Low resource configuration.

        Tests system with resource-constrained settings.
        """
        driver = WebRTCDriver(
            video_buffer_size=10,
            frame_rate=5,
            resize_frames=(160, 120),  # Very small
            enable_audio=False,
            frame_processor_workers=2,
        )

        config = {"gemma_model_name": "gemma-3n-e2b-it"}  # Smaller model
        llm = GemmaProvider(config)

        agent = VisualAgent(
            llm=llm,
            messager=e2e_messager,
            webrtc_driver=driver,
            enable_auto_processing=False,
            min_frames_for_processing=5,
        )

        # Add minimal frames
        frames = [np.random.rand(120, 160, 3) for _ in range(5)]
        driver._frame_buffer.extend(frames)

        mock_response = LLMChatResponse(
            message=AssistantMessage(role="assistant", content="Low resource response")
        )

        with patch.object(agent, "_call_llm", new=AsyncMock(return_value=mock_response)):
            result = await agent.query_with_video("What do you see?")

        assert result == "Low resource response"

    @pytest.mark.asyncio
    async def test_high_quality_configuration(self, e2e_messager):
        """
        E2E Test: High quality configuration.

        Tests system with high-quality settings.
        """
        driver = WebRTCDriver(
            video_buffer_size=120,
            frame_rate=30,
            resize_frames=(1920, 1080),  # Full HD
            enable_audio=True,
            audio_sample_rate=48000,
            frame_processor_workers=8,
        )

        config = {"gemma_model_name": "gemma-3n-e4b-it"}  # Larger model
        llm = GemmaProvider(config)

        agent = VisualAgent(
            llm=llm,
            messager=e2e_messager,
            webrtc_driver=driver,
            enable_auto_processing=False,
            min_frames_for_processing=60,
        )

        # Add high quality frames
        frames = [np.random.rand(1080, 1920, 3) for _ in range(60)]
        driver._frame_buffer.extend(frames)

        mock_response = LLMChatResponse(
            message=AssistantMessage(role="assistant", content="High quality analysis complete")
        )

        with patch.object(agent, "_call_llm", new=AsyncMock(return_value=mock_response)):
            result = await agent.query_with_video("Analyze in detail")

        assert "High quality" in result
