"""Unit tests for WebRTC Frame Source."""

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from argentic.core.agent.frame_source import FrameSourceConfig
from argentic.core.agent.frame_sources.webrtc_source import WebRTCFrameSource
from argentic.core.drivers.webrtc_driver import WebRTCDriver


class TestWebRTCFrameSourceInitialization:
    """Test WebRTCFrameSource initialization."""

    def test_init_with_driver(self):
        """Test initialization with WebRTC driver."""
        driver = WebRTCDriver()
        source = WebRTCFrameSource(driver)

        assert source.driver == driver
        assert source.config is not None

    def test_init_with_driver_and_config(self):
        """Test initialization with driver and custom config."""
        driver = WebRTCDriver()
        config = FrameSourceConfig(buffer_size=60)
        source = WebRTCFrameSource(driver, config)

        assert source.driver == driver
        assert source.config.buffer_size == 60

    def test_init_creates_default_config(self):
        """Test that default config is created if not provided."""
        driver = WebRTCDriver()
        source = WebRTCFrameSource(driver)

        # Should create default config
        assert isinstance(source.config, FrameSourceConfig)


class TestWebRTCFrameSourceGetFrames:
    """Test getting frames from WebRTC source."""

    @pytest.mark.asyncio
    async def test_get_frames_empty_buffer(self):
        """Test getting frames when buffer is empty."""
        driver = WebRTCDriver()
        source = WebRTCFrameSource(driver)

        frames = await source.get_frames()

        assert frames == []

    @pytest.mark.asyncio
    async def test_get_frames_with_data(self):
        """Test getting frames when buffer has data."""
        driver = WebRTCDriver()
        source = WebRTCFrameSource(driver)

        # Add frames to driver's buffer
        test_frames = [np.random.rand(480, 640, 3) for _ in range(3)]
        driver._frame_buffer.extend(test_frames)

        frames = await source.get_frames()

        assert len(frames) == 3
        assert all(isinstance(f, np.ndarray) for f in frames)

    @pytest.mark.asyncio
    async def test_get_frames_calls_driver_method(self):
        """Test that get_frames delegates to driver."""
        driver = WebRTCDriver()
        driver.get_frame_buffer = AsyncMock(return_value=[np.random.rand(480, 640, 3)])

        source = WebRTCFrameSource(driver)
        frames = await source.get_frames()

        driver.get_frame_buffer.assert_awaited_once()
        assert len(frames) == 1


class TestWebRTCFrameSourceGetAudio:
    """Test getting audio from WebRTC source."""

    @pytest.mark.asyncio
    async def test_get_audio_disabled(self):
        """Test getting audio when audio is disabled."""
        driver = WebRTCDriver(enable_audio=False)
        source = WebRTCFrameSource(driver)

        audio = await source.get_audio()

        assert audio is None

    @pytest.mark.asyncio
    async def test_get_audio_with_data(self):
        """Test getting audio when buffer has data."""
        driver = WebRTCDriver(enable_audio=True)
        source = WebRTCFrameSource(driver)

        # Add audio to driver's buffer
        test_audio = np.random.randn(1000)
        driver._audio_buffer.extend(test_audio)

        audio = await source.get_audio()

        assert isinstance(audio, np.ndarray)
        assert len(audio) == 1000

    @pytest.mark.asyncio
    async def test_get_audio_calls_driver_method(self):
        """Test that get_audio delegates to driver."""
        driver = WebRTCDriver()
        driver.get_audio_buffer = AsyncMock(return_value=np.random.randn(500))

        source = WebRTCFrameSource(driver)
        audio = await source.get_audio()

        driver.get_audio_buffer.assert_awaited_once()
        assert len(audio) == 500


class TestWebRTCFrameSourceGetLatestFrame:
    """Test getting latest frame from WebRTC source."""

    @pytest.mark.asyncio
    async def test_get_latest_frame_empty(self):
        """Test getting latest frame when buffer is empty."""
        driver = WebRTCDriver()
        source = WebRTCFrameSource(driver)

        latest = await source.get_latest_frame()

        assert latest is None

    @pytest.mark.asyncio
    async def test_get_latest_frame_with_data(self):
        """Test getting latest frame when buffer has data."""
        driver = WebRTCDriver()
        source = WebRTCFrameSource(driver)

        # Add frames to buffer
        frames = [np.random.rand(480, 640, 3) for _ in range(3)]
        driver._frame_buffer.extend(frames)

        latest = await source.get_latest_frame()

        assert latest is not None
        assert np.array_equal(latest, frames[-1])


class TestWebRTCFrameSourceClearBuffers:
    """Test clearing buffers."""

    def test_clear_buffers_calls_driver(self):
        """Test that clear_buffers delegates to driver."""
        driver = WebRTCDriver()
        driver.clear_buffers = MagicMock()

        source = WebRTCFrameSource(driver)
        source.clear_buffers()

        driver.clear_buffers.assert_called_once()

    def test_clear_buffers_removes_data(self):
        """Test that clear_buffers actually removes data."""
        driver = WebRTCDriver()
        source = WebRTCFrameSource(driver)

        # Add data to buffers
        driver._frame_buffer.extend([np.random.rand(480, 640, 3) for _ in range(3)])
        driver._audio_buffer.extend(np.random.randn(1000))

        source.clear_buffers()

        assert len(driver._frame_buffer) == 0
        assert len(driver._audio_buffer) == 0


class TestWebRTCFrameSourceLifecycle:
    """Test start/stop lifecycle methods."""

    @pytest.mark.asyncio
    async def test_start_connects_and_starts_capture(self):
        """Test that start connects driver and starts capture."""
        driver = WebRTCDriver()
        driver.connect = AsyncMock()
        driver.start_capture = AsyncMock()

        source = WebRTCFrameSource(driver)
        await source.start()

        driver.connect.assert_awaited_once()
        driver.start_capture.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stop_stops_capture_and_disconnects(self):
        """Test that stop stops capture and disconnects driver."""
        driver = WebRTCDriver()
        driver.stop_capture = AsyncMock()
        driver.disconnect = AsyncMock()

        source = WebRTCFrameSource(driver)
        await source.stop()

        driver.stop_capture.assert_awaited_once()
        driver.disconnect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_full_lifecycle(self):
        """Test complete start/stop lifecycle."""
        driver = WebRTCDriver()
        source = WebRTCFrameSource(driver)

        await source.start()
        assert driver._connected is True

        await source.stop()
        assert driver._connected is False


class TestWebRTCFrameSourceGetInfo:
    """Test getting source information."""

    def test_get_info_structure(self):
        """Test that get_info returns correct structure."""
        driver = WebRTCDriver(
            video_buffer_size=60,
            enable_audio=False,
            frame_rate=15,
            resize_frames=(320, 240),
        )
        source = WebRTCFrameSource(driver)

        info = source.get_info()

        assert info["type"] == "WebRTCFrameSource"
        assert info["driver"] == "WebRTCDriver"
        assert info["config"]["buffer_size"] == 60
        assert info["config"]["enable_audio"] is False
        assert info["config"]["frame_rate"] == 15
        assert info["config"]["resize_frames"] == (320, 240)

    def test_get_info_with_audio_enabled(self):
        """Test get_info when audio is enabled."""
        driver = WebRTCDriver(enable_audio=True)
        source = WebRTCFrameSource(driver)

        info = source.get_info()

        assert info["config"]["enable_audio"] is True


class TestWebRTCFrameSourceIntegration:
    """Integration tests for WebRTC frame source."""

    @pytest.mark.asyncio
    async def test_integration_with_webrtc_driver(self):
        """Test integration with actual WebRTC driver."""
        driver = WebRTCDriver(video_buffer_size=10)
        source = WebRTCFrameSource(driver)

        # Start the source
        await source.start()

        # Should be connected
        assert driver._connected is True

        # Add some frames
        test_frames = [np.random.rand(480, 640, 3) for _ in range(5)]
        driver._frame_buffer.extend(test_frames)

        # Get frames through source
        frames = await source.get_frames()
        assert len(frames) == 5

        # Clear buffers through source
        source.clear_buffers()
        frames = await source.get_frames()
        assert len(frames) == 0

        # Stop the source
        await source.stop()
        assert driver._connected is False

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test that WebRTC source works as async context manager (if implemented)."""
        driver = WebRTCDriver()
        source = WebRTCFrameSource(driver)

        # Manually test lifecycle
        await source.start()
        assert driver._connected is True

        await source.stop()
        assert driver._connected is False
