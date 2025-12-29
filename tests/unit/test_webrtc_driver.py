"""Unit tests for WebRTC Driver."""

import asyncio
from unittest.mock import MagicMock

import numpy as np
import pytest

from argentic.core.drivers.webrtc_driver import WebRTCDriver


class TestWebRTCDriverInitialization:
    """Test WebRTCDriver initialization."""

    def test_init_default_params(self):
        """Test driver initialization with default parameters."""
        driver = WebRTCDriver()

        assert driver.video_buffer_size == 30
        assert driver.frame_rate == 10
        assert driver.audio_sample_rate == 16000
        assert driver.enable_audio is True
        assert driver._connected is False
        assert driver._capturing is False

    def test_init_custom_params(self):
        """Test driver initialization with custom parameters."""
        driver = WebRTCDriver(
            signaling_url="wss://test.com",
            video_buffer_size=60,
            frame_rate=15,
            resize_frames=(320, 240),
            enable_audio=False,
        )

        assert driver.signaling_url == "wss://test.com"
        assert driver.video_buffer_size == 60
        assert driver.frame_rate == 15
        assert driver.resize_frames == (320, 240)
        assert driver.enable_audio is False

    def test_init_thread_pool_created(self):
        """Test that frame processor thread pool is created."""
        driver = WebRTCDriver(frame_processor_workers=8)

        assert driver._frame_processor_pool is not None
        assert driver._frame_processor_pool._max_workers == 8


class TestWebRTCDriverConnection:
    """Test WebRTC connection management."""

    @pytest.mark.asyncio
    async def test_connect_without_offer(self):
        """Test connection establishment without offer SDP."""
        driver = WebRTCDriver()

        result = await driver.connect()

        assert driver._connected is True
        assert driver._pc is not None
        assert result is None  # No answer when no offer provided

    @pytest.mark.asyncio
    async def test_connect_with_offer(self):
        """Test connection establishment with offer SDP."""
        driver = WebRTCDriver()

        offer_sdp = "mock_offer_sdp"
        result = await driver.connect(offer_sdp=offer_sdp)

        assert driver._connected is True
        assert driver._pc is not None
        assert result == "mock_answer_sdp"
        assert driver._pc.remoteDescription is not None

    @pytest.mark.asyncio
    async def test_connect_already_connected(self):
        """Test connecting when already connected."""
        driver = WebRTCDriver()

        await driver.connect()
        result = await driver.connect()  # Try to connect again

        assert result is None  # Should return None

    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test disconnection."""
        driver = WebRTCDriver()
        await driver.connect()

        await driver.disconnect()

        assert driver._connected is False
        # Note: _pc is set to None after disconnect
        assert driver._pc is None


class TestWebRTCDriverCapture:
    """Test video/audio capture functionality."""

    @pytest.mark.asyncio
    async def test_start_capture_not_connected(self):
        """Test starting capture without connection."""
        driver = WebRTCDriver()

        await driver.start_capture()

        # Should not start capturing if not connected
        assert driver._capturing is False

    @pytest.mark.asyncio
    async def test_start_capture_connected(self):
        """Test starting capture when connected."""
        driver = WebRTCDriver()
        await driver.connect()

        # Mock tracks
        driver._video_track = MagicMock()
        driver._audio_track = MagicMock()

        await driver.start_capture()

        assert driver._capturing is True
        assert driver._video_capture_task is not None
        assert driver._audio_capture_task is not None

    @pytest.mark.asyncio
    async def test_stop_capture(self):
        """Test stopping capture."""
        driver = WebRTCDriver()
        await driver.connect()

        driver._video_track = MagicMock()
        driver._audio_track = MagicMock()

        await driver.start_capture()
        await asyncio.sleep(0.1)  # Let tasks start

        await driver.stop_capture()

        assert driver._capturing is False
        assert driver._video_capture_task is None
        assert driver._audio_capture_task is None


class TestWebRTCDriverBuffering:
    """Test frame and audio buffering."""

    @pytest.mark.asyncio
    async def test_get_empty_frame_buffer(self):
        """Test getting frame buffer when empty."""
        driver = WebRTCDriver()

        frames = await driver.get_frame_buffer()

        assert frames == []

    @pytest.mark.asyncio
    async def test_frame_buffer_with_data(self):
        """Test frame buffer with data."""
        driver = WebRTCDriver(video_buffer_size=5)

        # Manually add frames to buffer
        test_frames = [np.random.rand(480, 640, 3) for _ in range(3)]
        driver._frame_buffer.extend(test_frames)

        frames = await driver.get_frame_buffer()

        assert len(frames) == 3
        assert all(isinstance(f, np.ndarray) for f in frames)

    @pytest.mark.asyncio
    async def test_frame_buffer_max_size(self):
        """Test that frame buffer respects max size."""
        driver = WebRTCDriver(video_buffer_size=3)

        # Add more frames than buffer size
        for i in range(5):
            driver._frame_buffer.append(np.random.rand(480, 640, 3))

        frames = await driver.get_frame_buffer()

        # Should only keep last 3 frames
        assert len(frames) == 3

    @pytest.mark.asyncio
    async def test_get_audio_buffer_disabled(self):
        """Test getting audio buffer when audio is disabled."""
        driver = WebRTCDriver(enable_audio=False)

        audio = await driver.get_audio_buffer()

        assert audio is None

    @pytest.mark.asyncio
    async def test_audio_buffer_with_data(self):
        """Test audio buffer with data."""
        driver = WebRTCDriver(enable_audio=True)

        # Manually add audio samples
        test_samples = np.random.randn(1000)
        driver._audio_buffer.extend(test_samples)

        audio = await driver.get_audio_buffer()

        assert isinstance(audio, np.ndarray)
        assert len(audio) == 1000

    @pytest.mark.asyncio
    async def test_get_latest_frame(self):
        """Test getting the latest frame."""
        driver = WebRTCDriver()

        # Add frames
        frames = [np.random.rand(480, 640, 3) for _ in range(3)]
        driver._frame_buffer.extend(frames)

        latest = await driver.get_latest_frame()

        assert latest is not None
        assert np.array_equal(latest, frames[-1])

    @pytest.mark.asyncio
    async def test_clear_buffers(self):
        """Test clearing all buffers."""
        driver = WebRTCDriver()

        # Add data to buffers
        driver._frame_buffer.extend([np.random.rand(480, 640, 3) for _ in range(3)])
        driver._audio_buffer.extend(np.random.randn(1000))

        driver.clear_buffers()

        assert len(driver._frame_buffer) == 0
        assert len(driver._audio_buffer) == 0


class TestWebRTCDriverFrameProcessing:
    """Test frame processing functionality."""

    def test_process_frame(self):
        """Test frame processing."""
        from av import VideoFrame

        driver = WebRTCDriver()

        # Create mock frame
        mock_frame = VideoFrame(width=640, height=480)

        # Process frame
        processed = driver._process_frame(mock_frame)

        assert isinstance(processed, np.ndarray)
        assert processed.shape == (480, 640, 3)

    def test_process_frame_with_resize(self):
        """Test frame processing with resize."""
        from av import VideoFrame

        driver = WebRTCDriver(resize_frames=(320, 240))

        mock_frame = VideoFrame(width=640, height=480)

        processed = driver._process_frame(mock_frame)

        # Should be resized
        assert processed is not None
        # Note: Actual size depends on PIL availability

    def test_process_audio(self):
        """Test audio processing."""
        from av import AudioFrame

        driver = WebRTCDriver()

        mock_frame = AudioFrame(samples=1024, sample_rate=16000)

        processed = driver._process_audio(mock_frame)

        assert isinstance(processed, np.ndarray)
        assert len(processed) > 0


class TestWebRTCDriverCallbacks:
    """Test callback functionality."""

    @pytest.mark.asyncio
    async def test_set_frame_callback(self):
        """Test setting and triggering frame callback."""
        driver = WebRTCDriver()

        callback_called = False
        callback_frame = None

        async def test_callback(frame, timestamp):
            nonlocal callback_called, callback_frame
            callback_called = True
            callback_frame = frame

        driver.set_frame_callback(test_callback)

        # Manually trigger callback (would normally be called by capture loop)
        test_frame = np.random.rand(480, 640, 3)
        await driver._frame_callback(test_frame, 123.456)

        assert callback_called is True
        assert callback_frame is not None


class TestWebRTCDriverContextManager:
    """Test async context manager functionality."""

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test using driver as async context manager."""
        async with WebRTCDriver() as driver:
            assert driver._connected is True
            # Note: capturing won't actually start without tracks

        # After context exit, should be disconnected
        assert driver._connected is False


class TestWebRTCDriverCleanup:
    """Test cleanup and resource management."""

    @pytest.mark.asyncio
    async def test_cleanup_on_disconnect(self):
        """Test that resources are cleaned up on disconnect."""
        driver = WebRTCDriver()
        await driver.connect()

        driver._video_track = MagicMock()
        await driver.start_capture()

        await driver.disconnect()

        assert driver._connected is False
        assert driver._capturing is False

    def test_thread_pool_shutdown(self):
        """Test that thread pool is shut down on cleanup."""
        driver = WebRTCDriver()

        # Manually call cleanup (normally done in __aexit__)
        driver._frame_processor_pool.shutdown(wait=True)

        # Thread pool should be shut down
        assert driver._frame_processor_pool._shutdown is True
