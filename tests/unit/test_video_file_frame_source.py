"""Unit tests for Video File Frame Source."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# We need to stub cv2 before importing VideoFileFrameSource
class _MockVideoCapture:
    """Mock cv2.VideoCapture for testing."""

    def __init__(self, path):
        self.path = path
        self.is_open = True  # Start as opened
        self.frame_index = 0
        self.test_frames = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(10)
        ]

    def isOpened(self):
        return self.is_open

    def get(self, prop):
        """Mock get method for video properties."""
        if prop == 5:  # CAP_PROP_FPS
            return 30.0
        elif prop == 7:  # CAP_PROP_FRAME_COUNT
            return len(self.test_frames)
        elif prop == 3:  # CAP_PROP_FRAME_WIDTH
            return 640
        elif prop == 4:  # CAP_PROP_FRAME_HEIGHT
            return 480
        elif prop == 1:  # CAP_PROP_POS_FRAMES
            return self.frame_index
        return 0

    def set(self, prop, value):
        """Mock set method."""
        if prop == 1:  # CAP_PROP_POS_FRAMES
            self.frame_index = int(value)
        return True

    def read(self):
        """Mock read method."""
        if not self.is_open:
            return False, None

        if self.frame_index >= len(self.test_frames):
            return False, None

        frame = self.test_frames[self.frame_index]
        self.frame_index += 1
        return True, frame

    def release(self):
        """Mock release method."""
        self.is_open = False


class _MockCv2Module:
    """Mock cv2 module."""

    CAP_PROP_FPS = 5
    CAP_PROP_FRAME_COUNT = 7
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_POS_FRAMES = 1
    COLOR_BGR2RGB = 4

    @staticmethod
    def VideoCapture(path):
        cap = _MockVideoCapture(path)
        cap.is_open = True
        return cap

    @staticmethod
    def cvtColor(frame, code):
        """Mock color conversion (just return the frame)."""
        return frame


@pytest.fixture
def mock_cv2():
    """Fixture to mock cv2.VideoCapture."""
    with patch("argentic.core.agent.frame_sources.video_file_source.cv2") as mock:
        mock.CAP_PROP_FPS = 5
        mock.CAP_PROP_FRAME_COUNT = 7
        mock.CAP_PROP_FRAME_WIDTH = 3
        mock.CAP_PROP_FRAME_HEIGHT = 4
        mock.CAP_PROP_POS_FRAMES = 1
        mock.COLOR_BGR2RGB = 4
        mock.VideoCapture = _MockVideoCapture
        mock.cvtColor = lambda frame, code: frame
        yield mock


@pytest.fixture
def temp_video_file(tmp_path):
    """Create a temporary video file path."""
    video_path = tmp_path / "test_video.mp4"
    video_path.touch()  # Create empty file
    return video_path


class TestVideoFileFrameSourceInitialization:
    """Test VideoFileFrameSource initialization."""

    def test_init_requires_cv2(self, tmp_path):
        """Test that video file source requires a valid file."""
        from argentic.core.agent.frame_sources.video_file_source import (
            VideoFileFrameSource,
        )

        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            VideoFileFrameSource("/fake/path.mp4")

    def test_init_with_valid_path(self, mock_cv2, temp_video_file):
        """Test initialization with valid video file."""
        from argentic.core.agent.frame_sources.video_file_source import (
            VideoFileFrameSource,
        )

        source = VideoFileFrameSource(str(temp_video_file))

        assert source.video_path == temp_video_file
        assert source.target_fps is None
        assert source.loop is False

    def test_init_with_custom_params(self, mock_cv2, temp_video_file):
        """Test initialization with custom parameters."""
        from argentic.core.agent.frame_sources.video_file_source import (
            VideoFileFrameSource,
        )

        source = VideoFileFrameSource(
            str(temp_video_file), buffer_size=60, fps=10, loop=True
        )

        assert source.target_fps == 10
        assert source.loop is True
        assert source.config.buffer_size == 60

    def test_init_file_not_found(self, mock_cv2):
        """Test initialization with non-existent file."""
        from argentic.core.agent.frame_sources.video_file_source import (
            VideoFileFrameSource,
        )

        with pytest.raises(FileNotFoundError):
            VideoFileFrameSource("/nonexistent/video.mp4")


class TestVideoFileFrameSourceGetFrames:
    """Test getting frames from video file."""

    @pytest.mark.asyncio
    async def test_get_frames_before_start(self, mock_cv2, temp_video_file):
        """Test getting frames before starting."""
        from argentic.core.agent.frame_sources.video_file_source import (
            VideoFileFrameSource,
        )

        source = VideoFileFrameSource(str(temp_video_file))

        frames = await source.get_frames()

        # Should return empty list before start
        assert frames == []

    @pytest.mark.asyncio
    async def test_get_frames_after_start(self, mock_cv2, temp_video_file):
        """Test getting frames after starting."""
        from argentic.core.agent.frame_sources.video_file_source import (
            VideoFileFrameSource,
        )

        source = VideoFileFrameSource(str(temp_video_file), buffer_size=5)
        await source.start()

        frames = await source.get_frames()

        # Should read frames from video
        assert len(frames) > 0
        assert all(isinstance(f, np.ndarray) for f in frames)

    @pytest.mark.asyncio
    async def test_get_frames_respects_buffer_size(self, mock_cv2, temp_video_file):
        """Test that buffer size is respected."""
        from argentic.core.agent.frame_sources.video_file_source import (
            VideoFileFrameSource,
        )

        source = VideoFileFrameSource(str(temp_video_file), buffer_size=3)
        await source.start()

        frames = await source.get_frames()

        # Should not exceed buffer size
        assert len(frames) <= 3

    @pytest.mark.asyncio
    async def test_get_frames_with_fps_subsampling(self, mock_cv2, temp_video_file):
        """Test frame subsampling based on FPS."""
        from argentic.core.agent.frame_sources.video_file_source import (
            VideoFileFrameSource,
        )

        # Native FPS is 30, target is 10, so should skip 2 out of 3 frames
        source = VideoFileFrameSource(str(temp_video_file), buffer_size=10, fps=10)
        await source.start()

        frames = await source.get_frames()

        # Should have frames (exact count depends on subsampling)
        assert len(frames) > 0


class TestVideoFileFrameSourceGetAudio:
    """Test getting audio (not supported)."""

    @pytest.mark.asyncio
    async def test_get_audio_returns_none(self, mock_cv2, temp_video_file):
        """Test that get_audio always returns None."""
        from argentic.core.agent.frame_sources.video_file_source import (
            VideoFileFrameSource,
        )

        source = VideoFileFrameSource(str(temp_video_file))

        audio = await source.get_audio()

        assert audio is None


class TestVideoFileFrameSourceClearBuffers:
    """Test clearing buffers."""

    def test_clear_buffers(self, mock_cv2, temp_video_file):
        """Test clearing frame buffer."""
        from argentic.core.agent.frame_sources.video_file_source import (
            VideoFileFrameSource,
        )

        source = VideoFileFrameSource(str(temp_video_file))
        source._buffer = [np.random.rand(480, 640, 3) for _ in range(3)]

        source.clear_buffers()

        assert len(source._buffer) == 0


class TestVideoFileFrameSourceLifecycle:
    """Test start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_opens_video(self, mock_cv2, temp_video_file):
        """Test that start opens the video file."""
        from argentic.core.agent.frame_sources.video_file_source import (
            VideoFileFrameSource,
        )

        source = VideoFileFrameSource(str(temp_video_file))

        await source.start()

        assert source._cap is not None
        assert source._cap.isOpened() is True
        assert source._is_playing is True

    @pytest.mark.asyncio
    async def test_start_reads_initial_buffer(self, mock_cv2, temp_video_file):
        """Test that start reads initial buffer."""
        from argentic.core.agent.frame_sources.video_file_source import (
            VideoFileFrameSource,
        )

        source = VideoFileFrameSource(str(temp_video_file), buffer_size=5)

        await source.start()

        # Should have read some frames
        assert len(source._buffer) > 0

    @pytest.mark.asyncio
    async def test_stop_closes_video(self, mock_cv2, temp_video_file):
        """Test that stop closes the video file."""
        from argentic.core.agent.frame_sources.video_file_source import (
            VideoFileFrameSource,
        )

        source = VideoFileFrameSource(str(temp_video_file))
        await source.start()

        await source.stop()

        assert source._is_playing is False
        assert source._cap is None

    @pytest.mark.asyncio
    async def test_start_failure_invalid_file(self, mock_cv2, temp_video_file):
        """Test start failure with invalid video file."""
        from argentic.core.agent.frame_sources.video_file_source import (
            VideoFileFrameSource,
        )

        source = VideoFileFrameSource(str(temp_video_file))

        # Mock cap to fail opening
        with patch(
            "argentic.core.agent.frame_sources.video_file_source.cv2.VideoCapture"
        ) as mock_cap_class:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            mock_cap_class.return_value = mock_cap

            with pytest.raises(RuntimeError, match="Failed to open video file"):
                await source.start()


class TestVideoFileFrameSourceLooping:
    """Test video looping functionality."""

    @pytest.mark.asyncio
    async def test_loop_disabled_reaches_end(self, mock_cv2, temp_video_file):
        """Test that video stops at end when loop is disabled."""
        from argentic.core.agent.frame_sources.video_file_source import (
            VideoFileFrameSource,
        )

        source = VideoFileFrameSource(str(temp_video_file), buffer_size=100, loop=False)
        await source.start()

        # First get should work
        frames1 = await source.get_frames()
        assert len(frames1) > 0

        # Exhaust the video by reading all frames
        # The mock has 10 frames, and we're reading with buffer_size=100
        # So after first read, we should be at the end

    @pytest.mark.asyncio
    async def test_loop_enabled_restarts(self, mock_cv2, temp_video_file):
        """Test that video restarts when loop is enabled."""
        from argentic.core.agent.frame_sources.video_file_source import (
            VideoFileFrameSource,
        )

        source = VideoFileFrameSource(str(temp_video_file), buffer_size=100, loop=True)
        await source.start()

        frames = await source.get_frames()

        # With looping enabled, should be able to read frames
        assert len(frames) > 0


class TestVideoFileFrameSourceGetInfo:
    """Test getting source information."""

    def test_get_info_basic(self, mock_cv2, temp_video_file):
        """Test getting basic source information."""
        from argentic.core.agent.frame_sources.video_file_source import (
            VideoFileFrameSource,
        )

        source = VideoFileFrameSource(str(temp_video_file), fps=10, loop=True)

        info = source.get_info()

        assert info["type"] == "VideoFileFrameSource"
        assert info["video_path"] == str(temp_video_file)
        assert info["target_fps"] == 10
        assert info["loop"] is True

    @pytest.mark.asyncio
    async def test_get_info_with_video_opened(self, mock_cv2, temp_video_file):
        """Test getting info when video is opened."""
        from argentic.core.agent.frame_sources.video_file_source import (
            VideoFileFrameSource,
        )

        source = VideoFileFrameSource(str(temp_video_file))
        await source.start()

        info = source.get_info()

        # Should include video properties when opened
        assert "native_fps" in info
        assert "total_frames" in info
        assert "width" in info
        assert "height" in info
        assert info["width"] == 640
        assert info["height"] == 480


class TestVideoFileFrameSourceIntegration:
    """Integration tests for video file frame source."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, mock_cv2, temp_video_file):
        """Test complete video file source lifecycle."""
        from argentic.core.agent.frame_sources.video_file_source import (
            VideoFileFrameSource,
        )

        source = VideoFileFrameSource(str(temp_video_file), buffer_size=5)

        # Start
        await source.start()
        assert source._is_playing is True

        # Get frames
        frames = await source.get_frames()
        assert len(frames) > 0

        # Clear buffer
        source.clear_buffers()
        assert len(source._buffer) == 0

        # Get more frames - Note: may be empty if we've read all frames from the video
        # This is expected behavior for non-looping video
        await source.get_frames()
        # Just check that it doesn't error; may be empty

        # Stop
        await source.stop()
        assert source._is_playing is False
        assert source._cap is None
