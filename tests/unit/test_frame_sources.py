"""Unit tests for frame sources."""

from unittest.mock import AsyncMock

import numpy as np
import pytest

from argentic.core.agent.frame_source import (
    FrameSource,
    FrameSourceConfig,
    FunctionalFrameSource,
    StaticFrameSource,
)


class TestFrameSourceConfig:
    """Test FrameSourceConfig dataclass."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = FrameSourceConfig()

        assert config.buffer_size == 30

    def test_config_custom_values(self):
        """Test custom configuration values."""
        config = FrameSourceConfig(buffer_size=60)

        assert config.buffer_size == 60


class TestStaticFrameSource:
    """Test StaticFrameSource implementation."""

    def test_init_with_frames(self):
        """Test initialization with frames."""
        frames = [np.random.rand(480, 640, 3) for _ in range(5)]
        source = StaticFrameSource(frames)

        assert source._frames == frames
        assert len(source._frames) == 5

    def test_init_empty_frames(self):
        """Test initialization with empty frames list."""
        source = StaticFrameSource([])

        assert source._frames == []

    @pytest.mark.asyncio
    async def test_get_frames(self):
        """Test getting frames from static source."""
        frames = [np.random.rand(480, 640, 3) for _ in range(3)]
        source = StaticFrameSource(frames)

        result = await source.get_frames()

        assert len(result) == 3
        assert all(np.array_equal(result[i], frames[i]) for i in range(3))

    @pytest.mark.asyncio
    async def test_get_audio_returns_none(self):
        """Test that static source returns None for audio."""
        frames = [np.random.rand(480, 640, 3)]
        source = StaticFrameSource(frames)

        audio = await source.get_audio()

        assert audio is None

    @pytest.mark.asyncio
    async def test_start_stop_no_op(self):
        """Test that start/stop are no-ops for static source."""
        frames = [np.random.rand(480, 640, 3)]
        source = StaticFrameSource(frames)

        # Should not raise any errors
        await source.start()
        await source.stop()

    def test_get_info(self):
        """Test getting source information."""
        frames = [np.random.rand(480, 640, 3) for _ in range(5)]
        source = StaticFrameSource(frames)

        info = source.get_info()

        assert info["type"] == "StaticFrameSource"
        assert info["config"]["buffer_size"] == 30


class TestFunctionalFrameSource:
    """Test FunctionalFrameSource implementation."""

    @pytest.mark.asyncio
    async def test_init_with_frames_function(self):
        """Test initialization with frames function."""

        async def get_frames():
            return [np.random.rand(480, 640, 3) for _ in range(3)]

        source = FunctionalFrameSource(get_frames_fn=get_frames)

        assert source.get_frames_fn == get_frames
        assert source.get_audio_fn is None

    @pytest.mark.asyncio
    async def test_init_with_frames_and_audio_functions(self):
        """Test initialization with both frames and audio functions."""

        async def get_frames():
            return [np.random.rand(480, 640, 3)]

        async def get_audio():
            return np.random.randn(1000)

        source = FunctionalFrameSource(get_frames_fn=get_frames, get_audio_fn=get_audio)

        assert source.get_frames_fn == get_frames
        assert source.get_audio_fn == get_audio

    @pytest.mark.asyncio
    async def test_get_frames_calls_function(self):
        """Test that get_frames calls the provided function."""
        called = False
        expected_frames = [np.random.rand(480, 640, 3) for _ in range(2)]

        async def get_frames():
            nonlocal called
            called = True
            return expected_frames

        source = FunctionalFrameSource(get_frames_fn=get_frames)

        result = await source.get_frames()

        assert called is True
        assert len(result) == 2
        assert all(np.array_equal(result[i], expected_frames[i]) for i in range(2))

    @pytest.mark.asyncio
    async def test_get_audio_calls_function(self):
        """Test that get_audio calls the provided function."""
        called = False
        expected_audio = np.random.randn(1000)

        async def get_audio():
            nonlocal called
            called = True
            return expected_audio

        source = FunctionalFrameSource(
            get_frames_fn=AsyncMock(return_value=[]), get_audio_fn=get_audio
        )

        result = await source.get_audio()

        assert called is True
        assert np.array_equal(result, expected_audio)

    @pytest.mark.asyncio
    async def test_get_audio_returns_none_when_no_function(self):
        """Test that get_audio returns None when no audio function provided."""

        async def get_frames():
            return [np.random.rand(480, 640, 3)]

        source = FunctionalFrameSource(get_frames_fn=get_frames)

        audio = await source.get_audio()

        assert audio is None

    @pytest.mark.asyncio
    async def test_get_frames_with_dynamic_generation(self):
        """Test dynamic frame generation across multiple calls."""
        counter = {"value": 0}

        async def get_frames():
            counter["value"] += 1
            return [np.ones((10, 10, 3)) * counter["value"]]

        source = FunctionalFrameSource(get_frames_fn=get_frames)

        frames1 = await source.get_frames()
        frames2 = await source.get_frames()
        frames3 = await source.get_frames()

        # Should generate different frames each time
        assert counter["value"] == 3
        assert frames1[0][0, 0, 0] == 1
        assert frames2[0][0, 0, 0] == 2
        assert frames3[0][0, 0, 0] == 3

    @pytest.mark.asyncio
    async def test_start_stop_no_op(self):
        """Test that start/stop are no-ops for functional source."""

        async def get_frames():
            return []

        source = FunctionalFrameSource(get_frames_fn=get_frames)

        # Should not raise any errors
        await source.start()
        await source.stop()

    def test_get_info(self):
        """Test getting source information."""

        async def get_frames():
            return []

        source = FunctionalFrameSource(get_frames_fn=get_frames)

        info = source.get_info()

        assert info["type"] == "FunctionalFrameSource"
        assert "config" in info

    def test_get_info_with_audio(self):
        """Test getting source information with audio function."""

        async def get_frames():
            return []

        async def get_audio():
            return None

        source = FunctionalFrameSource(get_frames_fn=get_frames, get_audio_fn=get_audio)

        info = source.get_info()

        assert info["type"] == "FunctionalFrameSource"
        assert "config" in info


class TestFrameSourceBase:
    """Test FrameSource base class behavior."""

    @pytest.mark.asyncio
    async def test_abstract_methods_must_be_implemented(self):
        """Test that abstract methods must be implemented by subclasses."""
        # FrameSource is abstract, so we can't instantiate it directly
        with pytest.raises(TypeError):
            FrameSource(FrameSourceConfig())  # type: ignore

    def test_custom_frame_source_implementation(self):
        """Test creating a custom frame source implementation."""

        class CustomFrameSource(FrameSource):
            """Custom test frame source."""

            async def get_frames(self):
                return [np.random.rand(100, 100, 3)]

            async def get_audio(self):
                return None

            async def start(self):
                pass

            async def stop(self):
                pass

        # Should be able to instantiate custom implementation
        source = CustomFrameSource(FrameSourceConfig())
        assert isinstance(source, FrameSource)

    @pytest.mark.asyncio
    async def test_custom_frame_source_get_info(self):
        """Test that custom frame source inherits get_info."""

        class CustomFrameSource(FrameSource):
            async def get_frames(self):
                return []

            async def get_audio(self):
                return None

            async def start(self):
                pass

            async def stop(self):
                pass

        source = CustomFrameSource(FrameSourceConfig(buffer_size=50))
        info = source.get_info()

        assert info["type"] == "CustomFrameSource"
        assert info["config"]["buffer_size"] == 50
