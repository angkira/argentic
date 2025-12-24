"""
Frame Source Interface for Visual Agent

Defines the interface for providing video frames and audio to the VisualAgent.
Supports multiple sources: WebRTC, file uploads, cameras, video files, etc.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional

try:
    import numpy as np

    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False
    np = None


@dataclass
class FrameSourceConfig:
    """Configuration for frame source."""

    buffer_size: int = 30
    """Maximum number of frames to buffer"""

    enable_audio: bool = False
    """Whether audio should be captured"""

    frame_rate: Optional[int] = None
    """Target frame rate (optional)"""

    auto_clear_buffer: bool = False
    """Whether to automatically clear buffer after retrieval"""


class FrameSource(ABC):
    """
    Abstract base class for frame sources.

    A frame source provides video frames (and optionally audio) to the VisualAgent.
    This abstraction allows VisualAgent to work with any source:
    - WebRTC streams
    - Video files
    - Camera devices
    - Screen capture
    - Image sequences
    - Network streams
    - etc.

    Example implementations:
    - WebRTCFrameSource: Real-time WebRTC video
    - VideoFileFrameSource: Pre-recorded video files
    - CameraFrameSource: Local camera device
    - ScreenCaptureFrameSource: Screen recording
    """

    def __init__(self, config: Optional[FrameSourceConfig] = None):
        """
        Initialize frame source.

        Args:
            config: Optional configuration
        """
        if not _NUMPY_AVAILABLE:
            raise ImportError(
                "numpy is required for FrameSource. Install it with: pip install numpy"
            )

        self.config = config or FrameSourceConfig()

    @abstractmethod
    async def get_frames(self) -> List["np.ndarray"]:
        """
        Get buffered video frames.

        Returns:
            List of numpy arrays (H, W, C) representing video frames

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("get_frames must be implemented by subclass")

    async def get_audio(self) -> Optional["np.ndarray"]:
        """
        Get buffered audio (optional).

        Returns:
            Numpy array of audio samples, or None if audio not supported

        Default implementation returns None.
        """
        return None

    async def get_latest_frame(self) -> Optional["np.ndarray"]:
        """
        Get the most recent frame only.

        Returns:
            Single frame as numpy array, or None if no frames available

        Default implementation gets all frames and returns last one.
        """
        frames = await self.get_frames()
        return frames[-1] if frames else None

    def clear_buffers(self):
        """
        Clear internal buffers (optional).

        Override this if your source maintains buffers that should be cleared.
        """
        pass

    async def start(self):
        """
        Start the frame source (optional).

        Override this if your source needs explicit start/stop lifecycle.
        """
        pass

    async def stop(self):
        """
        Stop the frame source (optional).

        Override this if your source needs explicit start/stop lifecycle.
        """
        pass

    def get_info(self) -> dict:
        """
        Get information about this frame source.

        Returns:
            Dict with source information (type, config, status, etc.)
        """
        return {
            "type": self.__class__.__name__,
            "config": {
                "buffer_size": self.config.buffer_size,
                "enable_audio": self.config.enable_audio,
                "frame_rate": self.config.frame_rate,
            },
        }

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


class FunctionalFrameSource(FrameSource):
    """
    Frame source that wraps callback functions.

    This allows creating frame sources from simple async functions
    without needing to subclass FrameSource.

    Example:
        ```python
        async def get_my_frames():
            return [frame1, frame2, frame3]

        async def get_my_audio():
            return audio_samples

        source = FunctionalFrameSource(
            get_frames_fn=get_my_frames,
            get_audio_fn=get_my_audio,
        )
        ```
    """

    def __init__(
        self,
        get_frames_fn: Callable[[], List["np.ndarray"]],
        get_audio_fn: Optional[Callable[[], Optional["np.ndarray"]]] = None,
        clear_buffers_fn: Optional[Callable[[], None]] = None,
        start_fn: Optional[Callable[[], None]] = None,
        stop_fn: Optional[Callable[[], None]] = None,
        config: Optional[FrameSourceConfig] = None,
    ):
        """
        Initialize functional frame source.

        Args:
            get_frames_fn: Async function that returns list of frames
            get_audio_fn: Optional async function that returns audio
            clear_buffers_fn: Optional function to clear buffers
            start_fn: Optional async function to start source
            stop_fn: Optional async function to stop source
            config: Optional configuration
        """
        super().__init__(config)

        self.get_frames_fn = get_frames_fn
        self.get_audio_fn = get_audio_fn
        self.clear_buffers_fn = clear_buffers_fn
        self.start_fn = start_fn
        self.stop_fn = stop_fn

    async def get_frames(self) -> List["np.ndarray"]:
        """Get frames using provided function."""
        return await self.get_frames_fn()

    async def get_audio(self) -> Optional["np.ndarray"]:
        """Get audio using provided function."""
        if self.get_audio_fn:
            return await self.get_audio_fn()
        return None

    def clear_buffers(self):
        """Clear buffers using provided function."""
        if self.clear_buffers_fn:
            self.clear_buffers_fn()

    async def start(self):
        """Start source using provided function."""
        if self.start_fn:
            await self.start_fn()

    async def stop(self):
        """Stop source using provided function."""
        if self.stop_fn:
            await self.stop_fn()


class StaticFrameSource(FrameSource):
    """
    Frame source that returns a static set of frames.

    Useful for testing or processing pre-loaded images.

    Example:
        ```python
        frames = [frame1, frame2, frame3]
        source = StaticFrameSource(frames)

        retrieved = await source.get_frames()
        # retrieved == frames
        ```
    """

    def __init__(
        self,
        frames: List["np.ndarray"],
        audio: Optional["np.ndarray"] = None,
        config: Optional[FrameSourceConfig] = None,
    ):
        """
        Initialize static frame source.

        Args:
            frames: List of frames to return
            audio: Optional audio samples
            config: Optional configuration
        """
        super().__init__(config)
        self._frames = frames
        self._audio = audio

    async def get_frames(self) -> List["np.ndarray"]:
        """Return the static frames."""
        return self._frames

    async def get_audio(self) -> Optional["np.ndarray"]:
        """Return the static audio."""
        return self._audio

    def clear_buffers(self):
        """Clear the static frames."""
        self._frames = []
        self._audio = None
