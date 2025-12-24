"""
Video File Frame Source

Provides frames from pre-recorded video files.
"""

from pathlib import Path
from typing import List, Optional

try:
    import cv2

    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    cv2 = None

try:
    import numpy as np

    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False
    np = None

from argentic.core.agent.frame_source import FrameSource, FrameSourceConfig


class VideoFileFrameSource(FrameSource):
    """
    Frame source that reads from a video file.

    Supports any video format supported by OpenCV (mp4, avi, mkv, etc.).

    Example:
        ```python
        from argentic.core.agent.frame_sources.video_file_source import VideoFileFrameSource
        from argentic.core.agent.visual_agent import VisualAgent

        # Create video file source
        source = VideoFileFrameSource(
            video_path="/path/to/video.mp4",
            buffer_size=30,
            fps=10,  # Subsample to 10 FPS
        )

        # Use with VisualAgent
        agent = VisualAgent(
            llm=llm,
            messager=messager,
            frame_source=source,
        )

        async with agent:
            response = await agent.query_with_video("What's happening in this video?")
        ```
    """

    def __init__(
        self,
        video_path: str,
        buffer_size: int = 30,
        fps: Optional[int] = None,
        loop: bool = False,
        config: Optional[FrameSourceConfig] = None,
    ):
        """
        Initialize video file frame source.

        Args:
            video_path: Path to video file
            buffer_size: Maximum number of frames to buffer
            fps: Target FPS (None = use video's native FPS)
            loop: Whether to loop video when it ends
            config: Optional frame source configuration
        """
        if not _CV2_AVAILABLE:
            raise ImportError(
                "opencv-python is required for VideoFileFrameSource. "
                "Install it with: pip install opencv-python"
            )

        if config is None:
            config = FrameSourceConfig(buffer_size=buffer_size)

        super().__init__(config)

        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.target_fps = fps
        self.loop = loop

        self._cap: Optional[cv2.VideoCapture] = None
        self._buffer: List[np.ndarray] = []
        self._is_playing = False
        self._current_frame_index = 0

    async def get_frames(self) -> List["np.ndarray"]:
        """
        Get buffered frames from video file.

        Returns:
            List of numpy arrays representing video frames
        """
        if not self._is_playing:
            await self._read_frames()

        return self._buffer.copy()

    async def get_audio(self) -> Optional["np.ndarray"]:
        """
        Get audio (not supported for video files via OpenCV).

        Returns:
            None (audio not supported)
        """
        # OpenCV doesn't easily support audio extraction
        # For audio, would need to use moviepy or similar
        return None

    async def _read_frames(self):
        """Read frames from video file into buffer."""
        if self._cap is None or not self._cap.isOpened():
            return

        self._buffer.clear()

        # Calculate frame skip for FPS subsampling
        native_fps = self._cap.get(cv2.CAP_PROP_FPS)
        frame_skip = 1
        if self.target_fps and native_fps > 0:
            frame_skip = max(1, int(native_fps / self.target_fps))

        frames_read = 0
        while frames_read < self.config.buffer_size:
            ret, frame = self._cap.read()

            if not ret:
                # End of video
                if self.loop:
                    # Restart from beginning
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._current_frame_index = 0
                    continue
                else:
                    break

            # Subsample frames
            if self._current_frame_index % frame_skip == 0:
                # Convert BGR to RGB (OpenCV uses BGR)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self._buffer.append(frame_rgb)
                frames_read += 1

            self._current_frame_index += 1

    def clear_buffers(self):
        """Clear the frame buffer."""
        self._buffer.clear()

    async def start(self):
        """Open the video file."""
        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {self.video_path}")

        self._is_playing = True
        self._current_frame_index = 0

        # Read initial buffer
        await self._read_frames()

    async def stop(self):
        """Close the video file."""
        self._is_playing = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def get_info(self) -> dict:
        """Get information about video file source."""
        info = super().get_info()
        info.update(
            {
                "video_path": str(self.video_path),
                "target_fps": self.target_fps,
                "loop": self.loop,
            }
        )

        if self._cap is not None and self._cap.isOpened():
            info["native_fps"] = self._cap.get(cv2.CAP_PROP_FPS)
            info["total_frames"] = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            info["width"] = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            info["height"] = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        return info
