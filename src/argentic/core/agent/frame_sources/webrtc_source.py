"""
WebRTC Frame Source Adapter

Adapts the WebRTC driver to work with the FrameSource interface.
"""

from typing import List, Optional

try:
    import numpy as np

    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False
    np = None

from argentic.core.agent.frame_source import FrameSource, FrameSourceConfig
from argentic.core.drivers.webrtc_driver import WebRTCDriver


class WebRTCFrameSource(FrameSource):
    """
    Frame source adapter for WebRTC driver.

    Wraps WebRTCDriver to provide the FrameSource interface,
    allowing WebRTC streams to be used with VisualAgent.

    Example:
        ```python
        from argentic.core.drivers.webrtc_driver import WebRTCDriver
        from argentic.core.agent.frame_sources.webrtc_source import WebRTCFrameSource
        from argentic.core.agent.visual_agent import VisualAgent

        # Create WebRTC driver
        driver = WebRTCDriver(
            video_buffer_size=30,
            frame_rate=10,
            resize_frames=(640, 480),
            enable_audio=True,
        )

        # Wrap in frame source
        frame_source = WebRTCFrameSource(driver)

        # Use with VisualAgent
        agent = VisualAgent(
            llm=llm,
            messager=messager,
            frame_source=frame_source,
        )
        ```
    """

    def __init__(
        self,
        driver: WebRTCDriver,
        config: Optional[FrameSourceConfig] = None,
    ):
        """
        Initialize WebRTC frame source.

        Args:
            driver: WebRTCDriver instance
            config: Optional frame source configuration (not used, driver has its own config)
        """
        super().__init__(config)
        self.driver = driver

    async def get_frames(self) -> List["np.ndarray"]:
        """
        Get buffered video frames from WebRTC driver.

        Returns:
            List of numpy arrays representing video frames
        """
        return await self.driver.get_frame_buffer()

    async def get_audio(self) -> Optional["np.ndarray"]:
        """
        Get buffered audio from WebRTC driver.

        Returns:
            Numpy array of audio samples, or None if audio disabled
        """
        return await self.driver.get_audio_buffer()

    async def get_latest_frame(self) -> Optional["np.ndarray"]:
        """
        Get the most recent frame from WebRTC driver.

        Returns:
            Single frame as numpy array, or None if no frames
        """
        return await self.driver.get_latest_frame()

    def clear_buffers(self):
        """Clear WebRTC driver's internal buffers."""
        self.driver.clear_buffers()

    async def start(self):
        """Start WebRTC connection and capture."""
        await self.driver.connect()
        await self.driver.start_capture()

    async def stop(self):
        """Stop WebRTC capture and disconnect."""
        await self.driver.stop_capture()
        await self.driver.disconnect()

    def get_info(self) -> dict:
        """Get information about WebRTC frame source."""
        return {
            "type": "WebRTCFrameSource",
            "config": {
                "buffer_size": self.driver.video_buffer_size,
                "enable_audio": self.driver.enable_audio,
                "frame_rate": self.driver.frame_rate,
                "resize_frames": self.driver.resize_frames,
            },
            "driver": self.driver.__class__.__name__,
        }
