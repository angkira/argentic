"""
Frame Sources for Visual Agent

Provides various frame source implementations for different video/image inputs.
"""

from argentic.core.agent.frame_sources.video_file_source import VideoFileFrameSource
from argentic.core.agent.frame_sources.webrtc_source import WebRTCFrameSource

__all__ = ["WebRTCFrameSource", "VideoFileFrameSource"]
