import asyncio
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Tuple

try:
    import numpy as np

    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False
    np = None

try:
    from aiortc import RTCPeerConnection, RTCSessionDescription
    from aiortc.contrib.media import MediaRelay
    from av import VideoFrame, AudioFrame

    _AIORTC_AVAILABLE = True
except ImportError:
    _AIORTC_AVAILABLE = False
    RTCPeerConnection = None
    RTCSessionDescription = None
    MediaRelay = None
    VideoFrame = None
    AudioFrame = None

try:
    from PIL import Image

    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False
    Image = None

from argentic.core.logger import LogLevel, get_logger


class WebRTCDriver:
    """
    WebRTC driver for capturing video and audio streams with low latency.

    Uses aiortc for WebRTC connection, maintains frame and audio buffers,
    and provides non-blocking access to buffered media.

    Threading strategy:
    - Main asyncio loop: WebRTC signaling and connection management
    - Frame processing pool: CPU-intensive frame operations (resize, format conversion)
    - Thread-safe queues for communication between async and sync contexts
    """

    def __init__(
        self,
        # Connection parameters
        signaling_url: Optional[str] = None,
        ice_servers: Optional[List[Dict[str, str]]] = None,
        # Frame buffering
        video_buffer_size: int = 30,
        frame_rate: int = 10,  # Target FPS for capture
        # Audio buffering
        audio_buffer_duration: float = 5.0,  # Seconds
        audio_sample_rate: int = 16000,
        enable_audio: bool = True,
        # Processing options
        resize_frames: Optional[Tuple[int, int]] = None,  # (width, height)
        frame_format: str = "rgb24",
        # Threading
        frame_processor_workers: int = 4,
        log_level: str = "INFO",
    ):
        if not _AIORTC_AVAILABLE:
            raise ImportError("aiortc is not installed. Install it with: pip install aiortc av")
        if not _NUMPY_AVAILABLE:
            raise ImportError(
                "numpy is required for WebRTCDriver. Install it with: pip install numpy"
            )

        self.logger = get_logger("webrtc_driver", LogLevel[log_level.upper()])

        # Connection parameters
        self.signaling_url = signaling_url
        self.ice_servers = ice_servers or [{"urls": "stun:stun.l.google.com:19302"}]

        # Buffer configuration
        self.video_buffer_size = video_buffer_size
        self.frame_rate = frame_rate
        self.audio_buffer_duration = audio_buffer_duration
        self.audio_sample_rate = audio_sample_rate
        self.enable_audio = enable_audio
        self.resize_frames = resize_frames
        self.frame_format = frame_format

        # WebRTC components
        self._pc: Optional[RTCPeerConnection] = None
        self._relay = MediaRelay()
        self._video_track = None
        self._audio_track = None

        # Buffers (thread-safe via asyncio.Queue and deque)
        self._frame_buffer: deque = deque(maxlen=video_buffer_size)
        self._audio_buffer: deque = deque(maxlen=int(audio_sample_rate * audio_buffer_duration))
        self._frame_timestamps: deque = deque(maxlen=video_buffer_size)
        self._audio_timestamps: deque = deque(maxlen=int(audio_sample_rate * audio_buffer_duration))

        # Thread pool for frame processing (CPU-intensive operations)
        self._frame_processor_pool = ThreadPoolExecutor(
            max_workers=frame_processor_workers, thread_name_prefix="frame-processor"
        )

        # Capture tasks
        self._video_capture_task: Optional[asyncio.Task] = None
        self._audio_capture_task: Optional[asyncio.Task] = None
        self._connected = False
        self._capturing = False

        # Frame callback
        self._frame_callback: Optional[Callable] = None

        self.logger.info(
            f"WebRTCDriver initialized: buffer_size={video_buffer_size}, "
            f"frame_rate={frame_rate}, audio={enable_audio}"
        )

    async def connect(self, offer_sdp: Optional[str] = None) -> Optional[str]:
        """
        Establish WebRTC connection.

        Args:
            offer_sdp: Optional SDP offer string for connection setup

        Returns:
            Answer SDP string if offer was provided, None otherwise
        """
        if self._connected:
            self.logger.warning("Already connected")
            return None

        self.logger.info("Establishing WebRTC connection...")

        # Create peer connection
        self._pc = RTCPeerConnection(configuration={"iceServers": self.ice_servers})

        # Set up track handlers
        @self._pc.on("track")
        async def on_track(track):
            self.logger.info(f"Track received: {track.kind}")

            if track.kind == "video":
                self._video_track = track
            elif track.kind == "audio" and self.enable_audio:
                self._audio_track = track

        @self._pc.on("connectionstatechange")
        async def on_connectionstatechange():
            self.logger.info(f"Connection state: {self._pc.connectionState}")
            if self._pc.connectionState == "failed":
                await self.disconnect()

        # Handle offer/answer exchange if offer provided
        answer_sdp = None
        if offer_sdp:
            await self._pc.setRemoteDescription(RTCSessionDescription(sdp=offer_sdp, type="offer"))
            answer = await self._pc.createAnswer()
            await self._pc.setLocalDescription(answer)
            answer_sdp = self._pc.localDescription.sdp
            self.logger.info("Answer SDP created")

        self._connected = True
        self.logger.info("WebRTC connection established")

        return answer_sdp

    async def disconnect(self):
        """Close WebRTC connection gracefully."""
        self.logger.info("Disconnecting WebRTC...")

        # Stop capture first
        await self.stop_capture()

        # Close peer connection
        if self._pc:
            await self._pc.close()
            self._pc = None

        self._connected = False
        self.logger.info("WebRTC disconnected")

    async def start_capture(self):
        """Begin capturing video and audio frames."""
        if not self._connected:
            self.logger.error("Cannot start capture: not connected")
            return

        if self._capturing:
            self.logger.warning("Already capturing")
            return

        self.logger.info("Starting media capture...")

        # Start video capture task
        if self._video_track:
            self._video_capture_task = asyncio.create_task(self._capture_video_loop())

        # Start audio capture task
        if self._audio_track and self.enable_audio:
            self._audio_capture_task = asyncio.create_task(self._capture_audio_loop())

        self._capturing = True
        self.logger.info("Media capture started")

    async def stop_capture(self):
        """Stop capturing video and audio frames."""
        if not self._capturing:
            return

        self.logger.info("Stopping media capture...")

        # Cancel capture tasks
        if self._video_capture_task:
            self._video_capture_task.cancel()
            try:
                await self._video_capture_task
            except asyncio.CancelledError:
                pass
            self._video_capture_task = None

        if self._audio_capture_task:
            self._audio_capture_task.cancel()
            try:
                await self._audio_capture_task
            except asyncio.CancelledError:
                pass
            self._audio_capture_task = None

        self._capturing = False
        self.logger.info("Media capture stopped")

    async def _capture_video_loop(self):
        """Continuously capture video frames (runs in asyncio task)."""
        self.logger.info("Video capture loop started")
        frame_interval = 1.0 / self.frame_rate
        last_capture_time = 0.0

        try:
            while True:
                current_time = time.time()

                # Throttle frame rate
                if current_time - last_capture_time < frame_interval:
                    await asyncio.sleep(frame_interval - (current_time - last_capture_time))
                    continue

                # Receive frame from track
                try:
                    frame = await self._video_track.recv()
                    last_capture_time = current_time

                    # Process frame in thread pool to avoid blocking
                    processed_frame = await asyncio.get_event_loop().run_in_executor(
                        self._frame_processor_pool, self._process_frame, frame
                    )

                    if processed_frame is not None:
                        # Add to buffer (deque automatically removes oldest if full)
                        self._frame_buffer.append(processed_frame)
                        self._frame_timestamps.append(current_time)

                        # Trigger callback if registered
                        if self._frame_callback:
                            await self._frame_callback(processed_frame, current_time)

                except Exception as e:
                    self.logger.error(f"Error receiving video frame: {e}")
                    await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            self.logger.info("Video capture loop cancelled")
        except Exception as e:
            self.logger.error(f"Video capture loop error: {e}", exc_info=True)

    def _process_frame(self, frame: "VideoFrame") -> Optional["np.ndarray"]:
        """
        Process video frame (runs in thread pool).

        Args:
            frame: VideoFrame from aiortc

        Returns:
            numpy array of frame data or None on error
        """
        try:
            # Convert to numpy array
            img = frame.to_ndarray(format=self.frame_format)

            # Resize if requested
            if self.resize_frames and _PIL_AVAILABLE:
                pil_img = Image.fromarray(img)
                pil_img = pil_img.resize(self.resize_frames, Image.LANCZOS)
                img = np.array(pil_img)

            return img
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
            return None

    async def _capture_audio_loop(self):
        """Continuously capture audio samples (runs in asyncio task)."""
        self.logger.info("Audio capture loop started")

        try:
            while True:
                try:
                    # Receive audio frame
                    frame = await self._audio_track.recv()
                    current_time = time.time()

                    # Process audio in thread pool
                    processed_audio = await asyncio.get_event_loop().run_in_executor(
                        self._frame_processor_pool, self._process_audio, frame
                    )

                    if processed_audio is not None:
                        self._audio_buffer.extend(processed_audio)
                        # Store timestamp for each sample
                        for _ in range(len(processed_audio)):
                            self._audio_timestamps.append(current_time)

                except Exception as e:
                    self.logger.error(f"Error receiving audio frame: {e}")
                    await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            self.logger.info("Audio capture loop cancelled")
        except Exception as e:
            self.logger.error(f"Audio capture loop error: {e}", exc_info=True)

    def _process_audio(self, frame: "AudioFrame") -> Optional["np.ndarray"]:
        """
        Process audio frame (runs in thread pool).

        Args:
            frame: AudioFrame from aiortc

        Returns:
            numpy array of audio samples or None on error
        """
        try:
            # Convert to numpy array
            audio_data = frame.to_ndarray()

            # Resample if needed (simplified - should use proper resampling)
            if frame.sample_rate != self.audio_sample_rate:
                self.logger.warning(
                    f"Audio sample rate mismatch: {frame.sample_rate} vs {self.audio_sample_rate}"
                )

            # Convert to mono if stereo
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)

            return audio_data.flatten()
        except Exception as e:
            self.logger.error(f"Error processing audio: {e}")
            return None

    async def get_frame_buffer(self) -> List["np.ndarray"]:
        """
        Get current video frame buffer (non-blocking).

        Returns:
            List of numpy arrays representing frames
        """
        return list(self._frame_buffer)

    async def get_audio_buffer(self) -> Optional["np.ndarray"]:
        """
        Get current audio buffer (non-blocking).

        Returns:
            numpy array of audio samples or None if audio disabled
        """
        if not self.enable_audio or not self._audio_buffer:
            return None

        return np.array(list(self._audio_buffer))

    async def get_latest_frame(self) -> Optional["np.ndarray"]:
        """Get the most recent frame from buffer."""
        if not self._frame_buffer:
            return None
        return self._frame_buffer[-1]

    def set_frame_callback(self, callback: Callable):
        """
        Register a callback function for new frames.

        Args:
            callback: Async function called with (frame, timestamp) for each new frame
        """
        self._frame_callback = callback

    def clear_buffers(self):
        """Clear all buffered frames and audio."""
        self._frame_buffer.clear()
        self._audio_buffer.clear()
        self._frame_timestamps.clear()
        self._audio_timestamps.clear()
        self.logger.debug("Buffers cleared")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        await self.start_capture()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop_capture()
        await self.disconnect()
        # Cleanup thread pool
        self._frame_processor_pool.shutdown(wait=True)

    def __del__(self):
        """Cleanup on deletion."""
        try:
            # Shutdown thread pool if not already done
            if hasattr(self, "_frame_processor_pool"):
                self._frame_processor_pool.shutdown(wait=False)
        except Exception:
            pass
