import asyncio
import time
from typing import Any, Dict, List, Optional

try:
    import numpy as np

    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False
    np = None

from argentic.core.agent.agent import Agent
from argentic.core.drivers.webrtc_driver import WebRTCDriver
from argentic.core.llm.providers.base import ModelProvider
from argentic.core.messager.messager import Messager
from argentic.core.protocol.enums import MessageSource
from argentic.core.protocol.message import AnswerMessage


class VisualAgent(Agent):
    """
    Visual AI agent with video and audio processing capabilities.

    Extends the base Agent to support real-time video/audio input via WebRTC driver.
    Automatically processes buffered frames at intervals and publishes responses via MQTT.

    Threading model:
    - WebRTC driver handles frame capture in background threads
    - Auto-processing loop runs as asyncio task
    - Model inference happens in provider's thread pool
    - All communication via async/await patterns
    """

    def __init__(
        self,
        llm: ModelProvider,
        messager: Messager,
        webrtc_driver: WebRTCDriver,  # Driver instance passed in
        # Agent-specific params
        auto_process_interval: float = 5.0,  # Process buffer every N seconds
        visual_prompt_template: str = "Describe what you see in the video: {question}",
        # Response publishing
        visual_response_topic: str = "agent/visual/response",
        # Processing control
        enable_auto_processing: bool = True,
        process_on_buffer_full: bool = True,
        min_frames_for_processing: int = 10,  # Minimum frames before processing
        **kwargs,  # Pass remaining to base Agent
    ):
        super().__init__(llm, messager, **kwargs)

        if not _NUMPY_AVAILABLE:
            raise ImportError(
                "numpy is required for VisualAgent. Install it with: pip install numpy"
            )

        self.driver = webrtc_driver
        self.auto_process_interval = auto_process_interval
        self.visual_prompt_template = visual_prompt_template
        self.visual_response_topic = visual_response_topic
        self.enable_auto_processing = enable_auto_processing
        self.process_on_buffer_full = process_on_buffer_full
        self.min_frames_for_processing = min_frames_for_processing

        # Processing state
        self._processing_task: Optional[asyncio.Task] = None
        self._last_process_time: float = 0.0
        self._processing_active = False

        self.logger.info(
            f"VisualAgent initialized: auto_interval={auto_process_interval}s, "
            f"min_frames={min_frames_for_processing}"
        )

    async def async_init(self):
        """Initialize agent and start video processing."""
        # Initialize base agent first
        await super().async_init()

        self.logger.info("Initializing VisualAgent...")

        # Connect WebRTC driver
        await self.driver.connect()
        await self.driver.start_capture()

        # Start auto-processing task if enabled
        if self.enable_auto_processing:
            self._processing_task = asyncio.create_task(self._auto_process_loop())
            self.logger.info("Auto-processing enabled")

        self.logger.info("VisualAgent initialized and capturing")

    async def _auto_process_loop(self):
        """
        Continuously process video buffers at intervals.

        Runs as asyncio task, non-blocking with respect to main event loop.
        """
        self.logger.info("Auto-processing loop started")
        self._processing_active = True

        try:
            while self._processing_active:
                try:
                    # Wait for interval
                    await asyncio.sleep(self.auto_process_interval)

                    current_time = time.time()
                    elapsed = current_time - self._last_process_time

                    # Skip if processed too recently
                    if elapsed < self.auto_process_interval:
                        continue

                    # Get buffered frames and audio
                    frames = await self.driver.get_frame_buffer()
                    audio = await self.driver.get_audio_buffer()

                    # Check if we have enough frames
                    if not frames or len(frames) < self.min_frames_for_processing:
                        self.logger.debug(
                            f"Skipping processing: only {len(frames) if frames else 0} frames "
                            f"(min {self.min_frames_for_processing})"
                        )
                        continue

                    self.logger.info(
                        f"Auto-processing {len(frames)} frames, "
                        f"audio={'yes' if audio is not None else 'no'}"
                    )

                    # Process with default prompt
                    result = await self._process_visual_input(
                        frames=frames, audio=audio, prompt="What is happening in the video?"
                    )

                    # Publish result via Messager
                    await self._publish_visual_response(result)

                    self._last_process_time = current_time

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Error in auto-process loop: {e}", exc_info=True)
                    # Continue loop despite errors
                    await asyncio.sleep(1.0)

        except asyncio.CancelledError:
            self.logger.info("Auto-process loop cancelled")
        finally:
            self._processing_active = False
            self.logger.info("Auto-process loop stopped")

    async def _process_visual_input(
        self, frames: List["np.ndarray"], audio: Optional["np.ndarray"], prompt: str
    ) -> str:
        """
        Send buffered media to model and get response.

        Args:
            frames: List of numpy arrays (video frames)
            audio: Optional numpy array of audio samples
            prompt: Text prompt to accompany visual input

        Returns:
            Generated text response from model
        """
        # Create multimodal message
        multimodal_content = {"text": prompt, "images": frames}

        if audio is not None:
            multimodal_content["audio"] = audio

        # Format for LLM
        messages = [
            {
                "role": "system",
                "content": self.system_prompt or "You are a visual AI assistant.",
            },
            {"role": "user", "content": multimodal_content},
        ]

        self.logger.debug(
            f"Sending to model: {len(frames)} frames, "
            f"audio={audio is not None}, prompt='{prompt[:50]}...'"
        )

        # Call LLM (handles multimodal via GemmaProvider)
        # This will use the provider's thread pool for inference
        response = await self._call_llm(messages, llm_config=self.llm_config)

        self.logger.info(f"Model response: {response.message.content[:100]}...")

        return response.message.content

    async def query_with_video(self, question: str) -> str:
        """
        Query with current video buffer.

        Args:
            question: Question/prompt to ask about the video

        Returns:
            Generated response from model
        """
        self.logger.info(f"Query with video: '{question}'")

        # Get current buffers
        frames = await self.driver.get_frame_buffer()
        audio = await self.driver.get_audio_buffer()

        if not frames:
            warning_msg = "No video frames available in buffer."
            self.logger.warning(warning_msg)
            return warning_msg

        if len(frames) < self.min_frames_for_processing:
            warning_msg = (
                f"Only {len(frames)} frames available "
                f"(minimum {self.min_frames_for_processing} recommended)"
            )
            self.logger.warning(warning_msg)

        # Process visual input
        result = await self._process_visual_input(frames, audio, question)

        # Publish response
        await self._publish_visual_response(result, question=question)

        return result

    async def query(
        self, question: str, user_id: Optional[str] = None, max_iterations: Optional[int] = None
    ) -> str:
        """
        Override base query to use visual context if available.

        Args:
            question: Question text
            user_id: Optional user identifier
            max_iterations: Max iterations for tool loop

        Returns:
            Generated response
        """
        # Get current video buffer
        frames = await self.driver.get_frame_buffer()
        audio = await self.driver.get_audio_buffer()

        # If we have visual data, use visual processing
        if frames and len(frames) >= self.min_frames_for_processing:
            self.logger.info("Using visual context for query")
            return await self._process_visual_input(frames, audio, question)
        else:
            # Fall back to base agent query (text-only)
            self.logger.info("No visual context, using base agent query")
            return await super().query(question, user_id, max_iterations)

    async def _publish_visual_response(self, response: str, question: str = ""):
        """
        Publish response via Messager.

        Args:
            response: Generated response text
            question: Original question (optional)
        """
        message = AnswerMessage(
            question=question,
            answer=response,
            source=MessageSource.AGENT,
            data=None,
        )

        await self.messager.publish(self.visual_response_topic, message)
        self.logger.debug(f"Published response to {self.visual_response_topic}")

    def pause_auto_processing(self):
        """Pause automatic video processing."""
        self._processing_active = False
        self.logger.info("Auto-processing paused")

    def resume_auto_processing(self):
        """Resume automatic video processing."""
        if self._processing_task and not self._processing_task.done():
            self._processing_active = True
            self.logger.info("Auto-processing resumed")
        else:
            self.logger.warning("Cannot resume: processing task not running")

    async def stop(self):
        """Stop agent and cleanup resources."""
        self.logger.info("Stopping VisualAgent...")

        # Stop auto-processing
        self._processing_active = False
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
            self._processing_task = None

        # Stop WebRTC driver
        await self.driver.stop_capture()
        await self.driver.disconnect()

        # Stop base agent
        await super().stop()

        self.logger.info("VisualAgent stopped")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.async_init()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
