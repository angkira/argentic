import asyncio
import time
from typing import Any, Callable, List, Optional

try:
    import numpy as np

    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False
    np = None

from argentic.core.agent.agent import Agent
from argentic.core.agent.frame_source import FrameSource
from argentic.core.llm.providers.base import ModelProvider
from argentic.core.messager.messager import Messager
from argentic.core.protocol.enums import MessageSource
from argentic.core.protocol.message import AnswerMessage


class VisualAgent(Agent):
    """
    Visual AI agent with video and audio processing capabilities.

    Extends the base Agent to support real-time video/audio input from any frame source.
    Automatically processes buffered frames at intervals and publishes responses via MQTT.

    Frame Sources:
    The agent works with any FrameSource implementation:
    - WebRTCFrameSource: Real-time WebRTC video
    - VideoFileFrameSource: Pre-recorded video files
    - CameraFrameSource: Local camera device
    - StaticFrameSource: Fixed set of images
    - FunctionalFrameSource: Custom callback-based source

    Visual Processing Modes:
    1. Direct frames mode (default): Passes raw video frames directly to the LLM provider
    2. Embeddings mode: Uses a custom embedding function to convert frames to embeddings
       before passing to the LLM provider

    The embedding function should be async and have the signature:
        async def embed_frames(frames: List[np.ndarray]) -> np.ndarray | Dict[str, Any]

    Embedding Support by Provider:
    - VLLMNativeProvider: Full support via multi_modal_data parameter
    - VLLMProvider: Not supported (OpenAI API limitation)
    - GoogleGeminiProvider: Not supported (uses internal vision encoder)
    - TransformersProvider: Model-dependent (some support pre-computed embeddings)

    Threading model:
    - Frame source handles capture in its own way (threads, async, etc.)
    - Auto-processing loop runs as asyncio task
    - Embedding function (if provided) processes frames asynchronously
    - Model inference happens in provider's thread pool
    - All communication via async/await patterns
    """

    def __init__(
        self,
        llm: ModelProvider,
        messager: Messager,
        frame_source: FrameSource,  # Generic frame source
        # Visual embedding function (optional)
        embedding_function: Optional[Callable[[List["np.ndarray"]], Any]] = None,
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
        """
        Initialize Visual Agent.

        Args:
            llm: Language model provider
            messager: MQTT messager instance
            frame_source: Frame source for video/audio (WebRTC, file, camera, etc.)
            embedding_function: Optional async function to convert frames to embeddings.
                                Should have signature: async def(frames: List[np.ndarray]) -> np.ndarray | Dict
                                If provided, embeddings will be passed to the LLM instead of raw frames.
            auto_process_interval: Interval in seconds between auto-processing runs
            visual_prompt_template: Template for visual prompts
            visual_response_topic: MQTT topic for publishing responses
            enable_auto_processing: Whether to enable automatic periodic processing
            process_on_buffer_full: Whether to process when buffer is full
            min_frames_for_processing: Minimum number of frames required for processing
            **kwargs: Additional arguments passed to base Agent
        """
        super().__init__(llm, messager, **kwargs)

        if not _NUMPY_AVAILABLE:
            raise ImportError(
                "numpy is required for VisualAgent. Install it with: pip install numpy"
            )

        self.frame_source = frame_source
        self.embedding_function = embedding_function
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

        mode_info = ", mode=embeddings" if self.embedding_function else ", mode=frames"
        source_info = self.frame_source.get_info()

        self.logger.info(
            f"VisualAgent initialized: source={source_info['type']}, "
            f"auto_interval={auto_process_interval}s, "
            f"min_frames={min_frames_for_processing}{mode_info}"
        )

    async def async_init(self):
        """Initialize agent and start video processing."""
        # Initialize base agent first
        await super().async_init()

        self.logger.info("Initializing VisualAgent...")

        # Start frame source
        await self.frame_source.start()

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
                    frames = await self.frame_source.get_frames()
                    audio = await self.frame_source.get_audio()

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
                        frames=frames,
                        audio=audio,
                        prompt="What is happening in the video?",
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
        # Process frames through embedding function if provided
        if self.embedding_function is not None:
            self.logger.debug(
                f"Encoding {len(frames)} frames using custom embedding function"
            )

            try:
                # Call the user-provided embedding function
                # Function should be async and return embeddings
                visual_data = await self.embedding_function(frames)

                # Handle different return types
                if isinstance(visual_data, dict):
                    # Function returned dict with embeddings and metadata
                    embeddings = visual_data.get("embeddings", visual_data)
                    metadata = {
                        k: v for k, v in visual_data.items() if k != "embeddings"
                    }
                    self.logger.debug(f"Embedding metadata: {metadata}")
                else:
                    # Function returned raw embeddings
                    embeddings = visual_data

                # Create multimodal message with embeddings
                multimodal_content = {
                    "text": prompt,
                    "image_embeddings": embeddings,  # Use 'image_embeddings' key for provider compatibility
                }

                self.logger.debug(
                    f"Using visual embeddings: shape={embeddings.shape if hasattr(embeddings, 'shape') else 'N/A'}"
                )

            except Exception as e:
                self.logger.error(f"Error in embedding function: {e}", exc_info=True)
                # Fall back to raw frames
                self.logger.warning("Falling back to raw frames due to embedding error")
                multimodal_content = {"text": prompt, "images": frames}
        else:
            # Use raw frames (default behavior)
            multimodal_content = {"text": prompt, "images": frames}

            self.logger.debug(f"Using raw frames: {len(frames)} frames")

        # Add audio if available
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
            f"audio={audio is not None}, prompt='{prompt[:50]}...', "
            f"mode={'embeddings' if self.embedding_function else 'frames'}"
        )

        # Call LLM (handles multimodal via provider)
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
        frames = await self.frame_source.get_frames()
        audio = await self.frame_source.get_audio()

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
        self,
        question: str,
        user_id: Optional[str] = None,
        max_iterations: Optional[int] = None,
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
        frames = await self.frame_source.get_frames()
        audio = await self.frame_source.get_audio()

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

        # Stop frame source
        await self.frame_source.stop()

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
