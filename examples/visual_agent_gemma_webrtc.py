"""
Visual Agent with Gemma 3n E4B - Edge AI Model (WIP)

⚠️ WORK IN PROGRESS - Gemma 3n requires specific integration
This example is a template for future Gemma integration.

Current status:
- TransformersProvider: ✅ Ready (universal HF models)
- Gemma 3n E4B: ⚠️ Requires model-specific integration
- Recommendation: Use visual_agent_gemini_test.py (Gemini API - fully working)

This example demonstrates the architecture for:
1. Setting up WebRTC driver for low-latency video/audio capture
2. Creating VisualAgent with multimodal models
3. Auto-processing video stream at intervals
4. Manual queries with current video buffer
5. Publishing responses via MQTT

For working visual agent, use:
    python examples/visual_agent_gemini_test.py

TODO: Integrate Gemma 3n with proper formatting (requires research)
"""

import asyncio
import signal
import sys

from argentic.core.agent.visual_agent import VisualAgent
from argentic.core.drivers import WebRTCDriver
from argentic.core.llm.llm_factory import LLMFactory
from argentic.core.messager import Messager

# For mock testing with bird.jpg
from pathlib import Path
from typing import List
import numpy as np
from PIL import Image


class MockWebRTCDriver:
    """Mock WebRTC driver using bird.jpg as static video frames"""

    def __init__(self, image_path: str, video_buffer_size: int = 1, **kwargs):
        """Mock driver - use 1 frame to minimize VRAM usage"""
        self.image_path = image_path
        self.video_buffer_size = video_buffer_size
        self._frames = None
        self._connected = False
        self._capturing = False

    async def connect(self, offer_sdp: str = None):
        """Load bird image and create frames"""
        self._connected = True
        img = Image.open(self.image_path)
        img_array = np.array(img.convert("RGB"))
        # Create buffer_size frames (all the same bird)
        self._frames = [img_array.copy() for _ in range(self.video_buffer_size)]
        return None

    async def start_capture(self):
        self._capturing = True

    async def stop_capture(self):
        self._capturing = False

    async def disconnect(self):
        self._connected = False
        self._frames = None

    async def get_frame_buffer(self) -> List[np.ndarray]:
        if self._frames:
            return self._frames
        return []

    async def get_audio_buffer(self):
        return None

    def clear_buffers(self):
        pass


# Configuration - using Transformers provider (universal for HF models)
CONFIG = {
    "llm": {
        "provider": "transformers",  # or "gemma" as alias
        "hf_model_id": "google/gemma-3n-E4B-it",
        # "hf_model_path": "./models/gemma-3n-4b",  # Optional local path
        "hf_device": "auto",  # "cuda" for GPU, "cpu", or "auto"
        "hf_torch_dtype": "float16",  # "float16", "bfloat16", "float32", or "auto"
        "max_new_tokens": 128,  # Reduced for faster inference
        "temperature": 0.7,
        "top_p": 0.95,
        "do_sample": True,
    },
    "messaging": {"broker_address": "localhost", "port": 1883},
    "webrtc": {
        "signaling_url": None,  # Set if using signaling server
        "ice_servers": [{"urls": "stun:stun.l.google.com:19302"}],
        "video_buffer_size": 1,  # Use 1 frame for minimal VRAM
        "frame_rate": 10,
        "audio_buffer_duration": 5.0,
        "resize_frames": (640, 480),  # Resize frames for efficiency
        "enable_audio": True,
        "frame_processor_workers": 4,
    },
    "visual_agent": {
        "auto_process_interval": 5.0,  # Process every 5 seconds
        "enable_auto_processing": True,
        "visual_response_topic": "agent/visual/response",
        "min_frames_for_processing": 1,  # Process with just 1 frame
        "system_prompt": "You are a helpful visual AI assistant. Describe what you see in the video clearly and concisely.",
    },
}


async def main():
    """Main application flow."""
    print("=== Visual Agent with Gemma 3n E4B ===\n")

    # 1. Setup Messager (for responses only, not video)
    print("1. Connecting to MQTT broker...")
    messager = Messager(
        broker_address=CONFIG["messaging"]["broker_address"],
        port=CONFIG["messaging"]["port"],
    )
    await messager.connect()
    print("   ✓ MQTT connected\n")

    # 2. Create Mock WebRTC driver with bird.jpg
    print("2. Initializing Mock WebRTC driver with bird.jpg...")
    bird_image = Path(__file__).parent / "bird.jpg"
    if not bird_image.exists():
        print(f"   ❌ Error: {bird_image} not found!")
        return

    driver = MockWebRTCDriver(
        image_path=str(bird_image),
        video_buffer_size=CONFIG["webrtc"]["video_buffer_size"],
    )
    print(
        f"   ✓ Mock driver initialized with {bird_image.name} ({CONFIG['webrtc']['video_buffer_size']} frames)\n"
    )

    # 3. Create Gemma provider
    print("3. Loading Gemma 3n E4B model...")
    print("   (This may take a moment - loading several GB)")
    llm = LLMFactory.create(CONFIG, messager)
    print("   ✓ Gemma provider created\n")

    # 4. Create VisualAgent (NO auto-processing - single test)
    print("4. Creating VisualAgent...")
    agent = VisualAgent(
        llm=llm,
        messager=messager,
        webrtc_driver=driver,
        enable_auto_processing=False,  # Single test only
        visual_response_topic=CONFIG["visual_agent"]["visual_response_topic"],
        system_prompt=CONFIG["visual_agent"]["system_prompt"],
        role="visual_assistant",
    )
    print("   ✓ VisualAgent created\n")

    # 5. Initialize agent (connects mock driver with bird frames)
    print("5. Initializing agent...")
    await agent.async_init()
    print("   ✓ Agent initialized\n")

    # 6. ONE question - ONE answer - DONE
    print("6. Asking question about the bird...")
    question = "Describe the bird you see in this video. What species might it be?"
    print(f"   Question: {question}\n")

    print("7. Processing with Gemma 3n E4B...")
    print("   (This will take a moment for model inference)\n")

    try:
        response = await agent.query_with_video(question)

        print("=" * 70)
        print("GEMMA RESPONSE:")
        print("-" * 70)
        print(response)
        print("=" * 70)
        print()

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()

    # 8. Cleanup
    print("\n8. Cleaning up...")
    await agent.stop()
    await messager.disconnect()
    print("   ✓ Done\n")

    print("✨ Test completed!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
