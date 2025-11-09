"""
Visual Agent with Gemma 3 (JAX) - Official Google Implementation

Uses official `gemma` library from google-deepmind for multimodal inference.

Requirements:
    pip install "jax[cuda12_local]" gemma argentic

Environment:
    KAGGLE_USERNAME=your_username
    KAGGLE_KEY=your_key
"""

import asyncio
from pathlib import Path

from argentic.core.agent.visual_agent import VisualAgent
from argentic.core.llm.llm_factory import LLMFactory
from argentic.core.messager import Messager
from PIL import Image
import numpy as np


class MockWebRTCDriver:
    """Mock driver for testing with static image."""

    def __init__(self, image_path: str, video_buffer_size: int = 1):
        self.image_path = image_path
        self.video_buffer_size = video_buffer_size
        self._frames = None

    async def connect(self, offer_sdp: str = None):
        img = Image.open(self.image_path)
        img_array = np.array(img.convert("RGB"))
        self._frames = [img_array.copy() for _ in range(self.video_buffer_size)]
        return None

    async def start_capture(self):
        pass

    async def stop_capture(self):
        pass

    async def disconnect(self):
        self._frames = None

    async def get_frame_buffer(self):
        return self._frames or []

    async def get_audio_buffer(self):
        return None

    def clear_buffers(self):
        pass


CONFIG = {
    "llm": {
        "provider": "gemma_jax",
        "gemma_model_size": "E4B",  # Gemma3n E4B (multimodal)
        # gemma_checkpoint_path not set - auto-download from gs:// via gcloud
        # Requires: gcloud auth application-default login
        "max_output_tokens": 128,
        "temperature": 0.7,
    },
    "messaging": {"broker_address": "localhost", "port": 1883},
}


async def main():
    print("=== Visual Agent with Gemma 3 (JAX) ===\n")

    # Setup
    print("1. Connecting to MQTT...")
    messager = Messager(
        broker_address=CONFIG["messaging"]["broker_address"],
        port=CONFIG["messaging"]["port"],
    )
    await messager.connect()
    print("   ✓ Connected\n")

    print("2. Loading Gemma 3n E4B model (JAX)...")
    print("   (Loading from local checkpoint - ~10GB RAM needed)")
    llm = LLMFactory.create(CONFIG, messager)
    print("   ✓ Model ready\n")

    print("3. Setting up mock driver with bird.jpg...")
    bird_path = Path(__file__).parent / "bird.jpg"
    if not bird_path.exists():
        print(f"   ❌ {bird_path} not found!")
        return
    driver = MockWebRTCDriver(str(bird_path))
    print("   ✓ Driver ready\n")

    print("4. Creating VisualAgent...")
    agent = VisualAgent(
        llm=llm,
        messager=messager,
        webrtc_driver=driver,
        enable_auto_processing=False,
        system_prompt="You are a helpful visual AI assistant.",
        role="visual_assistant",
    )
    await agent.async_init()
    print("   ✓ Agent ready\n")

    print("5. Asking question...")
    question = "Describe the bird in this image. What species might it be?"
    print(f"   Q: {question}\n")

    try:
        response = await agent.query_with_video(question)
        print("=" * 70)
        print("GEMMA RESPONSE:")
        print("-" * 70)
        print(response)
        print("=" * 70)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()

    print("\n6. Cleaning up...")
    await agent.stop()
    await messager.disconnect()
    print("   ✓ Done\n")

    print("✨ Test complete!")


if __name__ == "__main__":
    asyncio.run(main())
