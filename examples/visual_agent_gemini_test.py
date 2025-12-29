"""
Visual Agent Test with Google Gemini 2.5 Flash (No Model Download Required!)

This uses Google Gemini 2.5 Flash - the latest multimodal model with vision support.
Just set your API key in .env and run!

Usage:
    # Add to .env file in project root:
    GEMINI_API_KEY="your-key-here"

    # Then run:
    python examples/visual_agent_gemini_test.py
"""

import asyncio
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Add parent to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Load .env file from project root
try:
    from dotenv import load_dotenv

    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Loaded environment from {env_path}\n")
except ImportError:
    print("⚠️  python-dotenv not installed. Using system environment variables.\n")
    print("   Install with: pip install python-dotenv\n")

from argentic.core.agent.visual_agent import VisualAgent
from argentic.core.llm.llm_factory import LLMFactory
from argentic.core.messager import Messager


class MockWebRTCDriver:
    """Mock driver using bird.jpg"""

    def __init__(self, image_path: str, num_frames: int = 30):
        self.image_path = image_path
        self.num_frames = num_frames
        self._frames = None

    async def connect(self, offer_sdp=None):
        img = Image.open(self.image_path)
        img_array = np.array(img.convert("RGB"))
        self._frames = [img_array.copy() for _ in range(self.num_frames)]
        print(f"   📸 Loaded {len(self._frames)} frames from {Path(self.image_path).name}")
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


async def main():
    print("=" * 70)
    print("  Visual Agent with Google Gemini 2.5 Flash")
    print("=" * 70)
    print()

    # Check API key (try both variable names)
    api_key = os.getenv("GOOGLE_GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GOOGLE_GEMINI_API_KEY or GEMINI_API_KEY not set!")
        print()
        print("Get your API key from: https://makersuite.google.com/app/apikey")
        print("Then add to .env file: GEMINI_API_KEY='your-key-here'")
        return

    print("✅ Using Gemini API key from environment\n")

    # Configuration - Google Gemini 2.5 Flash (latest multimodal model)
    CONFIG = {
        "llm": {
            "provider": "google_gemini",
            "google_gemini_model_name": "gemini-2.5-flash",  # Latest with vision
            "google_gemini_api_key": api_key,
            "google_gemini_parameters": {
                "temperature": 0.7,
                "top_p": 0.95,
                "max_output_tokens": 2048,
            },
        },
        "messaging": {"broker_address": "localhost", "port": 1883},
    }

    try:
        # 1. Connect MQTT
        print("1. Connecting to MQTT...")
        messager = Messager(
            broker_address=CONFIG["messaging"]["broker_address"],
            port=CONFIG["messaging"]["port"],
        )
        await messager.connect()
        print("   ✓ Connected\n")

        # 2. Create mock driver
        print("2. Loading bird.jpg...")
        bird_image = Path(__file__).parent / "bird.jpg"
        if not bird_image.exists():
            print(f"   ❌ Error: {bird_image} not found!")
            return

        driver = MockWebRTCDriver(str(bird_image), num_frames=5)  # 5 frames for video context
        print()

        # 3. Create Gemini provider
        print("3. Initializing Google Gemini 2.5 Flash...")
        llm = LLMFactory.create(CONFIG, messager)
        print("   ✓ Ready\n")

        # 4. Create VisualAgent
        print("4. Creating VisualAgent...")
        agent = VisualAgent(
            llm=llm,
            messager=messager,
            webrtc_driver=driver,
            enable_auto_processing=False,
            system_prompt="You are a helpful AI that describes images accurately and concisely.",
        )
        await agent.async_init()
        print("   ✓ Initialized\n")

        # 5. Ask about the bird
        print("5. Asking Gemini about the bird...")
        question = "Describe the bird you see in this video. What species might it be? What colors and features do you notice? In 2 sentences or less."
        print(f"   Question: {question}\n")

        print("6. Processing...")
        response = await agent.query_with_video(question)

        print()
        print("=" * 70)
        print("GEMINI RESPONSE:")
        print("-" * 70)
        if response:
            print(response)
        else:
            print("(Empty response)")
        print("=" * 70)
        print()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # 7. Cleanup
        print("7. Cleaning up...")
        try:
            if "agent" in locals():
                await agent.stop()
            if "messager" in locals():
                await messager.disconnect()
            print("   ✓ Done\n")
        except Exception as cleanup_error:
            print(f"   Warning during cleanup: {cleanup_error}")

        print("✨ Test completed!")


if __name__ == "__main__":
    asyncio.run(main())
