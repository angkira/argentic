"""
Simple Visual Agent Test with Static Bird Image

This is a minimal test that:
1. Loads bird.jpg as mock video frames
2. Creates VisualAgent with mock driver
3. Asks one question
4. Gets one answer
5. Exits

No continuous processing, no real WebRTC - just a quick test.
"""

import asyncio
import sys
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from argentic.core.agent.visual_agent import VisualAgent
from argentic.core.llm.llm_factory import LLMFactory
from argentic.core.messager import Messager


class MockWebRTCDriver:
    """Mock WebRTC driver that uses bird.jpg as static frames"""

    def __init__(self, image_path: str, num_frames: int = 10):
        self.image_path = image_path
        self.num_frames = num_frames
        self._frames = None
        self._connected = False
        self._capturing = False
        print(f"   📸 Mock driver: Loading {image_path}")

    async def connect(self, offer_sdp: str = None):
        """Mock connection"""
        self._connected = True
        # Load the bird image
        img = Image.open(self.image_path)
        # Convert to numpy array (RGB)
        img_array = np.array(img.convert("RGB"))
        # Create multiple frames (all the same bird)
        self._frames = [img_array.copy() for _ in range(self.num_frames)]
        print(f"   ✓ Mock driver connected with {len(self._frames)} frames ({img_array.shape})")
        return None

    async def start_capture(self):
        """Mock capture start"""
        self._capturing = True
        print("   ✓ Mock capture started")

    async def stop_capture(self):
        """Mock capture stop"""
        self._capturing = False

    async def disconnect(self):
        """Mock disconnect"""
        self._connected = False
        self._frames = None

    async def get_frame_buffer(self) -> List[np.ndarray]:
        """Return our mock frames"""
        if self._frames:
            return self._frames
        return []

    async def get_audio_buffer(self):
        """No audio in this test"""
        return None

    def clear_buffers(self):
        """Mock buffer clear"""
        pass


async def main():
    print("=" * 70)
    print("  Visual Agent - Simple Bird Test")
    print("=" * 70)
    print()

    # Configuration
    CONFIG = {
        "llm": {
            "provider": "gemma",
            "gemma_model_name": "gemma-3n-e4b-it",
            "gemma_checkpoint_path": "GEMMA3_4B_IT",
            "gemma_enable_ple_caching": True,
            "gemma_parameters": {
                "temperature": 0.7,
                "top_p": 0.95,
                "max_output_tokens": 512,
            },
        },
        "messaging": {"broker_address": "localhost", "port": 1883},
    }

    try:
        # 1. Connect to MQTT broker
        print("1. Connecting to MQTT broker...")
        messager = Messager(
            broker_address=CONFIG["messaging"]["broker_address"],
            port=CONFIG["messaging"]["port"],
        )
        await messager.connect()
        print("   ✓ MQTT connected")
        print()

        # 2. Create mock WebRTC driver with bird image
        print("2. Creating mock driver with bird.jpg...")
        bird_image_path = Path(__file__).parent / "bird.jpg"
        if not bird_image_path.exists():
            print(f"   ❌ Error: {bird_image_path} not found!")
            return

        mock_driver = MockWebRTCDriver(
            image_path=str(bird_image_path), num_frames=10  # 10 static bird frames
        )
        print()

        # 3. Create Gemma provider
        print("3. Loading Gemma 3n E4B model...")
        print("   (This may take a moment - loading several GB)")
        llm = LLMFactory.create(CONFIG, messager)
        print("   ✓ Gemma provider created")
        print()

        # 4. Create VisualAgent
        print("4. Creating VisualAgent...")
        agent = VisualAgent(
            llm=llm,
            messager=messager,
            webrtc_driver=mock_driver,
            auto_process_interval=5.0,
            enable_auto_processing=False,  # Disable auto-processing for this test
            visual_response_topic="agent/visual/response",
            system_prompt="You are a visual AI assistant. Describe what you see in the images clearly and concisely.",
        )
        print("   ✓ VisualAgent created")
        print()

        # 5. Initialize agent (connects mock driver)
        print("5. Initializing agent...")
        await agent.async_init()
        print()

        # 6. Ask one question about the bird
        print("6. Asking question about the video...")
        question = "What do you see in this video? Describe the bird."
        print(f"   Question: {question}")
        print()

        print("7. Processing with Gemma 3n E4B...")
        print("   (This will take a moment for model inference)")
        response = await agent.query_with_video(question)
        print()

        # 8. Show response
        print("8. Response received!")
        print("=" * 70)
        print("GEMMA RESPONSE:")
        print("-" * 70)
        print(response)
        print("=" * 70)
        print()

        # 9. Cleanup
        print("9. Cleaning up...")
        await agent.stop()
        await messager.disconnect()
        print("   ✓ Done")
        print()

        print("✨ Test completed successfully!")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print()


if __name__ == "__main__":
    asyncio.run(main())
