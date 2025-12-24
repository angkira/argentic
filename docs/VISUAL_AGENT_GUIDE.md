# Visual Agent with Gemma 3n - Complete Guide

## Overview

The Visual Agent extends Argentic's capabilities to process real-time video and audio through multimodal AI models. It uses:

- **Frame Source Abstraction**: Flexible interface for video/image input from any source (WebRTC, video files, cameras, static images, custom callbacks)
- **Multiple LLM Providers**: Support for Gemma 3n, vLLM, OpenAI-compatible APIs, and more
- **Custom Embedding Functions**: Optional embedding function support for pre-computing visual features (supported by VLLMNativeProvider)
- **Async/Threading Architecture**: Non-blocking operation with proper thread pool management
- **MQTT Integration**: Responses distributed via MQTT while video processing happens independently

## Frame Sources

The Visual Agent uses a **frame source abstraction** that allows you to provide video/image input from any source:

### Available Frame Sources

1. **WebRTCFrameSource**: Real-time WebRTC video streaming (browser to server)
2. **VideoFileFrameSource**: Pre-recorded video files (mp4, avi, mkv, etc.)
3. **StaticFrameSource**: Fixed set of images (useful for testing)
4. **FunctionalFrameSource**: Custom callback-based source (for cameras, screen capture, etc.)

All frame sources implement the same interface, so you can easily swap them without changing your agent code.

### Visual Processing Modes

The Visual Agent supports two modes of visual processing:

1. **Direct Frames Mode** (default): Raw video frames are passed directly to the LLM provider (e.g., Gemma 3n), which handles visual encoding internally.

2. **Embeddings Mode**: Frames are first processed by a custom embedding function to produce embeddings, which are then passed to the LLM. This is useful when:
   - You have a separate vision encoder (CLIP, ViT, custom CNN, etc.)
   - You want to pre-compute visual features for efficiency
   - You need to separate visual encoding from language processing

**Important**: Embedding support varies by provider:
- ✅ **Supported**: `VLLMNativeProvider` (with `enable_mm_embeds=True`)
- ❌ **Not Supported**: `VLLMProvider` (OpenAI API), `GoogleGeminiProvider`, `OpenAICompatibleProvider`
- ⚠️ **Model-Dependent**: `TransformersProvider` (depends on model architecture)

📖 See [LLM Providers Guide](./LLM_PROVIDERS_GUIDE.md) for detailed provider documentation and examples.

## Architecture

### Direct Frames Mode (Default)

```
┌─────────────────────────┐
│    Frame Source         │ (WebRTC/File/Camera/Static)
│   - WebRTCFrameSource   │
│   - VideoFileFrameSource│
│   - StaticFrameSource   │
│   - FunctionalSource    │
└────────┬────────────────┘
         │ Implements FrameSource interface
         │ get_frames(), get_audio()
         ▼
┌──────────────────────────┐
│   VisualAgent            │
│  ┌─────────────────┐     │
│  │ Auto-Process    │     │ asyncio.Task
│  │     Loop        │     │
│  └─────────────────┘     │
└─────────┬────────────────┘
          │ Multimodal Message
          │ (frames + audio + text)
          ▼
┌───────────────────────────┐
│   LLM Provider            │
│  - VLLMNativeProvider     │
│  - OpenAICompatible       │
│  - GoogleGemini           │
│  ┌──────────────────┐     │
│  │ Inference Pool   │     │ Thread Pool
│  │ (GPU-intensive)  │     │ (GPU-intensive)
│  └──────────────────┘     │
└─────────┬─────────────────┘
          │ Text Response
          ▼
┌─────────────────────┐
│   Messager (MQTT)   │
└──────────┬──────────┘
           │
           ▼
    Clients/Other Agents
```

### Embeddings Mode (with Embedding Function)

```
┌─────────────────────────┐
│    Frame Source         │ (Any source)
└────────┬────────────────┘
         │ Raw Frames
         ▼
┌──────────────────────────┐
│   VisualAgent            │
│  ┌─────────────────┐     │
│  │ Auto-Process    │     │ asyncio.Task
│  │     Loop        │     │
│  └─────────────────┘     │
└─────────┬────────────────┘
          │ Frames
          ▼
┌───────────────────────────┐
│   embedding_function()    │ User-provided async function
│  ┌──────────────────┐     │
│  │ Custom Encoder   │     │ Async processing
│  │ (CLIP/ViT/CNN)   │     │ (can be GPU-based)
│  └──────────────────┘     │
└─────────┬─────────────────┘
          │ Visual Embeddings (np.ndarray)
          ▼
┌──────────────────────────┐
│   VisualAgent            │
│  Formats multimodal msg  │
│  {text, image_embeddings}│
└─────────┬────────────────┘
          │
          ▼
┌───────────────────────────┐
│   LLM Provider            │
│  - VLLMNativeProvider     │
│    (only one with support)│
│  - Others raise exception │
└─────────┬─────────────────┘
          │ Text Response
          ▼
┌─────────────────────┐
│   Messager (MQTT)   │
└──────────┬──────────┘
           │
           ▼
    Clients/Other Agents
```

## Installation

### 1. Base Installation

```bash
# Install Argentic with visual agent support
pip install -e ".[visual,gemma]"

# Or install components separately
pip install argentic
pip install aiortc av opencv-python  # WebRTC support
pip install gemma jax[cuda12]         # Gemma 3n (with CUDA)
pip install numpy pillow               # Already in base dependencies
```

### 2. Download Gemma 3n Model

```bash
# Option 1: Using Kaggle API
pip install kaggle
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_key

kaggle models instances versions download google/gemma-3n/jax/gemma-3n-e4b-it

# Option 2: Download manually from Kaggle
# https://www.kaggle.com/models/google/gemma-3n/
# Extract to a local directory and note the path
```

### 3. Setup MQTT Broker

```bash
# Using Docker (recommended)
docker run -d -p 1883:1883 --name mosquitto eclipse-mosquitto:2.0

# Or install locally
sudo apt install mosquitto mosquitto-clients  # Ubuntu/Debian
brew install mosquitto                         # macOS
```

## Configuration

### Basic Configuration (YAML)

```yaml
llm:
  provider: gemma
  gemma_model_name: gemma-3n-e4b-it
  gemma_checkpoint_path: /path/to/downloaded/checkpoint
  gemma_enable_ple_caching: true
  gemma_parameters:
    temperature: 0.7
    top_p: 0.95
    max_output_tokens: 2048

messaging:
  broker_address: localhost
  port: 1883

webrtc:
  video_buffer_size: 30
  frame_rate: 10
  resize_frames: [640, 480]
  enable_audio: true

visual_agent:
  auto_process_interval: 5.0
  enable_auto_processing: true
  min_frames_for_processing: 10
```

**Note**: Embedding function configuration is done in Python code, not in YAML config (see examples below).

## Usage Examples

### Example 1: Basic Visual Agent with WebRTC

```python
import asyncio
from argentic.core.agent.visual_agent import VisualAgent
from argentic.core.agent.frame_sources import WebRTCFrameSource
from argentic.core.drivers import WebRTCDriver
from argentic.core.llm.providers.vllm_provider import VLLMProvider
from argentic.core.messager import Messager

async def main():
    # Setup components
    messager = Messager(broker_address="localhost", port=1883)
    await messager.connect()

    # Create WebRTC driver
    driver = WebRTCDriver(
        video_buffer_size=30,
        frame_rate=10,
        resize_frames=(640, 480),
        enable_audio=True
    )

    # Wrap driver in frame source
    frame_source = WebRTCFrameSource(driver)

    llm = VLLMProvider(config={
        "vllm_base_url": "http://localhost:8000/v1",
        "vllm_model": "llava-hf/llava-v1.6-mistral-7b-hf"
    })

    # Create visual agent
    agent = VisualAgent(
        llm=llm,
        messager=messager,
        frame_source=frame_source,  # Use frame_source parameter
        auto_process_interval=5.0,
        system_prompt="You are a visual AI assistant."
    )

    # Initialize and start
    await agent.async_init()

    # Query with video
    response = await agent.query_with_video("What do you see?")
    print(response)

    # Cleanup
    await agent.stop()
    await messager.disconnect()

asyncio.run(main())
```

### Example 2: WebRTC Connection Setup

```python
# Server-side (receives video from client)
async def setup_webrtc_server():
    driver = WebRTCDriver(...)
    
    # Get offer from client (via signaling)
    offer_sdp = await receive_offer_from_client()
    
    # Create answer
    answer_sdp = await driver.connect(offer_sdp=offer_sdp)
    
    # Send answer back to client
    await send_answer_to_client(answer_sdp)
    
    # Start capturing
    await driver.start_capture()
    
    return driver
```

### Example 3: Custom Frame Processing

```python
async def on_new_frame(frame, timestamp):
    """Callback for each new frame."""
    print(f"New frame at {timestamp}: shape={frame.shape}")

driver = WebRTCDriver(...)
driver.set_frame_callback(on_new_frame)
await driver.connect()
await driver.start_capture()
```

### Example 4: Using Video File Source

```python
from argentic.core.agent.frame_sources import VideoFileFrameSource

# Create video file source
frame_source = VideoFileFrameSource(
    video_path="/path/to/video.mp4",
    buffer_size=30,
    fps=10,  # Subsample to 10 FPS
    loop=False
)

# Create agent with video file source
agent = VisualAgent(
    llm=llm,
    messager=messager,
    frame_source=frame_source,
    enable_auto_processing=False  # Manual control for video files
)

await agent.async_init()

# Process video
response = await agent.query_with_video("What's happening in this video?")
print(response)

await agent.stop()
```

### Example 5: Using Static Frames (Testing)

```python
import numpy as np
from argentic.core.agent.frame_source import StaticFrameSource

# Create some test frames
frames = [
    np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    for _ in range(10)
]

# Create static frame source
frame_source = StaticFrameSource(frames)

# Use with agent
agent = VisualAgent(
    llm=llm,
    messager=messager,
    frame_source=frame_source,
    enable_auto_processing=False
)

await agent.async_init()
response = await agent.query_with_video("What do you see?")
print(response)
await agent.stop()
```

### Example 6: Custom Functional Frame Source

```python
from argentic.core.agent.frame_source import FunctionalFrameSource
import cv2

# Define custom frame capture function
async def capture_screen_frames():
    """Capture frames from screen (example)."""
    frames = []
    # Your custom logic here (e.g., screen capture, camera, etc.)
    # For example, using opencv to capture from camera:
    cap = cv2.VideoCapture(0)
    for _ in range(10):
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    cap.release()
    return frames

# Create functional frame source
frame_source = FunctionalFrameSource(get_frames_fn=capture_screen_frames)

# Use with agent
agent = VisualAgent(
    llm=llm,
    messager=messager,
    frame_source=frame_source,
    enable_auto_processing=True,  # Auto-capture and process
    auto_process_interval=5.0
)

await agent.async_init()
# Agent will automatically call capture_screen_frames() every 5 seconds
```

### Example 7: Using Embedding Function (CLIP Example)

**Note**: This example demonstrates the embedding function approach. Most providers (Gemma, Gemini, vLLM OpenAI API) do NOT support pre-computed embeddings and will raise `NotImplementedError`. Use VLLMNativeProvider for embedding support.

```python
import asyncio
import numpy as np
from typing import List
from argentic.core.agent.visual_agent import VisualAgent
from argentic.core.agent.frame_source import StaticFrameSource
from argentic.core.messager import Messager

# Define your custom embedding function
async def clip_embedding_function(frames: List[np.ndarray]) -> np.ndarray:
    """
    Custom embedding function using CLIP (example).

    Args:
        frames: List of numpy arrays (H, W, C)

    Returns:
        numpy array of shape (num_frames, embedding_dim)
    """
    # Example with torch and CLIP (install: pip install torch transformers)
    import torch
    from transformers import CLIPProcessor, CLIPModel
    from PIL import Image

    # Load CLIP model (in production, load this once outside the function)
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Convert numpy frames to PIL Images
    pil_images = [Image.fromarray(frame) for frame in frames]

    # Process with CLIP
    inputs = processor(images=pil_images, return_tensors="pt")

    # Get embeddings (run in thread pool if CPU-bound)
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)

    # Return as numpy array
    return embeddings.cpu().numpy()

async def main():
    # Setup components
    messager = Messager(broker_address="localhost", port=1883)
    await messager.connect()

    # Create test frames
    frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(10)]
    frame_source = StaticFrameSource(frames)

    # NOTE: You need VLLMNativeProvider for embedding support
    # Other providers (Gemma, Gemini, vLLM OpenAI API) will raise NotImplementedError
    from argentic.core.llm.providers.vllm_native import VLLMNativeProvider

    llm = VLLMNativeProvider(config={
        "model_name": "llava-hf/llava-1.5-7b-hf",
        "enable_mm_embeds": True,  # Required for embeddings
    })

    # Create visual agent with embedding function
    agent = VisualAgent(
        llm=llm,
        messager=messager,
        frame_source=frame_source,
        embedding_function=clip_embedding_function,  # Pass the async function
        enable_auto_processing=False,
        system_prompt="You are a visual AI assistant."
    )

    # Initialize and start
    await agent.async_init()

    # Query with video - frames will be encoded to embeddings automatically
    try:
        response = await agent.query_with_video("What do you see?")
        print(response)
    except NotImplementedError as e:
        print(f"Provider does not support embeddings: {e}")

    # Cleanup
    await agent.stop()
    await messager.disconnect()

asyncio.run(main())
```

### Example 8: Embedding Function with Error Handling

```python
import asyncio
import numpy as np

async def robust_embedding_function(frames: List[np.ndarray]):
    """
    Embedding function with error handling and return metadata.

    Returns dict with embeddings and metadata.
    """
    try:
        # Your encoding logic here
        # embeddings = your_model.encode(frames)

        # Placeholder: random embeddings
        embeddings = np.random.randn(len(frames), 512).astype(np.float32)

        # Return dict with metadata
        return {
            "embeddings": embeddings,
            "num_frames": len(frames),
            "embedding_dim": 512,
            "model": "custom-encoder-v1"
        }
    except Exception as e:
        print(f"Embedding error: {e}")
        # Return None or raise - VisualAgent will fall back to raw frames
        raise

# Use with VisualAgent
agent = VisualAgent(
    llm=llm,
    messager=messager,
    frame_source=frame_source,
    embedding_function=robust_embedding_function,
)
```

## Embedding Function API

### Function Signature

Your embedding function must be async and have this signature:

```python
async def embedding_function(frames: List[np.ndarray]) -> Union[np.ndarray, Dict[str, Any]]:
    """
    Args:
        frames: List of numpy arrays with shape (H, W, C)

    Returns:
        - np.ndarray: Embeddings with shape (num_frames, embedding_dim)
        - Dict: Dictionary with 'embeddings' key and optional metadata

    Raises:
        Exception: On error, VisualAgent will fall back to raw frames
    """
    pass
```

### Best Practices

1. **Load models once** (not in the function):
   ```python
   # Global scope
   model = load_model()

   async def embed_frames(frames):
       # Use pre-loaded model
       return model.encode(frames)
   ```

2. **Use asyncio.to_thread** for blocking operations:
   ```python
   async def embed_frames(frames):
       # Run blocking code in thread pool
       embeddings = await asyncio.to_thread(model.encode, frames)
       return embeddings
   ```

3. **Return metadata** for debugging:
   ```python
   return {
       "embeddings": emb_array,
       "model_version": "v1.0",
       "processing_time": elapsed_time
   }
   ```

4. **Handle errors gracefully**:
   ```python
   try:
       return model.encode(frames)
   except Exception as e:
       logger.error(f"Encoding failed: {e}")
       raise  # VisualAgent will fall back to frames
   ```

## Threading and Performance

### Threading Architecture

The Visual Agent uses a multi-threaded architecture to prevent blocking:

1. **Main Asyncio Loop**
   - WebRTC signaling and connection management
   - MQTT messaging
   - Coordination between components

2. **Frame Processor Thread Pool** (WebRTCDriver)
   - Frame resizing and format conversion
   - Audio resampling
   - CPU-intensive preprocessing
   - Default: 4 workers

3. **Inference Thread Pool** (LLM Providers)
   - Model inference (CPU/GPU heavy)
   - Image encoding
   - Audio encoding
   - Provider-specific (varies by implementation)

### Performance Tuning

```python
# Adjust thread pool sizes
driver = WebRTCDriver(
    frame_processor_workers=8,  # More workers for faster frame processing
    video_buffer_size=60,        # Larger buffer for smoother operation
    frame_rate=15,               # Higher FPS
    resize_frames=(320, 240)     # Smaller frames for faster processing
)

# Gemma provider uses its own pool (2 workers by default)
# Increase if you have multiple GPUs
```

### Memory Management

- Frame buffers use `deque(maxlen=N)` for automatic size limiting
- Clear buffers periodically if not using auto-processing:
  ```python
  driver.clear_buffers()
  ```

- Enable PLE caching in Gemma to reduce memory:
  ```yaml
  gemma_enable_ple_caching: true
  ```

## WebRTC Integration Patterns

### Pattern 1: Browser to Server

```javascript
// Browser (JavaScript)
const pc = new RTCPeerConnection({
    iceServers: [{urls: 'stun:stun.l.google.com:19302'}]
});

// Add local video stream
navigator.mediaDevices.getUserMedia({video: true, audio: true})
    .then(stream => {
        stream.getTracks().forEach(track => pc.addTrack(track, stream));
    });

// Create offer
const offer = await pc.createOffer();
await pc.setLocalDescription(offer);

// Send offer to server (via WebSocket/HTTP)
await fetch('/webrtc/offer', {
    method: 'POST',
    body: JSON.stringify({sdp: offer.sdp})
});

// Receive answer from server
const answer = await fetch('/webrtc/answer').then(r => r.json());
await pc.setRemoteDescription(new RTCSessionDescription(answer));
```

```python
# Server (Python with Argentic)
from aiohttp import web

async def handle_offer(request):
    data = await request.json()
    offer_sdp = data['sdp']
    
    # Create answer
    answer_sdp = await driver.connect(offer_sdp=offer_sdp)
    
    return web.json_response({'sdp': answer_sdp})
```

### Pattern 2: Peer-to-Peer with Signaling Server

Use a signaling server (e.g., WebSocket) to exchange SDP offers/answers between peers.

## Troubleshooting

### Issue: "gemma library not found"

```bash
# Install gemma library
pip install gemma

# Verify installation
python -c "import gemma as gm; print(gm.__version__)"
```

### Issue: "JAX not found" or GPU not detected

```bash
# For CUDA 12.x
pip install jax[cuda12]

# For CPU only
pip install jax[cpu]

# Verify GPU
python -c "import jax; print(jax.devices())"
```

### Issue: "aiortc not available"

```bash
# Install WebRTC dependencies
pip install aiortc av

# On Ubuntu, you might need system libraries
sudo apt install libavdevice-dev libavfilter-dev libopus-dev libvpx-dev pkg-config
```

### Issue: No frames in buffer

- Check WebRTC connection is established: `driver._connected == True`
- Verify video track is received: `driver._video_track is not None`
- Check frame rate isn't too low: increase `frame_rate` parameter
- Look for errors in logs: `enable_dialogue_logging=True`

### Issue: Model inference is slow

- Use GPU: Install `jax[cuda12]` instead of `jax[cpu]`
- Reduce frame resolution: `resize_frames=(320, 240)`
- Use E2B model instead of E4B (smaller, faster)
- Enable PLE caching: `gemma_enable_ple_caching=true`

## API Reference

### FrameSource (Base Class)

```python
class FrameSource(ABC):
    """Base class for all frame sources."""

    async def get_frames() -> List[np.ndarray]
    async def get_audio() -> Optional[np.ndarray]
    async def start()
    async def stop()
    def clear_buffers()
    def get_info() -> dict
```

### WebRTCFrameSource

```python
class WebRTCFrameSource(FrameSource):
    """WebRTC video streaming frame source."""

    def __init__(driver: WebRTCDriver, config: Optional[FrameSourceConfig] = None)
    async def get_frames() -> List[np.ndarray]
    async def get_audio() -> Optional[np.ndarray]
    async def get_latest_frame() -> Optional[np.ndarray]
    def clear_buffers()
    async def start()  # Connects and starts capture
    async def stop()   # Stops capture and disconnects
    def get_info() -> dict
```

### VideoFileFrameSource

```python
class VideoFileFrameSource(FrameSource):
    """Video file frame source using OpenCV."""

    def __init__(
        video_path: str,
        buffer_size: int = 30,
        fps: Optional[int] = None,  # None = use native FPS
        loop: bool = False,
        config: Optional[FrameSourceConfig] = None
    )
    async def get_frames() -> List[np.ndarray]
    async def get_audio() -> Optional[np.ndarray]  # Not supported (returns None)
    def clear_buffers()
    async def start()  # Opens video file
    async def stop()   # Closes video file
    def get_info() -> dict
```

### StaticFrameSource

```python
class StaticFrameSource(FrameSource):
    """Static frame source for testing."""

    def __init__(frames: List[np.ndarray], config: Optional[FrameSourceConfig] = None)
    async def get_frames() -> List[np.ndarray]
    async def get_audio() -> Optional[np.ndarray]  # Returns None
    async def start()  # No-op
    async def stop()   # No-op
```

### FunctionalFrameSource

```python
class FunctionalFrameSource(FrameSource):
    """Frame source using callback functions."""

    def __init__(
        get_frames_fn: Callable[[], Awaitable[List[np.ndarray]]],
        get_audio_fn: Optional[Callable[[], Awaitable[Optional[np.ndarray]]]] = None,
        config: Optional[FrameSourceConfig] = None
    )
    async def get_frames() -> List[np.ndarray]  # Calls get_frames_fn
    async def get_audio() -> Optional[np.ndarray]  # Calls get_audio_fn if provided
    async def start()  # No-op
    async def stop()   # No-op
```

### VisualAgent

```python
class VisualAgent(Agent):
    def __init__(
        llm: ModelProvider,
        messager: Messager,
        frame_source: FrameSource,  # Any FrameSource implementation
        embedding_function: Optional[Callable] = None,  # Optional async embedding function
        auto_process_interval: float = 5.0,
        visual_prompt_template: str = "Describe what you see in the video: {question}",
        visual_response_topic: str = "agent/visual/response",
        enable_auto_processing: bool = True,
        process_on_buffer_full: bool = True,
        min_frames_for_processing: int = 10,
        **kwargs
    )

    async def async_init()
    async def query_with_video(question: str) -> str
    async def query(question: str, ...) -> str
    def pause_auto_processing()
    def resume_auto_processing()
    async def stop()
```

### Embedding Function Type

```python
# Type signature for embedding_function parameter
async def embedding_function(frames: List[np.ndarray]) -> Union[np.ndarray, Dict[str, Any]]:
    """
    Convert frames to embeddings.

    Args:
        frames: List of numpy arrays (H, W, C)

    Returns:
        np.ndarray: Shape (num_frames, embedding_dim)
        OR
        Dict: With 'embeddings' key and optional metadata

    Raises:
        Exception: On error (VisualAgent will fall back to raw frames)
    """
    ...
```

## Best Practices

1. **Always use async context managers**:
   ```python
   async with VisualAgent(...) as agent:
       response = await agent.query_with_video("...")
   ```

2. **Handle graceful shutdown**:
   ```python
   import signal
   
   def signal_handler(sig, frame):
       asyncio.create_task(agent.stop())
   
   signal.signal(signal.SIGINT, signal_handler)
   ```

3. **Monitor buffer sizes**:
   ```python
   frames = await driver.get_frame_buffer()
   print(f"Buffer size: {len(frames)} frames")
   ```

4. **Use appropriate frame rates**:
   - 5-10 FPS for basic monitoring
   - 15-24 FPS for smooth video analysis
   - 30+ FPS for high-motion scenarios

5. **Test without video first**:
   ```python
   # Test model loading
   llm = VLLMProvider(config={"vllm_base_url": "http://localhost:8000/v1"})
   response = await llm.ainvoke("Hello, can you see?")
   ```

## Frame Source Guide

### Creating Custom Frame Sources

You can create your own frame source by implementing the `FrameSource` interface:

```python
from argentic.core.agent.frame_source import FrameSource, FrameSourceConfig
from typing import List, Optional
import numpy as np

class CustomFrameSource(FrameSource):
    """Example custom frame source."""

    def __init__(self, config: Optional[FrameSourceConfig] = None):
        super().__init__(config)
        # Your initialization here

    async def get_frames(self) -> List[np.ndarray]:
        """Return list of video frames."""
        # Your logic to get frames
        frames = []
        # ... capture/generate frames ...
        return frames

    async def get_audio(self) -> Optional[np.ndarray]:
        """Return audio buffer (optional)."""
        # Return None if audio not supported
        return None

    async def start(self):
        """Start frame capture/generation."""
        # Initialize resources
        pass

    async def stop(self):
        """Stop and cleanup."""
        # Release resources
        pass

    def get_info(self) -> dict:
        """Return information about this source."""
        info = super().get_info()
        info.update({
            "custom_param": "value",
        })
        return info
```

### Migration from WebRTC-only API

If you have existing code using the old `webrtc_driver` parameter:

**Old (deprecated):**
```python
from argentic.core.drivers import WebRTCDriver

driver = WebRTCDriver(...)
agent = VisualAgent(
    llm=llm,
    messager=messager,
    webrtc_driver=driver,  # Old parameter
)
```

**New (recommended):**
```python
from argentic.core.drivers import WebRTCDriver
from argentic.core.agent.frame_sources import WebRTCFrameSource

driver = WebRTCDriver(...)
frame_source = WebRTCFrameSource(driver)
agent = VisualAgent(
    llm=llm,
    messager=messager,
    frame_source=frame_source,  # New parameter
)
```

The functionality is identical - WebRTCFrameSource is a simple adapter that wraps the driver.

## Examples

See complete working examples at:
- `examples/visual_agent_frame_sources.py` - Multiple frame source examples
- `examples/visual_agent_gemma_webrtc.py` - Full WebRTC example
- `examples/visual_agent_vllm.py` - vLLM with embedding support
- `examples/config_visual_gemma_webrtc.yaml` - Configuration example

## References

- Gemma 3n Documentation: https://ai.google.dev/gemma/docs/gemma-3n
- aiortc Documentation: https://aiortc.readthedocs.io/
- Argentic Framework: https://github.com/angkira/argentic

