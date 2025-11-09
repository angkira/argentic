# Visual Agent with Gemma 3n - Complete Guide

## Overview

The Visual Agent extends Argentic's capabilities to process real-time video and audio through multimodal AI models. It uses:

- **Gemma 3n E4B/E2B**: Google's multimodal model with native support for text, images, video, and audio
- **WebRTC Driver**: Low-latency video/audio capture without MQTT overhead
- **Async/Threading Architecture**: Non-blocking operation with proper thread pool management
- **MQTT Integration**: Responses distributed via MQTT while video stays in direct connection

## Architecture

```
┌─────────────────┐
│  WebRTC Stream  │ (Camera/Browser)
│  (Video/Audio)  │
└────────┬────────┘
         │ Low latency, direct connection
         │ (aiortc library)
         ▼
┌─────────────────────────┐
│   WebRTCDriver          │
│  ┌─────────────────┐    │
│  │ Frame Processor │    │ Thread Pool
│  │  Thread Pool    │    │ (CPU-intensive)
│  └─────────────────┘    │
│  ┌─────────────────┐    │
│  │ Frame Buffer    │    │ deque(maxlen=30)
│  │ Audio Buffer    │    │
│  └─────────────────┘    │
└─────────┬───────────────┘
          │ Async Queue
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
│   GemmaProvider           │
│  ┌──────────────────┐     │
│  │ Inference Pool   │     │ Thread Pool
│  │ (JAX/GPU heavy)  │     │ (GPU-intensive)
│  └──────────────────┘     │
│         Gemma 3n E4B      │
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

## Usage Examples

### Example 1: Basic Visual Agent

```python
import asyncio
from argentic.core.agent.visual_agent import VisualAgent
from argentic.core.drivers import WebRTCDriver
from argentic.core.llm.providers.gemma import GemmaProvider
from argentic.core.messager import Messager

async def main():
    # Setup components
    messager = Messager(broker_address="localhost", port=1883)
    await messager.connect()
    
    driver = WebRTCDriver(
        video_buffer_size=30,
        frame_rate=10,
        resize_frames=(640, 480),
        enable_audio=True
    )
    
    llm = GemmaProvider(config={
        "gemma_model_name": "gemma-3n-e4b-it",
        "gemma_checkpoint_path": "/path/to/checkpoint"
    })
    
    # Create visual agent
    agent = VisualAgent(
        llm=llm,
        messager=messager,
        webrtc_driver=driver,
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

### Example 4: Manual Buffer Control

```python
# Disable auto-processing
agent = VisualAgent(
    llm=llm,
    messager=messager,
    webrtc_driver=driver,
    enable_auto_processing=False  # Manual control
)
await agent.async_init()

# Manually process when needed
while True:
    await asyncio.sleep(10)
    
    frames = await driver.get_frame_buffer()
    if len(frames) >= 20:
        response = await agent.query_with_video("Describe the scene")
        print(response)
        driver.clear_buffers()  # Clear after processing
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

3. **Inference Thread Pool** (GemmaProvider)
   - JAX model inference (CPU/GPU heavy)
   - Image encoding
   - Audio encoding
   - Default: 2 workers

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

### WebRTCDriver

```python
class WebRTCDriver:
    async def connect(offer_sdp: Optional[str] = None) -> Optional[str]
    async def disconnect()
    async def start_capture()
    async def stop_capture()
    async def get_frame_buffer() -> List[np.ndarray]
    async def get_audio_buffer() -> Optional[np.ndarray]
    async def get_latest_frame() -> Optional[np.ndarray]
    def set_frame_callback(callback: Callable)
    def clear_buffers()
```

### VisualAgent

```python
class VisualAgent(Agent):
    async def async_init()
    async def query_with_video(question: str) -> str
    async def query(question: str, ...) -> str
    def pause_auto_processing()
    def resume_auto_processing()
    async def stop()
```

### GemmaProvider

```python
class GemmaProvider(ModelProvider):
    async def achat(messages: List[Dict], ...) -> LLMChatResponse
    async def ainvoke(prompt: str, ...) -> LLMChatResponse
    def get_model_name() -> str
    def supports_multimodal() -> bool
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
   llm = GemmaProvider(config={...})
   response = await llm.ainvoke("Hello, can you see?")
   ```

## Examples

See the complete working example at:
- `examples/visual_agent_gemma_webrtc.py`
- `examples/config_visual_gemma_webrtc.yaml`

## References

- Gemma 3n Documentation: https://ai.google.dev/gemma/docs/gemma-3n
- aiortc Documentation: https://aiortc.readthedocs.io/
- Argentic Framework: https://github.com/angkira/argentic

