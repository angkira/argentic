# LLM Providers Guide

## Overview

Argentic supports multiple LLM providers with different capabilities and use cases. This guide covers all available providers and their configurations.

## Provider Comparison

| Provider | Type | Embedding Support | Use Case |
|----------|------|-------------------|----------|
| `OpenAICompatibleProvider` | Generic | ❌ No (API) | Any OpenAI-compatible API |
| `VLLMProvider` | vLLM Server | ❌ No (API) | vLLM with OpenAI API |
| `VLLMNativeProvider` | vLLM Direct | ✅ **Yes** | Local vLLM with embeddings |
| `GemmaProvider` | Gemma JAX | ❌ No (internal) | Gemma 3n multimodal |
| `GoogleGeminiProvider` | Google API | ❌ No (internal) | Gemini multimodal |
| `TransformersProvider` | HuggingFace | ⚠️ Model-dependent | HuggingFace models |

## OpenAI-Compatible Provider

### Generic provider for any OpenAI-compatible service

**Best for:** OpenAI, vLLM servers, OpenRouter, Together AI, Anyscale

```python
from argentic.core.llm.providers.openai_compatible import OpenAICompatibleProvider

# OpenAI Official
config = {
    "base_url": "https://api.openai.com/v1",
    "api_key": "sk-...",
    "model_name": "gpt-4-vision-preview",
    "temperature": 0.7,
    "max_tokens": 2048,
}

# vLLM Server
config = {
    "base_url": "http://localhost:8000/v1",
    "api_key": "dummy",
    "model_name": "llava-hf/llava-1.5-7b-hf",
}

# OpenRouter
config = {
    "base_url": "https://openrouter.ai/api/v1",
    "api_key": "sk-or-...",
    "model_name": "anthropic/claude-3-opus",
}

llm = OpenAICompatibleProvider(config)
response = await llm.ainvoke("Hello!")
```

**Features:**
- Works with any OpenAI-compatible endpoint
- Multimodal support (images via base64)
- Streaming support
- Tool calling support

**Limitations:**
- Does NOT support pre-computed image embeddings
- Requires HTTP server

---

## vLLM Providers

### vLLMProvider (OpenAI Server Mode)

**Best for:** vLLM server already running, OpenAI API compatibility

```python
from argentic.core.llm.providers.vllm_provider import VLLMProvider

config = {
    "vllm_base_url": "http://localhost:8000/v1",
    "vllm_model_name": "llava-hf/llava-1.5-7b-hf",  # Auto-detected if omitted
    "vllm_api_key": "dummy",
    "temperature": 0.7,
    "max_tokens": 2048,
}

llm = VLLMProvider(config)
response = await llm.achat(messages)
```

**Features:**
- Auto-detects model from server
- vLLM-specific defaults
- Inherits from OpenAICompatibleProvider

**Limitations:**
- Requires vLLM server running
- No pre-computed embeddings support
- HTTP overhead

**Start vLLM Server:**
```bash
vllm serve llava-hf/llava-1.5-7b-hf --port 8000
```

---

### VLLMNativeProvider (Direct Model Loading) ⭐

**Best for:** Direct GPU access, **pre-computed image embeddings**, offline inference

```python
from argentic.core.llm.providers.vllm_native import VLLMNativeProvider

config = {
    "model_name": "llava-hf/llava-1.5-7b-hf",
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.9,
    "enable_mm_embeds": True,  # Enable embedding support!
    "temperature": 0.7,
    "max_tokens": 2048,
}

llm = VLLMNativeProvider(config)

# Works with raw images
response = await llm.achat([
    {
        "role": "user",
        "content": {
            "text": "What's in this image?",
            "images": [pil_image]
        }
    }
])

# Works with pre-computed embeddings!
response = await llm.achat([
    {
        "role": "user",
        "content": {
            "text": "What's in this image?",
            "image_embeddings": embeddings_tensor  # torch.Tensor
        }
    }
])
```

**Features:**
- ✅ **Full embedding support** with `enable_mm_embeds=True`
- Direct GPU access (no server needed)
- Lower latency (no HTTP)
- Batch inference
- Native `multi_modal_data` API

**Requirements:**
```bash
pip install vllm torch
```

**Configuration Options:**
- `model_name`: HuggingFace model ID (required)
- `tensor_parallel_size`: Number of GPUs (default: 1)
- `gpu_memory_utilization`: GPU memory fraction (default: 0.9)
- `enable_mm_embeds`: Enable embedding support (default: False)
- `trust_remote_code`: Trust remote code (default: False)

**Embedding Format:**
- `torch.Tensor`: Shape `(num_frames, feature_size, hidden_size)`
- `np.ndarray`: Auto-converted to torch tensor

---

## Using with VisualAgent

### vLLM OpenAI Server (No Embeddings)

```python
from argentic.core.agent.visual_agent import VisualAgent
from argentic.core.llm.providers.vllm_provider import VLLMProvider

llm = VLLMProvider({
    "vllm_base_url": "http://localhost:8000/v1",
    "vllm_model_name": "llava-hf/llava-1.5-7b-hf",
})

agent = VisualAgent(
    llm=llm,
    messager=messager,
    webrtc_driver=driver,
    # embedding_function=None  # Not supported
)
```

### vLLM Native with Embeddings ⭐

```python
from argentic.core.agent.visual_agent import VisualAgent
from argentic.core.llm.providers.vllm_native import VLLMNativeProvider
import torch

# Custom embedding function
async def clip_embeddings(frames):
    # Your CLIP or custom encoder
    embeddings = your_encoder.encode(frames)  # Shape: (N, feature_size, hidden_size)
    return torch.from_numpy(embeddings)

llm = VLLMNativeProvider({
    "model_name": "llava-hf/llava-1.5-7b-hf",
    "enable_mm_embeds": True,  # REQUIRED for embeddings
})

agent = VisualAgent(
    llm=llm,
    messager=messager,
    webrtc_driver=driver,
    embedding_function=clip_embeddings,  # ✅ Supported!
)
```

---

## Other Providers

### Google Gemini Provider

```python
from argentic.core.llm.providers.google_gemini import GoogleGeminiProvider

config = {
    "google_api_key": "AIza...",
    "google_model": "gemini-1.5-pro",
}

llm = GoogleGeminiProvider(config)
```

**Note:** Uses internal vision encoder, does not support pre-computed embeddings.

---

## Provider Selection Guide

### Choose **OpenAICompatibleProvider** when:
- Using OpenAI, OpenRouter, Together AI, etc.
- Need maximum compatibility
- Want to switch between services easily

### Choose **VLLMProvider** when:
- vLLM server already running
- Using vLLM with OpenAI API compatibility
- Don't need pre-computed embeddings

### Choose **VLLMNativeProvider** when:
- Need pre-computed image embeddings ⭐
- Want lowest latency (no HTTP)
- Running locally with GPU access
- Using custom vision encoders

### Choose **GemmaProvider** when:
- Using Gemma 3n multimodal models
- Want JAX/TPU optimization

### Choose **GoogleGeminiProvider** when:
- Using Google's Gemini API
- Need cloud-based inference

---

## Migration Examples

### From vLLMProvider to VLLMNativeProvider

**Before (Server Mode):**
```python
llm = VLLMProvider({
    "vllm_base_url": "http://localhost:8000/v1",
    "vllm_model_name": "llava-hf/llava-1.5-7b-hf",
})
```

**After (Native Mode with Embeddings):**
```python
llm = VLLMNativeProvider({
    "model_name": "llava-hf/llava-1.5-7b-hf",
    "enable_mm_embeds": True,  # Enable embeddings
})
```

### From OpenAI to vLLM

**Before:**
```python
llm = OpenAICompatibleProvider({
    "base_url": "https://api.openai.com/v1",
    "api_key": "sk-...",
    "model_name": "gpt-4-vision-preview",
})
```

**After:**
```python
llm = OpenAICompatibleProvider({
    "base_url": "http://localhost:8000/v1",
    "api_key": "dummy",
    "model_name": "llava-hf/llava-1.5-7b-hf",
})
```

---

## Embedding Support Summary

### ✅ Supports Pre-Computed Embeddings
- **VLLMNativeProvider** (with `enable_mm_embeds=True`)

### ❌ Does NOT Support Embeddings
- OpenAICompatibleProvider (API limitation)
- VLLMProvider (OpenAI API limitation)
- GemmaProvider (uses internal encoder)
- GoogleGeminiProvider (uses internal encoder)

### ⚠️ Model-Dependent
- TransformersProvider (check model architecture)

---

## References

- [vLLM Documentation](https://docs.vllm.ai/en/latest/features/multimodal_inputs/)
- [vLLM Image Embeddings PR](https://github.com/vllm-project/vllm/pull/6613)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Visual Agent Guide](./VISUAL_AGENT_GUIDE.md)

---

## Sources

- [Multimodal Inputs - vLLM](https://docs.vllm.ai/en/stable/features/multimodal_inputs/)
- [vLLM V1 Announcement](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
