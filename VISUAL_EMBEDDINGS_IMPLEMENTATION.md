# Visual Embeddings Support Implementation

## Summary

Successfully implemented support for visual embeddings in the VisualAgent, allowing users to provide custom embedding functions to pre-process visual data before passing it to LLM providers.

## Key Changes

### 1. VisualAgent Enhancement (`src/argentic/core/agent/visual_agent.py`)

**Added:**
- `embedding_function` parameter: Optional async callable that converts frames to embeddings
- Automatic embedding processing in `_process_visual_input()`
- Error handling with fallback to raw frames if embedding function fails
- Support for both numpy array and dict return types from embedding functions

**Removed:**
- Visual encoder class approach (too framework-specific)
- Moved to simple callback function pattern for flexibility

**Example Usage:**
```python
async def my_embedding_fn(frames: List[np.ndarray]) -> np.ndarray:
    # Your custom encoding logic
    return embeddings

agent = VisualAgent(
    llm=llm,
    messager=messager,
    webrtc_driver=driver,
    embedding_function=my_embedding_fn  # Optional
)
```

### 2. Provider Compatibility Checks

#### vLLM Provider (`src/argentic/core/llm/providers/vllm_provider.py`)
- **Status**: ❌ Not supported via OpenAI-compatible API
- **Raises**: `NotImplementedError` with helpful message
- **Note**: Native vLLM Python API supports embeddings via `multi_modal_data`, but requires custom provider implementation

#### Gemma JAX Provider (`src/argentic/core/llm/providers/gemma_jax.py`)
- **Status**: ❌ Not supported
- **Raises**: `NotImplementedError`
- **Reason**: Gemma 3n uses internal vision encoder, requires raw images

#### Google Gemini Provider (`src/argentic/core/llm/providers/google_gemini.py`)
- **Status**: ❌ Not supported
- **Raises**: `NotImplementedError`
- **Reason**: Gemini API uses internal vision encoder, requires raw images

#### Transformers Provider (`src/argentic/core/llm/providers/transformers_provider.py`)
- **Status**: ⚠️ Model-dependent
- **Warning**: Logs warning but allows attempt
- **Reason**: Some custom HuggingFace models might support it, most don't

### 3. Documentation Updates (`docs/VISUAL_AGENT_GUIDE.md`)

**Updated sections:**
- Visual Processing Modes: Clarified embedding function approach
- Architecture diagrams: Updated to show embedding function flow
- Configuration examples: Removed encoder-specific config (done in Python)
- Usage examples:
  - Example 5: CLIP embedding function
  - Example 6: Error handling and metadata
- API Reference: Added embedding function signature and best practices
- Provider compatibility matrix

### 4. Examples

**Created:**
- `examples/visual_agent_embedding_example.py`: Demonstrates embedding function usage with dummy provider

**Key Features Demonstrated:**
- Simple embedding function
- Embedding function with metadata
- Dummy provider for testing
- Both modes (with/without embeddings)

## Design Decisions

### Why Callback Function Instead of Encoder Class?

1. **Simplicity**: Users don't need to inherit from base classes or implement multiple methods
2. **Flexibility**: Any async function works - no framework lock-in
3. **Compatibility**: Easier to integrate existing encoding code
4. **Clarity**: Clear separation between framework code and user code

### Why Not Include Encoders in Framework?

1. **Scope**: Vision encoders are complex and model-specific
2. **Dependencies**: Would require torch/tensorflow/jax dependencies
3. **Maintenance**: Hard to maintain multiple encoder implementations
4. **User Choice**: Users should choose their own encoding approach

### Error Handling Strategy

- Embedding function errors trigger fallback to raw frames
- Providers check for `image_embeddings` key and raise clear exceptions
- Warnings in logs help users debug issues

## Provider Support Summary

| Provider | Embeddings Support | Status | Action |
|----------|-------------------|--------|--------|
| vLLM (OpenAI API) | ❌ No | NotImplementedError | Use raw frames or implement native vLLM provider |
| Gemma JAX | ❌ No | NotImplementedError | Use raw frames (Gemma has internal encoder) |
| Google Gemini | ❌ No | NotImplementedError | Use raw frames (Gemini has internal encoder) |
| Transformers | ⚠️ Maybe | Warning logged | Model-dependent, usually fails |
| Custom Provider | ✅ Yes | User implements | Full control via custom provider |

## Migration Guide

### Before (Old Encoder Approach - Removed)
```python
# This pattern was removed
from argentic.core.encoders import VisualEncoder

class MyEncoder(VisualEncoder):
    # Complex class implementation...
    pass

agent = VisualAgent(..., visual_encoder=MyEncoder(), use_embeddings=True)
```

### After (New Callback Approach)
```python
# Simple async function
async def my_embedding_fn(frames):
    # Your encoding logic
    return embeddings

agent = VisualAgent(..., embedding_function=my_embedding_fn)
```

## Testing

### Validated:
- ✅ Syntax check on all modified files
- ✅ Import structure (no circular dependencies)
- ✅ Example code runs without errors
- ✅ Documentation examples are consistent with implementation

### To Test (by user):
- Integration with actual LLM providers
- WebRTC video stream processing
- Real embedding models (CLIP, ViT, etc.)
- Performance under load

## Future Enhancements

Potential improvements for future releases:

1. **Native vLLM Provider**: Implement provider using vLLM's Python API with `multi_modal_data` support
2. **Embedding Caching**: Cache embeddings for repeated frames
3. **Batch Processing**: Allow embedding functions to request batch processing
4. **Performance Metrics**: Track embedding time and throughput
5. **Example Encoders**: Provide reference implementations (CLIP, ViT) in examples/

## References

- [vLLM Image Embeddings PR](https://github.com/vllm-project/vllm/pull/6613)
- [vLLM Encoder Disaggregation](https://blog.vllm.ai/2025/12/15/vllm-epd.html)
- VisualAgent Guide: `docs/VISUAL_AGENT_GUIDE.md`
- Example Implementation: `examples/visual_agent_embedding_example.py`

## Files Modified

```
src/argentic/core/agent/visual_agent.py
src/argentic/core/llm/providers/vllm_provider.py
src/argentic/core/llm/providers/gemma_jax.py
src/argentic/core/llm/providers/google_gemini.py
src/argentic/core/llm/providers/transformers_provider.py
docs/VISUAL_AGENT_GUIDE.md
examples/visual_agent_embedding_example.py
```

## Implementation Complete ✅

All requested features have been implemented:
- ✅ Embedding function support in VisualAgent
- ✅ Provider compatibility checks
- ✅ Clear error messages for unsupported providers
- ✅ Comprehensive documentation
- ✅ Working examples
- ✅ No framework-specific encoder classes (user provides own)
