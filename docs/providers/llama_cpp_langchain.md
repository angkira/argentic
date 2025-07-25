# LlamaCpp LangChain Provider

`LlamaCppLangchainProvider` embeds local GGUF models via the **`llama-cpp-python`** binding exposed through LangChain’s `LlamaCpp` class.

---

## What it does
* Loads a GGUF model directly in-process (CPU and/or GPU layers) using `llama_cpp.Llama` under the hood.
* Leverages LangChain community wrapper for convenient prompt handling.
* Provides both sync and async execution paths consistent with the Argentic interface.

---

## How it works
1. Reads configuration → builds a `LlamaCpp` instance with parameters (`n_ctx`, `n_gpu_layers`, etc.).
2. On first `invoke` / `chat`, the model is loaded from disk into RAM / GPU.
3. The raw string completion is wrapped in a LangChain `AIMessage`.

---

## Requirements
| Requirement | Details |
|-------------|---------|
| **Python dependency** | `llama-cpp-python` ≥ 0.2.0 *(installs GGUF runtime / BLAS)* |
| **LangChain** | `langchain-community` ≥ 0.1.x |
| **GGUF model** | Locally accessible model file. |
| **Hardware** | Sufficient RAM; optional GPU w/ Metal / CUDA / Vulkan via `n_gpu_layers`. |

Install:
```bash
pip install "llama-cpp-python>=0.2.0" langchain-community
```

> Note: On Apple Silicon use `pip install llama-cpp-python --config-settings="--set=FORCE_CMAKE=1"` for metal acceleration.

---

## Configuration
```yaml
llm:
  llama_cpp_model_path: "~/models/phi-3-mini.gguf"   # REQUIRED

  # Optional advanced parameters (mirrors LlamaCpp kwargs)
  llama_cpp_langchain_parameters:
    temperature: 0.7
    max_tokens: 256
    n_ctx: 4096
    top_p: 0.9
    n_threads: 8
    n_gpu_layers: 35
    repeat_penalty: 1.1
    use_mlock: false   # pin in RAM
    verbose: false
```

### Key fields
| Key | Description | Default |
|-----|-------------|---------|
| `llama_cpp_model_path` | Absolute/`~` path to `.gguf` | *required* |
| `temperature` | Sampling temperature | 0.7 |
| `n_gpu_layers` | #layers to offload to GPU | 0 (CPU-only) |
| `n_ctx` | Context window tokens | 2048 |

---

## Environment variables
None needed – all options supplied via config.

---

## Supported capabilities
* Local inference, zero external calls
* **No** tool calling (`supports_tools()` → `False`)
* **No** streaming wrapper exposed (LangChain LlamaCpp currently sync only)

---

## Example
```python
from argentic.core.llm.providers.llama_cpp_langchain import LlamaCppLangchainProvider

provider = LlamaCppLangchainProvider({
    "llm": {
        "llama_cpp_model_path": "~/models/phi3-mini.gguf",
        "llama_cpp_langchain_parameters": {
            "temperature": 0.6,
            "n_ctx": 4096,
            "n_gpu_layers": 20,
        }
    }
})

msg = provider.invoke("Summarise the theory of relativity in one sentence.")
print(msg.content)
``` 