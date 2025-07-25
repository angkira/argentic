# LlamaCpp CLI Provider

`LlamaCppCLIProvider` integrates [llama.cpp](https://github.com/ggerganov/llama.cpp) **command-line interface** with Argentic’s unified provider API.

---

## What it does
* Spawns the compiled `main` (or `llama`) binary locally and feeds a prompt via the **`-p`** flag.
* Parses stdout and wraps the completion into a LangChain `AIMessage`.
* Supports synchronous (`invoke`, `chat`) and asynchronous (`ainvoke`, `achat`) calls; async version runs the process with `asyncio.create_subprocess_exec`.

---

## How it works (flow)
1. Builds the CLI command using required paths and optional parameters.
2. Executes with `subprocess.run` (sync) or `asyncio` (async), capturing output.
3. Removes the echoed prompt from the beginning of the response to return only the generation.
4. Converts text → `AIMessage` for downstream consumption.

---

## Requirements
| Requirement | Details |
|-------------|---------|
| **llama.cpp binary** | You must compile `llama.cpp` (`make`) or download a pre-built release. Pass its absolute path via config. |
| **GGUF model file** | Local `.gguf` model (e.g., `llama-3-8b-instruct.Q4_K_M.gguf`). |
| **Python deps** | None beyond Argentic defaults (does **not** need `llama-cpp-python`). |
| **OS** | Any POSIX system capable of running the binary (Linux/macOS/WSL). |

---

## Configuration
```yaml
llm:
  # REQUIRED
  llama_cpp_cli_binary: "/path/to/llama/main"
  llama_cpp_cli_model_path: "~/models/llama3.gguf"

  # Optional command-line args appended verbatim
  llama_cpp_cli_args:
    - "--ctx-size" 2048
    - "--threads" 8

  # High-level convenience parameters (mapped to CLI flags)
  llama_cpp_cli_parameters:
    temperature: 0.7          # → --temp
    top_k: 40                # → --top-k
    top_p: 0.95              # → --top-p
    n_predict: 256           # → --n-predict
    n_gpu_layers: 35         # → --n-gpu-layers
    repeat_penalty: 1.1
```

### Frequently used parameters
| Key | CLI flag | Default |
|-----|----------|---------|
| `temperature` | `--temp` | llama.cpp default (0.8) |
| `top_p` | `--top-p` | 0.95 |
| `top_k` | `--top-k` | 40 |
| `n_predict` | `--n-predict` | 128 |
| `ctx_size` | `--ctx-size` | 2048 |

---

## Environment variables
None – all paths/flags are supplied via config.

---

## Supported capabilities
* Local inference (no network)
* **No** tool/function calling (`supports_tools()` → `False`)
* **No** streaming wrapper exposed (CLI can stream but not surfaced)

---

## Example
```python
from argentic.core.llm.providers.llama_cpp_cli import LlamaCppCLIProvider

provider = LlamaCppCLIProvider({
    "llm": {
        "llama_cpp_cli_binary": "~/bin/llama",
        "llama_cpp_cli_model_path": "~/models/Llama-3-8B-Q4.gguf",
        "llama_cpp_cli_parameters": {
            "temperature": 0.7,
            "n_predict": 200,
        }
    }
})

answer = provider.invoke("Explain quantum entanglement in 2 lines.")
print(answer.content)
``` 