# LlamaCpp Server Provider

`LlamaCppServerProvider` communicates with a **llama.cpp HTTP server** (OpenAI-compatible) or optionally starts it locally.

---

## What it does
* Sends JSON requests to `/completion` (simple prompt) or `/v1/chat/completions` (OpenAI-style chat) endpoints exposed by `llama.cpp --server` or the standalone `llama.cpp/server` binary.
* Can **auto-start** the server process if provided with a binary path and args.
* Provides a fallback to LangChain local `LlamaCpp` if `use_langchain` flag is enabled (same behaviour as LangChain provider).

---

## How it works
1. If `llama_cpp_server_auto_start` is **true**, a subprocess is spawned with custom args and `Popen` is stored for later shutdown.
2. Requests are executed asynchronously with `httpx.AsyncClient` (sync variants use `asyncio.run`).
3. Responses are parsed and wrapped as `AIMessage`.

---

## Requirements
| Requirement | Details |
|-------------|---------|
| **llama.cpp server** | Either: <br> • Run `./server -m model.gguf [flags]` manually, **or** <br> • Provide `llama_cpp_server_binary` + `llama_cpp_server_args` to auto-start. |
| **Model** | GGUF model file reachable by the server. |
| **Python deps** | `httpx` (already required by core) |
| **Network** | HTTP access to `http://{host}:{port}` (default `127.0.0.1:8080`). |

---

## Configuration
```yaml
llm:
  # Connection
  llama_cpp_server_host: 127.0.0.1   # default
  llama_cpp_server_port: 8080        # default

  # Auto-start (optional)
  llama_cpp_server_auto_start: true
  llama_cpp_server_binary: "~/bin/llama-server"
  llama_cpp_server_args:
    - "-m" ~/models/llama3.gguf
    - "--threads" 8

  # Request parameters passed on every call
  llama_cpp_server_parameters:
    temperature: 0.7
    n_predict: 256
    stop: ["</s>"]

  # LangChain local fallback (optional)
  llama_cpp_use_langchain: false      # true → use in-proc instead of HTTP
  llama_cpp_model_path: "~/models/llama3.gguf"  # required if above true
```

### Common server CLI flags
| Flag | Meaning |
|------|---------|
| `-m` | Path to GGUF model |
| `--ctx-size` | Context window |
| `--host` / `--port` | Network address |
| `--threads` | CPU threads |
| `--n-gpu-layers` | GPU offload layers |

---

## Environment variables
None needed, but you may choose to set `LLAMA_SERVER_HOST`, etc., and map them into config yourself.

---

## Supported capabilities
* Local HTTP or remote server
* **No** tool calling (`supports_tools()` → `False`)
* **No** streaming (SSE available but not implemented)

---

## Example
```python
from argentic.core.llm.providers.llama_cpp_server import LlamaCppServerProvider

provider = LlamaCppServerProvider({
    "llm": {
        "llama_cpp_server_host": "localhost",
        "llama_cpp_server_port": 8080,
        "llama_cpp_server_auto_start": False,
    }
})

msg = provider.invoke("Write a haiku about snow.")
print(msg.content)
``` 