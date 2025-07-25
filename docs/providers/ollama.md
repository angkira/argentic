# Ollama Provider

`OllamaProvider` connects to a locally running [Ollama](https://github.com/jmorganca/ollama) server, giving access to a variety of pre-quantised GGUF models via simple HTTP.

---

## What it does
* Wraps either `langchain_ollama.ChatOllama` (chat) or `langchain_ollama.OllamaLLM` (completion) depending on `ollama_use_chat_model` flag.
* Supports sync/async (`invoke` / `ainvoke`) and chat methods.
* Adds lightweight retry logic using Tenacity (3 attempts, exponential back-off).

---

## How it works
1. Builds a `ChatOllama` or `OllamaLLM` object with host URL, model name and sampling parameters.
2. For chat, converts Argentic message list → LangChain messages.
3. Sends request to `http://{base_url}/api/generate` (under the hood) and returns `AIMessage`.

---

## Requirements
| Requirement | Details |
|-------------|---------|
| **Ollama daemon** | Install and run `ollama serve` (default `localhost:11434`). |
| **Model pulled** | Run `ollama pull <model>` beforehand or rely on automatic pull. |
| **Python deps** | `langchain-ollama` |
| **Network** | HTTP access to the daemon URL. |

Install:
```bash
brew install ollama     # macOS
ollama serve &          # starts the server
pip install langchain-ollama
```

---

## Configuration
```yaml
llm:
  ollama_model_name: "gemma:7b"           # REQUIRED – any model available in Ollama hub
  ollama_base_url: "http://localhost:11434"  # default
  ollama_use_chat_model: true             # false → use completion endpoint

  # Optional sampling / system parameters
  ollama_parameters:
    temperature: 0.7
    top_p: 0.9
    top_k: 50
    num_predict: 256
    repeat_penalty: 1.1
    num_ctx: 4096
```

### Frequently used parameters
See [Ollama docs](https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters).

---

## Environment variables
None required – configure URL / model via YAML.

---

## Supported capabilities
* Local HTTP inference
* Streaming supported (`supports_streaming()` → `True`)
* **No** function/tool calling (`supports_tools()` → `False`)

---

## Example
```python
from argentic.core.llm.providers.ollama import OllamaProvider

provider = OllamaProvider({
    "llm": {
        "ollama_model_name": "llama3:8b",
        "ollama_use_chat_model": True,
    }
})

resp = provider.chat([
    {"role": "user", "content": "Hello"},
])
print(resp.content)
``` 