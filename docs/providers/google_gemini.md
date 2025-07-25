# Google Gemini Provider

GoogleGeminiProvider offers access to Google’s **Gemini** family of chat-completion models through [LangChain-Google-GenAI](https://python.langchain.com/docs/integrations/providers/google_genai/).

---

## What it does
* Wraps `ChatGoogleGenerativeAI` (LangChain) and exposes a unified Argentic provider interface (`invoke`, `ainvoke`, `chat`, `achat`).
* Adds robust retry, circuit-breaker and error-classification logic using **`google-api-core`** exceptions and **Tenacity** back-off patterns.
* Transparently patches the `finish_reason` enum bug present in some Gemini responses (see source code for details).

---

## How it works (high-level flow)
1. Incoming prompts/messages are converted to LangChain `BaseMessage` sequence.
2. If a **`ToolMessage`** is encountered as the last turn, it is automatically transformed into a user-visible tool-result message to satisfy Gemini request requirements.
3. The provider builds a `ChatGoogleGenerativeAI` instance on first construction and optionally binds LangChain tools for function calling.
4. All requests run through a Tenacity decorator implementing exponential back-off + jitter and Google-recommended retryable errors.
5. Detailed error information is logged and mapped to Google’s typed exceptions.

---

## Requirements
| Requirement | Minimum Version | Notes |
|-------------|-----------------|-------|
| Python deps | `langchain-google-genai` ≥ 0.1.0 <br> `google-api-core` (indirect) <br> `tenacity` | Install via `pip install langchain-google-genai tenacity` (the rest are pulled automatically) |
| Google Cloud | Gemini API enabled in your project | Billing must be active |
| Network | Outbound HTTPS to `generativelanguage.googleapis.com` |

---

## Configuration
All configuration lives under the **`llm`** section of your Argentic config dictionary.

```yaml
llm:
  # REQUIRED
  google_gemini_api_key: "YOUR_API_KEY"  # can be provided via env var instead

  # Optional model selection
  google_gemini_model_name: "gemini-1.5-flash"  # default

  # Retry / resilience
  retry_config:
    max_retries: 3           # default 3
    initial_wait: 1.0        # seconds
    max_wait: 60.0
    enable_jitter: true
    circuit_breaker_threshold: 5   # errors within window
    circuit_breaker_window: 300    # seconds
    request_timeout: 60            # per-request timeout

  # Model parameters (forwarded verbatim)
  google_gemini_parameters:
    temperature: 0.7
    max_output_tokens: 1024
    top_k: 40
    top_p: 0.95
```

### Flat-key aliases (supported for convenience)
* `google_gemini_api_key` *(string)* – API key.
* `google_gemini_model_name` *(string)* – model version.
* `enable_google_search` *(bool)* – reserved for future Google Search integration.

---

## Environment variables
| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` | Primary place to put your key (preferred) |
| `GOOGLE_GEMINI_API_KEY` | Alternative variable (legacy) |

If the key is set in either env variable the explicit config field may be omitted.

---

## Supported capabilities
* Chat & single-prompt (`invoke` / `chat`)
* Streaming (`supports_streaming()` → `True`)
* Function/tool calling (`supports_tools()` → `True`)

---

## Example
```python
from argentic.core.llm.providers.google_gemini import GoogleGeminiProvider

provider = GoogleGeminiProvider({
    "llm": {
        "google_gemini_api_key": "$GEMINI_API_KEY",
        "google_gemini_model_name": "gemini-1.5-flash",
    }
})

reply = provider.invoke("Hello Gemini!")
print(reply.content)
``` 