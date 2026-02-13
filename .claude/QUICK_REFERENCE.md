# Argentic Quick Reference

## Common Code Patterns

### Creating an Agent
```python
from argentic.core.agent import Agent
from argentic.core.llm import LLMFactory
from argentic.core.messager import Messager

llm = LLMFactory.create_provider(config["llm"])
messager = Messager(protocol="mqtt", broker_address="localhost", port=1883)
await messager.connect()

agent = Agent(llm=llm, messager=messager, role="assistant")
await agent.async_init()
response = await agent.query("Your question here")
```

### Creating a Tool
```python
from argentic.core.tools.tool_base import BaseTool
from pydantic import BaseModel

class MyToolArgs(BaseModel):
    input: str
    count: int = 1

class MyTool(BaseTool):
    def __init__(self, messager):
        super().__init__(
            name="my_tool",
            manual="Human-readable description",
            api="LLM-readable description with examples",
            argument_schema=MyToolArgs,
            messager=messager,
        )

    async def _execute(self, arguments):
        return {"result": f"Processed: {arguments['input']}"}

# Usage
tool = MyTool(messager)
await tool.register(reg_topic, status_topic, call_base, resp_base)
```

### Publishing Messages
```python
from argentic.core.protocol.message import AskQuestionMessage

msg = AskQuestionMessage(question="What is AI?", user_id="user123")
await messager.publish("agent/command/ask_question", msg)
```

### Subscribing to Messages
```python
async def handle_answer(message: AnswerMessage):
    print(f"Answer: {message.answer}")

await messager.subscribe(
    topic="agent/response/answer",
    handler=handle_answer,
    message_cls=AnswerMessage,
)
```

---

## Message Types Quick Reference

| Message Type | Purpose | Key Fields |
|--------------|---------|------------|
| `AskQuestionMessage` | User query | `question`, `user_id` |
| `AnswerMessage` | Agent response | `answer`, `user_id` |
| `TaskMessage` | Tool execution | `tool_id`, `arguments`, `task_id` |
| `TaskResultMessage` | Tool success | `task_id`, `result` |
| `TaskErrorMessage` | Tool failure | `task_id`, `error`, `traceback` |
| `RegisterToolMessage` | Tool registration | `tool_name`, `tool_manual`, `tool_api` |
| `ToolRegisteredMessage` | Registration confirm | `tool_id` |

---

## Configuration Quick Reference

### LLM Providers

**Ollama**:
```yaml
llm:
  provider: ollama
  ollama_model_name: "gemma2:12b-it"
  ollama_base_url: "http://localhost:11434"
  ollama_parameters:
    temperature: 0.7
```

**Google Gemini**:
```yaml
llm:
  provider: google_gemini
  google_gemini_model_name: "gemini-2.0-flash-exp"
  google_gemini_api_key: "${GOOGLE_GEMINI_API_KEY}"
  google_gemini_parameters:
    temperature: 0.7
```

**Llama.cpp Server**:
```yaml
llm:
  provider: llama_cpp_server
  llama_cpp_server_host: "localhost"
  llama_cpp_server_port: 8080
```

### Messaging Protocols

**MQTT** (default):
```yaml
messaging:
  protocol: mqtt
  broker_address: localhost
  port: 1883
  keepalive: 60
```

**Kafka**:
```yaml
messaging:
  protocol: kafka
  broker_address: localhost
  port: 9092
```

**Redis**:
```yaml
messaging:
  protocol: redis
  broker_address: localhost
  port: 6379
```

---

## CLI Commands

```bash
# Run main agent
python -m argentic.main

# Run RAG tool service
python -m argentic.services.rag_tool_service

# Run tests
pytest tests/
pytest tests/unit/ -v

# Run specific test
pytest tests/unit/test_agent.py::test_agent_query -v

# Format code
black src/ tests/ --line-length 100

# Type check
mypy src/

# Build package
uv run python -m build

# Install locally
uv pip install -e .

# Install with extras
uv pip install -e ".[dev,docs]"
```

---

## Testing Patterns

### Basic Test
```python
import pytest
from unittest.mock import AsyncMock, MagicMock

@pytest.mark.asyncio
async def test_example():
    messager = MagicMock(spec=Messager)
    messager.connect = AsyncMock()
    messager.publish = AsyncMock()

    await messager.connect()
    messager.connect.assert_called_once()
```

### Mock LLM
```python
from argentic.core.llm.providers.mock import MockLLMProvider, MockResponse, MockResponseType

mock_llm = MockLLMProvider({})
mock_llm.set_responses([
    MockResponse(MockResponseType.DIRECT, content='{"content": "Test answer"}'),
])
```

---

## Debugging

### Enable Verbose Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### View Dialogue History
```python
agent = Agent(...)
# After queries
for entry in agent.dialogue_history:
    print(entry)
```

### Check Tool Registry
```python
tool_manager = ToolManager(messager)
await tool_manager.async_init()
descriptions = tool_manager.get_tool_descriptions()
print(descriptions)
```

---

## Common Issues

### Agent Not Responding
- Check messager connection: `await messager.connect()`
- Verify agent initialized: `await agent.async_init()`
- Check topic subscriptions match config

### Tool Not Found
- Ensure tool registered before agent query
- Check tool_manager has the tool: `tool_manager.tools_by_name`
- Verify registration topic matches

### Timeout Errors
- Increase tool execution timeout: `execute_tool(..., timeout=60)`
- Check tool service is running
- Verify message broker is accessible

### LLM Connection Failed
- Ollama: Check service running on port 11434
- Gemini: Verify API key in environment
- Check network connectivity

---

## Performance Tips

1. **Use STATELESS mode** for high-throughput scenarios
2. **Limit max_query_history_items** to prevent memory growth
3. **Set appropriate timeouts** on tool execution
4. **Use connection pooling** for message brokers
5. **Enable dialogue pruning** with context limits
6. **Mock LLM in tests** to avoid API costs

---

## Architecture Quick View

```
┌─────────────┐
│    User     │
└──────┬──────┘
       │ AskQuestionMessage
       v
┌─────────────────────────────────┐
│         Agent                   │
│  ┌──────────────────────────┐   │
│  │  LLM Provider            │   │
│  │  (Ollama/Gemini/etc)     │   │
│  └──────────────────────────┘   │
│  ┌──────────────────────────┐   │
│  │  Tool Manager            │   │
│  │  - Registry              │   │
│  │  - Execution             │   │
│  └──────────────────────────┘   │
└───────────┬─────────────────────┘
            │ TaskMessage
            v
    ┌──────────────┐
    │   Messager   │ ← MQTT/Kafka/Redis/RabbitMQ
    └──────┬───────┘
           │
           v
    ┌──────────────┐
    │  Tool (1..n) │
    │  - execute   │
    └──────────────┘
```

---

## Useful Links

- **Main Docs**: See `CLAUDE.md` in root
- **Examples**: `examples/` directory
- **Tests**: `tests/` directory
- **PyPI**: https://pypi.org/project/argentic/
