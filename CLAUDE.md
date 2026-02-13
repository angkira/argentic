# Argentic - AI Agent Framework

**Lightweight async microframework for building local AI agents with messaging-based architecture**

## Quick Overview

**What it is**: Multi-agent orchestration framework with LLM-agnostic design, tool system, and pure message-passing architecture.

**Core Pattern**: Agents communicate via async messaging (MQTT/Kafka/Redis/RabbitMQ/ZeroMQ), execute tools, and coordinate through supervisor pattern.

**Tech Stack**: Python 3.11+, asyncio, Pydantic, multiple LLM providers (Ollama, Llama.cpp, Gemini)

---

## Architecture

### Core Components

```
src/argentic/
├── core/
│   ├── agent/           # Agent class - LLM interaction, tool orchestration (1763 lines)
│   ├── llm/             # LLM abstraction (Ollama, Llama.cpp, Gemini, Mock)
│   ├── messager/        # Unified messaging (MQTT, Kafka, Redis, RabbitMQ, ZeroMQ)
│   ├── tools/           # Tool management (BaseTool, ToolManager)
│   ├── protocol/        # Message types (Pydantic models)
│   └── graph/           # Multi-agent coordination (Supervisor)
├── tools/               # Tool implementations (RAG, Environment, Google)
├── services/            # Runnable services (rag_tool_service, etc.)
└── cli_client.py        # CLI for interacting with agents
```

### Agent Architecture

```
Agent (core orchestrator)
├── LLM Provider       # Pluggable (Ollama/Llama.cpp/Gemini)
├── Tool Manager       # Tool registration & execution
├── Messager           # Message broker interface
├── State              # STATEFUL (history) or STATELESS
└── Supervisor         # Multi-agent coordinator (optional)
```

### Message Flow

```
User → AskQuestionMessage → Agent → LLM → Tool Calls → ToolManager
                                    ↓
Tool executes → TaskResultMessage → Agent → LLM → AnswerMessage → User
```

---

## Key Design Patterns

| Pattern | Where | Purpose |
|---------|-------|---------|
| **Factory** | `LLMFactory` | Instantiate LLM providers |
| **Strategy** | `Messager` drivers | Pluggable protocols |
| **Observer** | Agent handlers | Async message subscriptions |
| **Template Method** | `BaseTool` | Standard tool lifecycle |
| **Registry** | `ToolManager` | Track registered tools |
| **Event-driven** | asyncio + Futures | Non-blocking I/O |

---

## Code Conventions

### Naming

- **Classes**: `PascalCase` - `Agent`, `BaseTool`, `ModelProvider`
- **Functions**: `snake_case` - `async_init()`, `handle_task()`
- **Private**: `_method_name()` with single underscore
- **Messages**: `{Context}Message` - `AskQuestionMessage`, `TaskResultMessage`
- **Providers**: `{Name}Provider` - `OllamaProvider`, `GoogleGeminiProvider`

### Type Hints

**REQUIRED** on all functions:
```python
async def query(
    self, question: str, user_id: Optional[str] = None
) -> str:
    """Process question through LLM and tools."""
```

- Use `Optional[T]` for nullable values
- Use `Union[A, B]` for multiple types
- Use `Literal["x", "y"]` for restricted strings
- Always include `-> ReturnType` (even `-> None`)

### Async Patterns

**All I/O must be async**:
```python
await messager.publish(topic, message)
await messager.subscribe(topic, handler, MessageClass)
await agent.query("question")
```

**Thread pool for blocking operations**:
```python
loop = asyncio.get_running_loop()
result = await loop.run_in_executor(self._executor, blocking_func)
```

---

## Message Protocol

### Message Hierarchy

```
BaseMessage[T] (Pydantic)
├── type: str               # Message type discriminator
├── source: MessageSource   # USER, AGENT, SYSTEM, TOOL, LLM
├── timestamp: float        # Auto-set
├── message_id: UUID        # Auto-generated
└── data: Optional[T]       # Generic payload
```

### Key Message Types

**Task Execution**:
- `TaskMessage` - Tool execution request
- `TaskResultMessage` - Success result
- `TaskErrorMessage` - Execution error

**Tool Registration**:
- `RegisterToolMessage` - Tool announces itself
- `ToolRegisteredMessage` - Confirmation with tool_id

**Chat Protocol**:
- `SystemMessage(role="system")` - System prompts
- `UserMessage(role="user")` - User inputs
- `AssistantMessage(role="assistant")` - LLM responses
- `ToolMessage(role="tool")` - Tool results

**Public API**:
- `AskQuestionMessage` - User query
- `AnswerMessage` - Agent response

---

## Tool Development

### Tool Pattern

```python
class MyTool(BaseTool):
    """Custom tool implementation."""

    def __init__(self, messager: Messager):
        # Define Pydantic schema for validation
        class ArgsSchema(BaseModel):
            query: str
            limit: int = 10

        super().__init__(
            name="my_tool",
            manual="Tool description for humans",
            api="Tool description for LLM with examples",
            argument_schema=ArgsSchema,
            messager=messager,
        )

    async def _execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool logic. Arguments validated by Pydantic."""
        query = arguments["query"]
        # Your logic here
        return {"result": "processed"}
```

### Tool Lifecycle

1. **Register**: `await tool.register(reg_topic, status_topic, call_base, resp_base)`
2. **Receive Tasks**: Auto-subscribed to task topic after registration
3. **Execute**: `_execute()` called with validated arguments
4. **Return Result**: Published as `TaskResultMessage`
5. **Unregister**: `await tool.unregister()`

---

## Agent Development

### Basic Agent Setup

```python
# 1. Create LLM provider
llm = LLMFactory.create_provider(config["llm"])

# 2. Create messager
messager = Messager(
    protocol="mqtt",
    broker_address="localhost",
    port=1883,
)
await messager.connect()

# 3. Create agent
agent = Agent(
    llm=llm,
    messager=messager,
    role="researcher",
    system_prompt="You are a research assistant.",
    state_mode=AgentStateMode.STATEFUL,  # or STATELESS
)
await agent.async_init()

# 4. Query
response = await agent.query("What is the capital of France?")
```

### Agent State Modes

- **STATEFUL** (default): Maintains conversation history
- **STATELESS**: Each query is isolated, no history

### Multi-Agent with Supervisor

```python
supervisor = Supervisor(llm=llm, messager=messager, role="supervisor")
researcher = Agent(llm=llm, messager=messager, role="researcher")

await supervisor.assign_task("Analyze market trends", to_agent="researcher")
```

---

## Configuration

### config.yaml Structure

```yaml
llm:
  provider: ollama  # ollama | llama_cpp_server | llama_cpp_cli | google_gemini
  ollama_model_name: "gemma2:12b-it"
  ollama_base_url: "http://localhost:11434"
  ollama_parameters:
    temperature: 0.7
    top_p: 0.95

messaging:
  protocol: mqtt  # mqtt | kafka | redis | rabbitmq | zeromq
  broker_address: localhost
  port: 1883
  keepalive: 60

topics:
  commands:
    ask_question: agent/command/ask_question
  responses:
    answer: agent/response/answer
  tools:
    register: agent/tools/register
    call: agent/tools/call
```

### Environment Variables

- `GOOGLE_GEMINI_API_KEY` - Gemini API key
- `NO_COLOR` - Disable colored logging

### ZeroMQ Driver

**High-performance brokerless messaging for local multi-agent scenarios**

**Architecture**: Unlike centralized brokers (MQTT, Redis, Kafka), ZeroMQ uses an XPUB/XSUB proxy pattern for peer-to-peer communication.

**Proxy Modes**:
- **Embedded** (default): Driver auto-starts proxy in background thread
- **External**: Connect to user-managed proxy process

**Configuration Example**:
```yaml
messaging:
  protocol: zeromq
  broker_address: 127.0.0.1
  port: 5555              # Frontend (XSUB - publishers)
  backend_port: 5556      # Backend (XPUB - subscribers)
  start_proxy: true       # Auto-start embedded proxy
  proxy_mode: embedded    # embedded | external
  high_water_mark: 1000   # Message queue limit
  linger: 1000            # Socket close wait time (ms)
  connect_timeout: 5000   # Connection timeout (ms)
```

**Performance Characteristics**:
- **Latency**: ~50-100μs (10-20x faster than MQTT)
- **Throughput**: 1M+ messages/sec (5-10x faster than Redis)
- **Memory**: Low, bounded by `high_water_mark`
- **Use Case**: High-frequency local agent communication

**Limitations**:
- ❌ No message persistence (fire-and-forget)
- ❌ No QoS levels (`qos` parameter ignored)
- ❌ No retention (`retain` parameter ignored)
- ❌ Topics cannot contain spaces (wire format: `"<topic> <json>"`)
- ❌ Default config is localhost-only (not distributed by default)

**When to Use**:
- ✅ Local development with low-latency requirements
- ✅ High-frequency agent-to-agent communication
- ✅ Benchmarking and performance testing
- ❌ Production systems requiring persistence or QoS
- ❌ Distributed deployments (use MQTT/Kafka instead)

**Installation**:
```bash
pip install argentic[zeromq]
```

---

## Testing

### Test Structure

```
tests/
├── unit/              # Component isolation
├── core/messager/     # Messager tests (unit, integration, e2e)
├── integration/       # Multi-component workflows
└── conftest.py        # Global fixtures
```

### Test Patterns

```python
@pytest.mark.asyncio
async def test_agent_query(mock_llm, mock_messager):
    agent = Agent(llm=mock_llm, messager=mock_messager)
    await agent.async_init()

    mock_llm.set_responses([
        MockResponse(MockResponseType.DIRECT, content='{"content": "Answer"}')
    ])

    result = await agent.query("Question?")
    assert "Answer" in result
```

### Mocking

- Use `MockLLMProvider` for LLM testing
- Use `AsyncMock` for async methods
- Use `MagicMock(spec=Class)` for interfaces

---

## Important Files

### Core Framework
- `src/argentic/core/agent/agent.py` (1763 lines) - Main agent logic
- `src/argentic/core/tools/tool_manager.py` (467 lines) - Tool orchestration
- `src/argentic/core/messager/messager.py` (308 lines) - Unified messaging

### Examples
- `examples/single_agent_example.py` - Single agent usage
- `examples/multi_agent_example.py` - Supervisor pattern
- `examples/email_tool.py` - Custom tool example

### Configuration
- `config.yaml` - Main configuration
- `.env` - API keys and secrets
- `pyproject.toml` - Dependencies and project metadata

---

## Development Guidelines

### When Adding Features

1. **New Tool**: Subclass `BaseTool`, implement `_execute()`
2. **New Provider**: Implement `ModelProvider` interface
3. **New Message**: Subclass `BaseMessage`, add to `protocol/`
4. **Tests Required**: All new features need unit tests

### Code Quality

- ✅ Type hints on all functions
- ✅ Async for all I/O operations
- ✅ Pydantic for data validation
- ✅ Docstrings on public methods
- ✅ Error handling with proper logging

### Common Patterns

**Graceful Shutdown**:
```python
async def shutdown_handler():
    if tool:
        await tool.unregister()
    if messager:
        await messager.stop()
```

**Message Publishing**:
```python
message = AskQuestionMessage(question="...", user_id="...")
await messager.publish(topic, message)
```

**Message Subscription**:
```python
await messager.subscribe(
    topic="agent/response/answer",
    handler=handle_answer,
    message_cls=AnswerMessage,
)
```

---

## Recent Changes

- **v0.15.0** (2026-02-13): Trusted publishing, sentence-transformers 5.2.2
- **v0.14.0**: Dependency upgrade for transformers 5.x compatibility
- **v0.13.0** (2025-12): Auto git tags, auto-publish workflow
- **v0.12.0** (2025-10): Removed LangChain, pure messaging supervisor

---

## Key Architectural Decisions

1. **Messaging-First**: All communication via messages, not direct calls
2. **Async Everywhere**: All I/O non-blocking for high concurrency
3. **Configuration-Driven**: Behavior controlled via YAML
4. **Pluggable Providers**: LLM, messaging, storage are abstractions
5. **Type-Safe Messages**: Pydantic models for validation
6. **No External Orchestration**: Pure messaging-based coordination

---

## Common Tasks

### Running an Agent
```bash
python -m argentic.main
```

### Running a Tool Service
```bash
python -m argentic.services.rag_tool_service
```

### Running Tests
```bash
pytest tests/
pytest tests/unit/test_agent.py -v
```

### Building Package
```bash
uv run python -m build
```

### Publishing to PyPI
```bash
git push origin main  # Auto-publish via GitHub Actions
```

---

## Links

- **PyPI**: https://pypi.org/project/argentic/
- **Docs**: https://angkira.github.io/argentic/
- **GitHub**: https://github.com/angkira/argentic/

---

## For AI Agents

### What to Preserve

- ✅ Async patterns throughout
- ✅ Pydantic validation on all messages
- ✅ Type hints on all functions
- ✅ Message-based communication (no direct calls)
- ✅ Graceful shutdown handlers
- ✅ Error logging and handling

### What to Avoid

- ❌ Blocking I/O operations (use async)
- ❌ Direct component coupling (use messages)
- ❌ Missing type hints
- ❌ Skipping Pydantic validation
- ❌ Hardcoded configuration
- ❌ Sync code where async is expected

### Code Style

- Import organization: stdlib → third-party → relative
- 100 character line length (Black formatter)
- snake_case for functions, PascalCase for classes
- Private methods with single underscore prefix
- Comprehensive docstrings on public APIs
