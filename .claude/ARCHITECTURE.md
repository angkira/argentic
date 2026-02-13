# Argentic Architecture

## System Overview

Argentic is a **message-driven, async-first AI agent framework** built on these principles:

1. **Decoupled Communication**: All components communicate via messages, not direct calls
2. **Pluggable Providers**: LLM, messaging, and storage are abstracted interfaces
3. **Async Everything**: All I/O operations are non-blocking
4. **Type Safety**: Pydantic models for all data structures
5. **Configuration-Driven**: Behavior controlled via YAML

---

## Component Hierarchy

```
┌────────────────────────────────────────────────────────────┐
│                     Application Layer                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   CLI Client │  │ Single Agent │  │ Multi-Agent  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────────────────────────────────────────────┘
                             │
                             v
┌────────────────────────────────────────────────────────────┐
│                      Core Framework                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │    Agent     │  │  Supervisor  │  │ Tool Manager │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                  │                  │             │
│         └──────────────────┴──────────────────┘             │
│                             │                               │
│                             v                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Messager (Unified Interface)            │  │
│  └──────┬──────────────┬──────────────┬─────────────┬──┘  │
└─────────┼──────────────┼──────────────┼─────────────┼─────┘
          │              │              │             │
          v              v              v             v
┌─────────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐
│ MQTT Driver │  │  Kafka   │  │  Redis   │  │  RabbitMQ  │
└─────────────┘  └──────────┘  └──────────┘  └────────────┘
```

---

## Data Flow

### Query Processing Flow

```
1. User Input
   └─> AskQuestionMessage
       └─> Agent.handle_ask_question()

2. Agent Processing
   └─> Convert to ChatMessage[]
       └─> Agent._call_llm()
           └─> LLM Provider (Ollama/Gemini/etc)

3. LLM Response Analysis
   ├─> Direct Answer?
   │   └─> Return AnswerMessage
   │
   └─> Tool Calls?
       └─> For each tool:
           ├─> ToolManager.execute_tool()
           ├─> Publish TaskMessage
           ├─> Tool receives and executes
           └─> TaskResultMessage returned

4. Tool Results Integration
   └─> Add results to conversation
       └─> Back to step 2 (next iteration)

5. Final Response
   └─> AnswerMessage
       └─> Published to response topic
```

---

## Message Protocol Architecture

### Message Class Hierarchy

```
BaseMessage[T] (Generic Pydantic Model)
├── type: str
├── source: MessageSource
├── timestamp: float
├── message_id: UUID
└── data: Optional[T]

├─ System Messages
│  ├─ AgentSystemMessage
│  ├─ AgentLLMRequestMessage
│  └─ AgentLLMResponseMessage
│
├─ Task Messages
│  ├─ TaskMessage (base)
│  ├─ TaskResultMessage
│  └─ TaskErrorMessage
│
├─ Tool Messages
│  ├─ RegisterToolMessage
│  ├─ ToolRegisteredMessage
│  └─ ToolRegistrationErrorMessage
│
├─ Query Messages
│  ├─ AskQuestionMessage
│  └─ AnswerMessage
│
└─ Chat Messages
   ├─ SystemMessage (role="system")
   ├─ UserMessage (role="user")
   ├─ AssistantMessage (role="assistant")
   └─ ToolMessage (role="tool")
```

### Topic Structure

```
agent/
├── command/
│   └── ask_question          # User queries
├── response/
│   └── answer                # Agent responses
├── tools/
│   ├── register              # Tool registration
│   ├── call/{tool_id}        # Tool execution
│   └── response/{task_id}    # Tool results
└── events/
    ├── llm_response          # LLM outputs
    └── tool_result           # Tool execution events
```

---

## Agent State Machine

```
┌─────────────┐
│  Created    │
└──────┬──────┘
       │ __init__()
       v
┌─────────────┐
│ Initialized │
└──────┬──────┘
       │ async_init()
       v
┌─────────────┐
│   Ready     │ ←──────────┐
└──────┬──────┘            │
       │ query()           │
       v                   │
┌─────────────┐            │
│ Processing  │            │
├─────────────┤            │
│ - Call LLM  │            │
│ - Parse     │            │
│ - Tools?    │            │
└──────┬──────┘            │
       │                   │
       ├─> No Tools ───────┘
       │
       └─> Has Tools
           │
           v
    ┌──────────────┐
    │ Tool Exec    │
    ├──────────────┤
    │ - Publish    │
    │ - Await      │
    │ - Integrate  │
    └──────┬───────┘
           │
           └──> Back to Processing
```

---

## Tool Registration Flow

```
Tool                  ToolManager                Agent
 │                         │                       │
 │ RegisterToolMessage     │                       │
 ├────────────────────────>│                       │
 │                         │                       │
 │                    [Validate]                   │
 │                    [Assign ID]                  │
 │                         │                       │
 │ ToolRegisteredMessage   │                       │
 │<────────────────────────┤                       │
 │                         │                       │
 │ Subscribe to task topic │                       │
 │                         │                       │
 │                    [Cache tool                  │
 │                     descriptions]               │
 │                         │                       │
 │                         │  get_tool_descriptions│
 │                         │<──────────────────────┤
 │                         │                       │
 │                         │  Return JSON schema   │
 │                         ├──────────────────────>│
```

---

## Tool Execution Flow

```
Agent                ToolManager              Tool
 │                       │                     │
 │ execute_tool()        │                     │
 ├──────────────────────>│                     │
 │                       │                     │
 │                  [Create TaskMessage]       │
 │                  [Create Future]            │
 │                       │                     │
 │                       │  TaskMessage        │
 │                       ├────────────────────>│
 │                       │                     │
 │                       │              [Validate args]
 │                       │              [Execute]
 │                       │                     │
 │                       │  TaskResultMessage  │
 │                       │<────────────────────┤
 │                       │                     │
 │                  [Resolve Future]           │
 │                       │                     │
 │ Return result         │                     │
 │<──────────────────────┤                     │
```

---

## LLM Provider Architecture

```
┌────────────────────────────────────┐
│       LLMFactory (Factory)         │
│  create_provider(config) -> Provider
└────────────────┬───────────────────┘
                 │
                 v
         ┌───────────────┐
         │ ModelProvider │ (Abstract Interface)
         │ - chat()      │
         │ - achat()     │
         └───────┬───────┘
                 │
    ┌────────────┼────────────┬──────────────┐
    │            │            │              │
    v            v            v              v
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────┐
│ Ollama  │ │ Gemini  │ │Llama.cpp│ │    Mock     │
│Provider │ │Provider │ │Provider │ │  Provider   │
└─────────┘ └─────────┘ └─────────┘ └─────────────┘
```

### Provider Responsibilities

1. **Normalize Input**: Convert ChatMessage[] to provider format
2. **Call API**: Invoke LLM (sync or async)
3. **Parse Output**: Extract content and tool calls
4. **Return Response**: Standardized `LLMChatResponse` object

---

## Multi-Agent Architecture (Supervisor Pattern)

```
        ┌──────────────┐
        │  Supervisor  │
        │  (Orchestrates)
        └──────┬───────┘
               │
       ┌───────┴────────┐
       │                │
       v                v
┌─────────────┐  ┌─────────────┐
│ Agent 1     │  │ Agent 2     │
│ (Researcher)│  │ (Secretary) │
└─────────────┘  └─────────────┘
       │                │
       └────────┬───────┘
                │
         (Shared Messager)
```

### Supervisor Flow

1. Supervisor receives task
2. Analyzes and routes to appropriate agent
3. Agent processes and publishes result
4. Supervisor collects and synthesizes
5. Returns final response

---

## State Management

### Agent State Modes

**STATEFUL** (default):
```
┌────────────────────────────────┐
│  Agent Context                 │
├────────────────────────────────┤
│  history: List[BaseMessage]    │
│  ├─ System prompt              │
│  ├─ Previous questions         │
│  ├─ Previous answers           │
│  └─ Tool results               │
│                                │
│  Max items: configurable       │
│  Truncation: automatic         │
└────────────────────────────────┘
```

**STATELESS**:
```
┌────────────────────────────────┐
│  Agent Context                 │
├────────────────────────────────┤
│  ├─ System prompt              │
│  └─ Current question ONLY      │
│                                │
│  No history preserved          │
└────────────────────────────────┘
```

---

## Error Handling Architecture

### Error Flow

```
┌─────────────┐
│  Operation  │
└──────┬──────┘
       │
       v
   [Try Block]
       │
       ├─> Success
       │   └─> Return result
       │
       └─> Exception
           │
           v
      ┌────────────┐
      │ Log Error  │
      └─────┬──────┘
            │
            v
      ┌────────────┐
      │ Publish    │
      │ ErrorMsg   │
      └─────┬──────┘
            │
            v
      ┌────────────┐
      │ Return/    │
      │ Raise      │
      └────────────┘
```

### Error Categories

1. **Network Errors**: Connection failures, timeouts
2. **Validation Errors**: Pydantic validation failures
3. **Execution Errors**: Tool execution failures
4. **LLM Errors**: Provider API failures

---

## Deployment Patterns

### Single Agent Service
```
┌─────────────────────────────┐
│  Docker Container           │
│  ┌─────────────────────┐    │
│  │  Agent Process      │    │
│  │  - LLM: Ollama      │    │
│  │  - Messager: MQTT   │    │
│  └─────────────────────┘    │
└─────────────────────────────┘
         │
         v
┌─────────────────────────────┐
│  MQTT Broker (External)     │
└─────────────────────────────┘
```

### Multi-Agent System
```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Supervisor  │  │   Agent 1    │  │   Agent 2    │
│  Container   │  │  Container   │  │  Container   │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                  │
       └─────────────────┴──────────────────┘
                         │
                         v
              ┌─────────────────────┐
              │    Message Broker   │
              │  (MQTT/Kafka/Redis) │
              └─────────────────────┘
```

### Tool as Microservice
```
┌──────────────┐           ┌──────────────┐
│  Agent       │           │  Tool Service│
│  Container   │           │  Container   │
│              │           │  - RAG       │
│              │           │  - Search    │
└──────┬───────┘           └──────┬───────┘
       │                          │
       └──────────┬───────────────┘
                  │
                  v
       ┌─────────────────────┐
       │   Message Broker    │
       └─────────────────────┘
```

---

## Performance Characteristics

### Async Concurrency
- **Single Agent**: Handles multiple queries concurrently
- **Tool Execution**: Parallel execution via asyncio.gather()
- **Message Handling**: Non-blocking I/O throughout

### Resource Usage
- **Memory**: Grows with conversation history (truncation available)
- **CPU**: Minimal (mostly I/O-bound)
- **Network**: Depends on message broker throughput

### Scalability
- **Horizontal**: Add more agent instances
- **Vertical**: Increase LLM threads, connection pools
- **Tool Services**: Independent scaling per tool

---

## Security Considerations

1. **API Keys**: Stored in .env, not in config.yaml
2. **Message Validation**: Pydantic schemas prevent injection
3. **Timeout Protection**: All operations have timeouts
4. **Graceful Shutdown**: Proper cleanup on SIGTERM/SIGINT
5. **Error Sanitization**: Tracebacks in logs, not user-facing

---

## Extension Points

### Adding New Components

1. **LLM Provider**:
   - Implement `ModelProvider` interface
   - Add to `LLMFactory`
   - Update config schema

2. **Message Protocol**:
   - Add driver in `messager/drivers/`
   - Implement `DriverProtocol` interface
   - Update `MessagerProtocol` enum

3. **Tool**:
   - Subclass `BaseTool`
   - Implement `_execute()`
   - Deploy as service or embed

4. **Agent Behavior**:
   - Extend `Agent` class
   - Override `_call_llm()` or `_handle_tool_result()`
   - Custom prompting logic

---

## Architecture Benefits

1. **Loose Coupling**: Components communicate only via messages
2. **Testability**: Easy to mock message broker, LLM, tools
3. **Flexibility**: Swap providers without code changes
4. **Scalability**: Distribute components across services
5. **Observability**: All communication logged and traceable
6. **Resilience**: Timeout and retry logic throughout
