# Argentic API Reference

Comprehensive API documentation for the Argentic multi-agent framework with endless cycle support.

## Core Components

### 🤖 [Agent](agent.md)
The primary component for creating AI agents with LLM interaction, tool management, and advanced endless cycle features.

**Key Features:**
- Tool loop detection and prevention
- Task completion analysis
- Context management for long-running operations
- Configurable messaging control
- Real-time dialogue logging

### 👨‍💼 [Supervisor](supervisor.md)  
Multi-agent workflow coordinator with pure messaging-based communication and context management.

**Key Features:**
- Dynamic agent routing using LLM decisions
- Context truncation for endless cycles
- Automatic cleanup and memory management
- Workflow completion callbacks
- Task history management

### 🛠️ [BaseTool](base-tool.md)
Foundation class for creating custom tools with automatic validation, messaging, and error handling.

**Key Features:**
- Pydantic-based argument validation
- Automatic MQTT integration
- Comprehensive error handling
- Async/await support
- Built-in logging and debugging

### ⚙️ [ToolManager](tool-manager.md)
Handles tool registration, execution coordination, and result management with concurrent execution support.

**Key Features:**
- Automatic tool registration
- Concurrent tool execution
- Timeout management
- Tool discovery and description generation
- Result aggregation and error handling

## Quick Start

### Basic Agent Setup

```python
from argentic.core.agent.agent import Agent
from argentic.core.llm.providers.google_gemini import GoogleGeminiProvider
from argentic.core.messager.messager import Messager

# Initialize components
llm = GoogleGeminiProvider(config={"google_gemini_model_name": "gemini-2.0-flash-lite"})
messager = Messager(broker_address="localhost", port=1883)
await messager.connect()

# Create agent with endless cycle features
agent = Agent(
    llm=llm,
    messager=messager,
    role="assistant",
    description="Helpful AI assistant",
    enable_dialogue_logging=True,
    adaptive_max_iterations=True,
    enable_completion_analysis=True,
)
await agent.async_init()

# Use the agent
result = await agent.query("Help me create a report and email it")
```

### Multi-Agent Workflow

```python
from argentic.core.graph.supervisor import Supervisor

# Create supervisor
supervisor = Supervisor(
    llm=llm,
    messager=messager,
    enable_dialogue_logging=True,
)
await supervisor.async_init()

# Add specialized agents
researcher = Agent(llm=llm, messager=messager, role="researcher", 
                  description="Research and data analysis specialist")
writer = Agent(llm=llm, messager=messager, role="writer",
               description="Content creation and documentation specialist")

supervisor.add_agent(researcher)
supervisor.add_agent(writer)

# Start workflow
task_id = await supervisor.start_task(
    "Research AI trends and create a comprehensive report"
)
```

### Custom Tool Development

```python
from argentic.core.tools.tool_base import BaseTool
from pydantic import BaseModel, Field
import json

# Define tool arguments
class EmailArguments(BaseModel):
    recipient: str = Field(description="Email recipient address")
    subject: str = Field(description="Email subject line")
    message: str = Field(description="Email content")

# Implement tool
class EmailTool(BaseTool):
    def __init__(self, messager):
        api_schema = EmailArguments.model_json_schema()
        super().__init__(
            name="Email Sender",
            manual="Send emails to specified recipients",
            api=json.dumps(api_schema),
            argument_schema=EmailArguments,
            messager=messager,
        )
    
    async def _execute(self, recipient: str, subject: str, message: str) -> str:
        # Implement email sending logic
        send_email(recipient, subject, message)
        return f"✅ Email sent to {recipient}"

# Register tool
email_tool = EmailTool(messager)
await email_tool.register(
    registration_topic="agent/tools/register",
    status_topic="agent/status/info",
    call_topic_base="agent/tools/call",
    response_topic_base="agent/tools/response"
)
```

## Advanced Features

### Endless Cycle Management

All components include features for endless cycle operation:

- **Context Management**: Automatic cleanup and size limits
- **Loop Detection**: Prevents infinite tool calling cycles
- **Completion Analysis**: Smart task completion detection
- **Memory Management**: Bounded data structures and periodic cleanup
- **Performance Optimization**: Adaptive iteration limits and context truncation

### Messaging Control

Fine-grained control over MQTT messaging for different scenarios:

```python
# Production setup - minimal messaging
agent = Agent(
    llm=llm, messager=messager,
    publish_to_supervisor=True,           # Multi-agent coordination only
    publish_to_agent_topic=False,         # No monitoring overhead
    enable_tool_result_publishing=False,  # No detailed tool logging
)

# Development setup - full monitoring
agent = Agent(
    llm=llm, messager=messager,
    publish_to_supervisor=True,          # Full coordination
    publish_to_agent_topic=True,         # Monitoring enabled
    enable_tool_result_publishing=True,  # Detailed tool monitoring
    enable_dialogue_logging=True,        # Real-time dialogue
)
```

### System Prompt Management

Flexible system prompt configuration with utility rules:

```python
# Recommended: Custom prompt + utility rules
agent = Agent(
    llm=llm, messager=messager,
    system_prompt="You are a specialized research assistant.",
    override_default_prompts=False,  # Include tool/completion rules
)

# Advanced: Complete custom control
agent = Agent(
    llm=llm, messager=messager,
    system_prompt="Your domain-specific prompt here.",
    override_default_prompts=True,   # Skip all default rules
)
```

## Configuration Guides

### 📡 [Messaging Configuration](../messaging-configuration.md)
MQTT setup, topic structure, TLS configuration, and messaging patterns.

### 🎛️ [System Prompt Configuration](../system-prompt-configuration.md)  
Custom prompts, utility rules, and prompt management.

### ⚙️ [Advanced LLM Configuration](../advanced-llm-configuration.md)
Provider-specific parameters, performance tuning, and logging configuration.

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Supervisor    │    │      Agent       │    │     Tools       │
│                 │    │                  │    │                 │
│ • Task Routing  │◄──►│ • LLM Integration│◄──►│ • BaseTool      │
│ • Workflow Mgmt │    │ • Tool Management│    │ • Custom Logic  │
│ • Context Mgmt  │    │ • Loop Detection │    │ • Validation    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────────┐
                    │   Messager (MQTT)   │
                    │                     │
                    │ • Pub/Sub Messaging │
                    │ • Topic Management  │
                    │ • Error Handling    │
                    └─────────────────────┘
```

## Performance Considerations

### High Throughput Setup

```python
# Optimized for performance
agent = Agent(
    llm=llm, messager=messager,
    enable_dialogue_logging=False,      # Reduce overhead
    max_dialogue_history_items=20,      # Minimal memory
    publish_to_agent_topic=False,       # No monitoring
    adaptive_max_iterations=True,       # Smart limits
)
```

### Development/Debug Setup

```python
# Optimized for debugging
agent = Agent(
    llm=llm, messager=messager,
    enable_dialogue_logging=True,       # Full dialogue tracking
    enable_tool_result_publishing=True, # Detailed tool monitoring
    max_dialogue_history_items=200,     # Extended history
    log_level="DEBUG",                  # Detailed logging
)
```

## Error Handling

All components provide comprehensive error handling:

- **Graceful Degradation**: Continues operation despite individual failures
- **Detailed Error Messages**: Clear descriptions and troubleshooting information
- **Automatic Recovery**: Reconnection logic and retry mechanisms
- **Resource Protection**: Timeouts and cleanup to prevent resource exhaustion

## Migration Guide

### From Previous Versions

The new API maintains backward compatibility while adding powerful new features:

```python
# Old API (still works)
agent = Agent(llm=llm, messager=messager)

# New API with endless cycle features
agent = Agent(
    llm=llm, messager=messager,
    # New endless cycle parameters
    enable_completion_analysis=True,
    max_consecutive_tool_calls=3,
    adaptive_max_iterations=True,
    # New messaging control
    publish_to_supervisor=True,
    enable_dialogue_logging=True,
)
```

## Community and Support

- **Examples**: Check the `examples/` directory for complete implementations
- **Issues**: Report bugs and feature requests on GitHub
- **Documentation**: Comprehensive guides in the `docs/` directory
- **Testing**: Run the endless cycle test suite with `examples/endless_cycle_test.py`

## See Also

- [Configuration Guide](../configuration.md) - General framework configuration
- [Environment Variables](../environment-variables.md) - Environment setup
- [Examples](../../examples/) - Complete working examples
- [Endless Cycle Improvements](../ENDLESS_CYCLE_IMPROVEMENTS.md) - Technical implementation details 