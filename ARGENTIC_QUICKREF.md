# Argentic Framework - Quick Reference

Complete API reference and advanced patterns for AI agent development.

## Table of Contents

- [Installation & Setup](#installation--setup)
- [Core API Reference](#core-api-reference)
- [LLM Providers](#llm-providers)
- [Tool Development](#tool-development)
- [Multi-Agent Patterns](#multi-agent-patterns)
- [Advanced Features](#advanced-features)
- [Message Protocol](#message-protocol)
- [Configuration Reference](#configuration-reference)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Installation & Setup

### Package Installation

```bash
# From PyPI
pip install argentic

# From source (development)
git clone https://github.com/angkira/argentic.git
cd argentic
pip install -e .

# With optional dependencies
pip install -e ".[dev]"      # Development tools
pip install -e ".[docs]"     # Documentation
pip install -e ".[kafka]"    # Kafka messaging
pip install -e ".[redis]"    # Redis messaging
pip install -e ".[rabbitmq]" # RabbitMQ messaging
```

### Environment Setup

Create `.env` file:
```bash
# LLM API Keys
GOOGLE_GEMINI_API_KEY=your_key_here

# MQTT (if using authentication)
MQTT_USERNAME=your_username
MQTT_PASSWORD=your_password

# Logging
LOG_LEVEL=INFO
CONFIG_PATH=config.yaml
```

### MQTT Broker

```bash
# Docker (recommended)
docker run -d -p 1883:1883 --name mosquitto eclipse-mosquitto:2.0

# Ubuntu/Debian
sudo apt install mosquitto mosquitto-clients
sudo systemctl start mosquitto
sudo systemctl enable mosquitto

# macOS
brew install mosquitto
brew services start mosquitto
```

## Core API Reference

### Agent Class

**Full Constructor:**
```python
Agent(
    # Required
    llm: ModelProvider,                    # LLM provider instance
    messager: Messager,                    # Messaging instance
    
    # Optional Core
    tool_manager: Optional[ToolManager] = None,
    role: str = "agent",
    system_prompt: Optional[str] = None,
    description: str = "General AI agent...",
    
    # Output Configuration
    expected_output_format: Literal["json", "text", "code"] = "json",
    task_handling: Literal["direct", "llm"] = "llm",
    
    # MQTT Topics
    register_topic: str = "agent/tools/register",
    tool_call_topic_base: str = "agent/tools/call",
    tool_response_topic_base: str = "agent/tools/response",
    status_topic: str = "agent/status/info",
    answer_topic: str = "agent/response/answer",
    llm_response_topic: Optional[str] = None,
    tool_result_topic: Optional[str] = None,
    
    # System Behavior
    override_default_prompts: bool = False,
    graph_id: Optional[str] = None,
    llm_config: Optional[Dict[str, Any]] = None,
    
    # Logging and Debug
    log_level: Union[str, LogLevel] = LogLevel.INFO,
    enable_dialogue_logging: bool = False,
    max_dialogue_history_items: int = 100,
    max_query_history_items: int = 20,
    
    # Performance and Control
    adaptive_max_iterations: bool = True,
    max_consecutive_tool_calls: int = 3,
    tool_call_window_size: int = 5,
    enable_completion_analysis: bool = True,
    
    # Messaging Control
    publish_to_supervisor: bool = True,
    publish_to_agent_topic: bool = True,
    enable_tool_result_publishing: bool = False,
    
    # State Management
    state_mode: AgentStateMode = AgentStateMode.STATEFUL,
)
```

**Methods:**

```python
# Initialization
await agent.async_init() -> None
    """Initialize agent: subscribe to topics, setup handlers."""

# Query Interface
await agent.query(
    question: str,
    timeout: Optional[float] = None
) -> str
    """Direct query to agent. Returns response string."""

# Task Processing
await agent.process_task(task: AgentTaskMessage) -> None
    """Process task message from supervisor/external source."""

# Messaging
await agent.start_task(
    task: str,
    completion_callback: Callable
) -> str
    """Start async task with callback. Returns task_id."""

# Debug and Monitoring
agent.print_dialogue_summary() -> None
    """Print dialogue history (if logging enabled)."""

agent.get_conversation_history() -> List[Dict[str, Any]]
    """Get full conversation history."""

# Configuration
agent.set_log_level(level: Union[str, LogLevel]) -> None
    """Update logging level dynamically."""
```

### Messager Class

**Constructor:**
```python
Messager(
    broker_address: str = "localhost",
    port: int = 1883,
    client_id: str = "",              # Auto-generated if empty
    username: Optional[str] = None,
    password: Optional[str] = None,
    keepalive: int = 60,
)
```

**Methods:**
```python
# Connection
await messager.connect() -> None
await messager.disconnect() -> None

# Messaging
await messager.publish(
    topic: str,
    message: BaseMessage,
    qos: int = 0
) -> None

await messager.subscribe(
    topic: str,
    callback: Callable,
    message_cls: Type[BaseMessage],
    qos: int = 0
) -> None

await messager.unsubscribe(topic: str) -> None

# Logging
await messager.log(
    message: str,
    level: str = "info",
    topic: str = "agent/log"
) -> None
```

### ToolManager Class

**Constructor:**
```python
ToolManager(
    messager: Messager,
    log_level: Union[LogLevel, str] = LogLevel.INFO,
    register_topic: str = "agent/tools/register",
    tool_call_topic_base: str = "agent/tools/call",
    tool_response_topic_base: str = "agent/tools/response",
    status_topic: str = "agent/status/info",
    default_timeout: int = 30,
)
```

**Methods:**
```python
# Initialization
await tool_manager.async_init() -> None

# Tool Execution
await tool_manager.execute_tool(
    tool_name_or_id: str,
    arguments: Dict[str, Any],
    timeout: Optional[int] = None
) -> Any

# Tool Information
tools_description = tool_manager.get_tools_description() -> str
    """Returns JSON string of all tools for LLM."""

tool_ids = tool_manager.get_tool_ids() -> List[str]
tools = tool_manager.get_tools() -> Dict[str, Dict[str, Any]]

# Configuration
tool_manager.set_log_level(level: Union[LogLevel, str]) -> None
```

### Supervisor Class

**Constructor:**
```python
Supervisor(
    llm: ModelProvider,
    messager: Messager,
    log_level: Union[str, LogLevel] = LogLevel.INFO,
    role: str = "supervisor",
    system_prompt: Optional[str] = None,
    llm_config: Optional[Dict[str, Any]] = None,
    enable_dialogue_logging: bool = False,
    max_task_history_items: int = 10,
    max_dialogue_history_items: int = 50,
    context_cleanup_threshold: int = 100,
)
```

**Methods:**
```python
# Agent Management
supervisor.add_agent(agent: Agent) -> None
    """Register agent with supervisor."""

# Initialization
await supervisor.async_init() -> None

# Task Execution
await supervisor.start_task(
    task: str,
    completion_callback: Callable[[str, bool, str, str], None]
) -> str
    """Start multi-agent workflow. Returns task_id.
    
    Callback signature: (task_id, success, result, error)
    """

# Monitoring
supervisor.print_dialogue_summary() -> None
```

### BaseTool Class

**Constructor:**
```python
class MyTool(BaseTool):
    def __init__(self, messager: Messager):
        super().__init__(
            name: str,                      # Unique tool identifier
            manual: str,                    # Description for LLM
            api: str,                       # JSON schema string
            argument_schema: Type[BaseModel],  # Pydantic model
            messager: Messager,
        )
```

**Methods to Implement:**
```python
async def _execute(self, **kwargs) -> Any:
    """Implement tool logic. Return result or raise exception."""
    pass
```

**Methods to Use:**
```python
await tool.register(
    registration_topic: str,
    status_topic: str,
    call_topic_base: str,
    response_topic_base: str,
) -> None
    """Register tool with ToolManager."""
```

## LLM Providers

### Google Gemini

```python
from argentic.core.llm.providers.google_gemini import GoogleGeminiProvider

llm = GoogleGeminiProvider(config={
    "google_gemini_model_name": "gemini-2.0-flash",
    "google_gemini_api_key": os.getenv("GOOGLE_GEMINI_API_KEY"),
    "google_gemini_parameters": {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 2048,
        "stop_sequences": [],
    }
})
```

**Models:**
- `gemini-2.0-flash` - Fast, efficient (recommended)
- `gemini-2.0-flash-lite` - Ultra-fast for simple tasks
- `gemini-1.5-pro` - Advanced reasoning

### Ollama

```python
from argentic.core.llm.providers.ollama import OllamaProvider

llm = OllamaProvider(config={
    "ollama_model_name": "llama3",
    "ollama_base_url": "http://localhost:11434",
    "ollama_use_chat_model": True,
    "ollama_parameters": {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "num_predict": 128,
        "repeat_penalty": 1.1,
    }
})
```

**Popular Models:**
- `llama3` - Meta's latest
- `mistral` - Fast and capable
- `gemma` - Google's open model
- `codellama` - Code-specialized

### Llama.cpp Server

```python
from argentic.core.llm.providers.llama_cpp import LlamaCppServerProvider

llm = LlamaCppServerProvider(config={
    "llama_cpp_server_host": "127.0.0.1",
    "llama_cpp_server_port": 5000,
    "llama_cpp_server_auto_start": False,
    "llama_cpp_server_parameters": {
        "temperature": 0.8,
        "top_p": 0.95,
        "n_predict": 128,
    }
})
```

### Factory Pattern

```python
from argentic.core.llm.llm_factory import LLMFactory

config = {
    "provider": "google_gemini",
    "google_gemini_model_name": "gemini-2.0-flash",
}

llm = LLMFactory.create_from_config(config)
```

## Tool Development

### Complete Tool Example

```python
import json
from typing import Any
from pydantic import BaseModel, Field
from argentic.core.tools.tool_base import BaseTool
from argentic.core.messager.messager import Messager

# 1. Define Input Schema
class FileWriterInput(BaseModel):
    filename: str = Field(description="Name of file to write")
    content: str = Field(description="Content to write to file")
    append: bool = Field(default=False, description="Append instead of overwrite")

# 2. Implement Tool
class FileWriterTool(BaseTool):
    def __init__(self, messager: Messager):
        super().__init__(
            name="file_writer",
            manual="Write content to a file. Provide filename and content. Set append=true to append.",
            api=json.dumps(FileWriterInput.model_json_schema()),
            argument_schema=FileWriterInput,
            messager=messager,
        )
    
    async def _execute(self, **kwargs) -> Any:
        filename = kwargs["filename"]
        content = kwargs["content"]
        append = kwargs.get("append", False)
        
        mode = "a" if append else "w"
        
        try:
            with open(filename, mode) as f:
                f.write(content)
            return f"Successfully wrote to {filename}"
        except Exception as e:
            raise RuntimeError(f"Failed to write to {filename}: {e}")

# 3. Register Tool
async def setup_tools(messager):
    tool = FileWriterTool(messager)
    await tool.register(
        registration_topic="agent/tools/register",
        status_topic="agent/status/info",
        call_topic_base="agent/tools/call",
        response_topic_base="agent/tools/response",
    )
    return tool
```

### Tool Best Practices

1. **Clear Descriptions:**
   ```python
   manual="Search Wikipedia for information. Provide 'query' (search term) and optionally 'limit' (max results, default 5)."
   ```

2. **Comprehensive Schemas:**
   ```python
   class SearchInput(BaseModel):
       query: str = Field(description="Search query")
       limit: int = Field(default=5, ge=1, le=20, description="Max results (1-20)")
       language: str = Field(default="en", description="Language code (en, es, fr, etc)")
   ```

3. **Error Handling:**
   ```python
   async def _execute(self, **kwargs):
       try:
           result = await some_operation(kwargs["param"])
           return result
       except SpecificError as e:
           raise RuntimeError(f"Operation failed: {e}")
   ```

4. **Validation:**
   Pydantic handles basic validation automatically. Add custom validation:
   ```python
   @validator("limit")
   def validate_limit(cls, v):
       if not 1 <= v <= 100:
           raise ValueError("limit must be between 1 and 100")
       return v
   ```

## Multi-Agent Patterns

### Pattern: Specialized Worker Agents

```python
# Define agent roles
agents_config = [
    {
        "role": "researcher",
        "description": "Research and information gathering",
        "system_prompt": "You are a researcher. Find and synthesize information.",
        "topics": {
            "register": "agent/researcher/tools/register",
            "call": "agent/researcher/tools/call",
            "response": "agent/researcher/tools/response",
            "status": "agent/researcher/status/info",
        }
    },
    {
        "role": "analyst",
        "description": "Data analysis and insights",
        "system_prompt": "You are an analyst. Analyze data and provide insights.",
        "topics": {
            "register": "agent/analyst/tools/register",
            "call": "agent/analyst/tools/call",
            "response": "agent/analyst/tools/response",
            "status": "agent/analyst/status/info",
        }
    },
]

# Create agents
agents = []
for config in agents_config:
    agent = Agent(
        llm=llm,
        messager=messager,
        tool_manager=tool_manager,  # Shared!
        role=config["role"],
        description=config["description"],
        system_prompt=config["system_prompt"],
        register_topic=config["topics"]["register"],
        tool_call_topic_base=config["topics"]["call"],
        tool_response_topic_base=config["topics"]["response"],
        status_topic=config["topics"]["status"],
        graph_id="multi_agent_system",
    )
    await agent.async_init()
    agents.append(agent)

# Create supervisor
supervisor = Supervisor(llm=llm, messager=messager)
for agent in agents:
    supervisor.add_agent(agent)
await supervisor.async_init()
```

### Pattern: Agent-Specific Tools

```python
# Register tools for specific agents
email_tool = EmailTool(messager)
await email_tool.register(
    secretary_register_topic,
    secretary_status_topic,
    secretary_call_topic_base,
    secretary_response_topic_base,
)

search_tool = WebSearchTool(messager)
await search_tool.register(
    researcher_register_topic,
    researcher_status_topic,
    researcher_call_topic_base,
    researcher_response_topic_base,
)

# Tools are now agent-specific via topics
```

## Advanced Features

### Endless Cycle Support

For long-running agents that process many tasks:

```python
agent = Agent(
    llm=llm,
    messager=messager,
    tool_manager=tool_manager,
    # Endless cycle configuration
    adaptive_max_iterations=True,        # Dynamically adjust max iterations
    max_consecutive_tool_calls=3,        # Prevent tool loops
    tool_call_window_size=5,             # Window for pattern detection
    enable_completion_analysis=True,     # Detect task completion
    max_dialogue_history_items=100,      # Limit memory usage
    max_query_history_items=20,          # Limit per-query history
)
```

### State Management

```python
from argentic.core.agent.agent import AgentStateMode

# Stateful: Maintains conversation history
agent = Agent(..., state_mode=AgentStateMode.STATEFUL)

# Stateless: Each query independent
agent = Agent(..., state_mode=AgentStateMode.STATELESS)
```

### Dialogue Logging

```python
agent = Agent(
    ...,
    enable_dialogue_logging=True,
    max_dialogue_history_items=100,
)

# Later
agent.print_dialogue_summary()

# Or get raw history
history = agent.dialogue_history
for entry in history:
    print(f"{entry['role']}: {entry['content_preview']}")
```

### Custom LLM Configuration

```python
agent = Agent(
    ...,
    llm_config={
        "enable_google_search": True,    # Google Gemini grounding
        "temperature_override": 0.9,
        "custom_param": "value",
    }
)
```

## Message Protocol

### Key Message Classes

```python
from argentic.core.protocol.message import (
    AgentTaskMessage,
    AgentTaskResultMessage,
    BaseMessage,
)
from argentic.core.protocol.task import (
    TaskMessage,
    TaskResultMessage,
    TaskErrorMessage,
    TaskStatus,
)
from argentic.core.protocol.tool import (
    RegisterToolMessage,
    ToolRegisteredMessage,
    ToolCallRequest,
)
```

### Message Examples

**AgentTaskMessage:**
```python
task = AgentTaskMessage(
    task="Research quantum computing",
    sender_id="user_interface",
    graph_id="multi_agent_system",
)
await messager.publish("agent/researcher/tasks", task)
```

**TaskMessage (Tool Call):**
```python
task = TaskMessage(
    task_id="uuid-1234",
    tool_id="uuid-5678",
    arguments={"query": "quantum computing", "limit": 5},
)
await messager.publish("agent/tools/call/uuid-5678", task)
```

**TaskResultMessage:**
```python
result = TaskResultMessage(
    task_id="uuid-1234",
    status=TaskStatus.COMPLETED,
    result="Found 5 results about quantum computing...",
)
await messager.publish("agent/tools/response/uuid-5678", result)
```

## Configuration Reference

See `.cursorrules` for complete config.yaml structure. Key sections:

- `llm`: Provider and model configuration
- `messaging`: MQTT broker settings
- `topics`: Topic namespaces
- `agent`: System prompt customization (optional)

## Examples

### Example 1: Simple Q&A Agent

```python
import asyncio
from argentic import Agent, Messager, LLMFactory
from argentic.core.tools import ToolManager

async def main():
    llm = LLMFactory.create_from_config({"provider": "google_gemini", ...})
    messager = Messager(broker_address="localhost", port=1883)
    await messager.connect()
    
    tool_manager = ToolManager(messager)
    await tool_manager.async_init()
    
    agent = Agent(llm, messager, tool_manager, role="assistant")
    await agent.async_init()
    
    response = await agent.query("What is Python?")
    print(response)
    
    await messager.disconnect()

asyncio.run(main())
```

### Example 2: File Operations Tool

See complete FileWriterTool example in [Tool Development](#tool-development) section.

### Example 3: Multi-Agent Research System

See complete example in `.cursorrules` Pattern 3.

## Troubleshooting

### Common Issues

**1. MQTT Connection Failed**
```
Error: Connection refused
```
Solution: Ensure MQTT broker is running:
```bash
docker ps  # Check if mosquitto container is running
# Or
sudo systemctl status mosquitto
```

**2. Tool Not Registered**
```
Warning: Tool 'my_tool' not found
```
Solution:
- Add delay after registration: `await asyncio.sleep(1)`
- Check topic names match between tool and ToolManager
- Enable logging to see registration messages

**3. LLM API Error**
```
Error: Invalid API key
```
Solution:
- Check `.env` file exists and has correct key
- Load dotenv: `from dotenv import load_dotenv; load_dotenv()`
- Verify environment variable: `echo $GOOGLE_GEMINI_API_KEY`

**4. Tool Execution Timeout**
```
TimeoutError: Tool execution exceeded 30s
```
Solution:
- Increase timeout: `await tool_manager.execute_tool(..., timeout=60)`
- Optimize tool implementation
- Check if tool is publishing result

**5. Multi-Agent Not Working**
```
Supervisor not routing to agents
```
Solution:
- Verify supervisor system_prompt includes routing instructions
- Ensure all agents have different roles
- Check agent topics don't conflict
- Enable dialogue logging on supervisor

### Debug Mode

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

agent = Agent(
    ...,
    log_level="DEBUG",
    enable_dialogue_logging=True,
)
```

### Testing Components

Test each component independently:

```python
# Test LLM
response = await llm.generate("Hello", [])
print(response)

# Test Messager
await messager.publish("test/topic", BaseMessage())

# Test Tool
result = await tool._execute(param="value")
print(result)
```

## Additional Resources

- **Official Documentation**: https://argentic.readthedocs.io/ (or your docs site)
- **Examples Directory**: `/examples/` in repository
  - `single_agent_example.py` - Basic usage
  - `multi_agent_example.py` - Multi-agent system
  - `note_creator_tool.py` - Tool development
  - `email_tool.py` - Another tool example
- **GitHub**: https://github.com/angkira/argentic
- **Issues**: Report bugs and request features on GitHub Issues

---

**Last Updated**: October 2025  
**Framework Version**: 0.11.x

