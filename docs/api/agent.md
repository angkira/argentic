# Agent API Reference

The `Agent` class is the core component of the Argentic framework, providing LLM interaction, tool management, and task processing capabilities with advanced features for endless cycle operation.

## Class Definition

```python
class Agent:
    def __init__(
        self,
        llm: ModelProvider,
        messager: Messager,
        tool_manager: Optional[ToolManager] = None,
        log_level: Union[str, LogLevel] = LogLevel.INFO,
        register_topic: str = "agent/tools/register",
        tool_call_topic_base: str = "agent/tools/call",
        tool_response_topic_base: str = "agent/tools/response",
        status_topic: str = "agent/status/info",
        answer_topic: str = "agent/response/answer",
        llm_response_topic: Optional[str] = None,
        tool_result_topic: Optional[str] = None,
        system_prompt: Optional[str] = None,
        override_default_prompts: bool = False,
        role: str = "agent",
        description: str = "General AI agent capable of processing tasks and providing responses",
        graph_id: Optional[str] = None,
        expected_output_format: Literal["json", "text", "code"] = "json",
        llm_config: Optional[Dict[str, Any]] = None,
        task_handling: Literal["direct", "llm"] = "llm",
        enable_dialogue_logging: bool = False,
        max_dialogue_history_items: int = 100,
        max_query_history_items: int = 20,
        adaptive_max_iterations: bool = True,
        max_consecutive_tool_calls: int = 3,
        tool_call_window_size: int = 5,
        enable_completion_analysis: bool = True,
        publish_to_supervisor: bool = True,
        publish_to_agent_topic: bool = True,
        enable_tool_result_publishing: bool = False,
    )
```

## Parameters

### Core Parameters

#### `llm: ModelProvider` (required)
The LLM provider instance for generating responses.

#### `messager: Messager` (required) 
The messaging system for inter-component communication.

#### `tool_manager: Optional[ToolManager] = None`
Tool manager instance. If None, a new one will be created.

#### `log_level: Union[str, LogLevel] = LogLevel.INFO`
Logging level. Can be string ("DEBUG", "INFO", etc.) or LogLevel enum.

### Agent Identity

#### `role: str = "agent"`
Unique identifier for the agent within the system.

#### `description: str = "General AI agent..."`
Human-readable description of the agent's capabilities and purpose.

#### `graph_id: Optional[str] = None`
Graph identifier for multi-agent workflow coordination.

### System Prompt Configuration

#### `system_prompt: Optional[str] = None`
Custom system prompt for the agent. If None, uses default prompt.

#### `override_default_prompts: bool = False`
- `False`: Combines default utility rules with custom system prompt
- `True`: Uses only the custom system prompt, bypassing defaults

**Example:**
```python
# Recommended: Include default utility rules
agent = Agent(
    llm=llm,
    messager=messager,
    system_prompt="You are a helpful research assistant.",
    override_default_prompts=False  # Adds default tool/completion rules
)

# Advanced: Complete override for specialized cases
agent = Agent(
    llm=llm,
    messager=messager,
    system_prompt="You are a specialized domain expert.",
    override_default_prompts=True   # Only custom prompt, no defaults
)
```

### Output Configuration

#### `expected_output_format: Literal["json", "text", "code"] = "json"`
Expected response format from the agent:
- `"json"`: Structured JSON responses with tool calls
- `"text"`: Plain text responses  
- `"code"`: Code-focused responses

#### `llm_config: Optional[Dict[str, Any]] = None`
Provider-specific LLM configuration parameters.

#### `task_handling: Literal["direct", "llm"] = "llm"`
How the agent processes tasks:
- `"llm"`: Process through LLM query loop
- `"direct"`: Direct task processing

### Endless Cycle Management

#### `enable_dialogue_logging: bool = False`
Enable real-time dialogue logging for UI display and debugging.

#### `max_dialogue_history_items: int = 100`
Maximum dialogue entries to retain before cleanup.

#### `max_query_history_items: int = 20`
Maximum history items per query to prevent context overflow.

#### `adaptive_max_iterations: bool = True`
Automatically adjust max iterations based on task complexity.

### Tool Loop Prevention

#### `max_consecutive_tool_calls: int = 3`
Maximum identical consecutive tool calls before loop detection.

#### `tool_call_window_size: int = 5`
Window size for detecting tool call patterns.

#### `enable_completion_analysis: bool = True`
Analyze tool results to detect task completion.

### Messaging Control

#### `publish_to_supervisor: bool = True`
Publish task results to supervisor for multi-agent coordination.

#### `publish_to_agent_topic: bool = True`
Publish results to agent-specific topic for monitoring.

#### `enable_tool_result_publishing: bool = False`
Publish individual tool execution results.

### MQTT Topics

#### `register_topic: str = "agent/tools/register"`
Topic for tool registration messages.

#### `tool_call_topic_base: str = "agent/tools/call"`
Base topic for tool execution requests.

#### `tool_response_topic_base: str = "agent/tools/response"`
Base topic for tool execution responses.

#### `status_topic: str = "agent/status/info"`
Topic for agent status updates.

#### `answer_topic: str = "agent/response/answer"`
Topic for publishing agent responses.

#### `llm_response_topic: Optional[str] = None`
Optional topic for LLM response publishing.

#### `tool_result_topic: Optional[str] = None`
Optional topic for tool result publishing.

## Methods

### Core Methods

#### `async def async_init(self) -> None`
Initialize the agent and its components.

**Must be called** after agent creation and before use.

```python
agent = Agent(llm=llm, messager=messager)
await agent.async_init()  # Required
```

#### `async def query(self, question: str, user_id: Optional[str] = None, max_iterations: Optional[int] = None) -> str`
Process a question through the LLM and tool interaction loop.

**Parameters:**
- `question`: The question or task to process
- `user_id`: Optional user identifier for tracking
- `max_iterations`: Override default iteration limit

**Returns:** String response from the agent

**Example:**
```python
result = await agent.query("Create a report on quantum computing and email it to john@example.com")
```

#### `async def invoke(self, state: AgentState) -> dict[str, list[LangchainBaseMessage]]`
Invoke agent as part of a LangGraph workflow.

**Parameters:**
- `state`: Current graph state with messages

**Returns:** Updated state with agent's response

### Configuration Methods

#### `def set_system_prompt(self, system_prompt: str, override_default_prompts: Optional[bool] = None) -> None`
Update the system prompt and rebuild prompt template.

**Parameters:**
- `system_prompt`: New system prompt
- `override_default_prompts`: Override current setting if provided

#### `def get_system_prompt(self) -> str`
Get the current effective system prompt (including defaults if applicable).

#### `def set_log_level(self, level: Union[str, LogLevel]) -> None`
Update the logging level for the agent and its components.

### Dialogue and History

#### `def get_dialogue_history(self) -> List[Dict[str, Any]]`
Get copy of the complete dialogue history.

#### `def print_dialogue_summary(self) -> None`
Print formatted summary of dialogue history.

### Task Handling

#### `async def handle_task(self, message: BaseMessage) -> None`
Handler for incoming task messages via MQTT.

#### `async def handle_ask_question(self, message: AskQuestionMessage) -> None`
Handler for incoming question messages via MQTT.

### Utility Methods

#### `async def stop(self) -> None`
Stop the agent and clean up resources.

## Usage Examples

### Basic Agent Setup

```python
from argentic.core.agent.agent import Agent
from argentic.core.llm.providers.google_gemini import GoogleGeminiProvider
from argentic.core.messager.messager import Messager

# Initialize components
llm = GoogleGeminiProvider(config={"google_gemini_model_name": "gemini-2.0-flash-lite"})
messager = Messager(broker_address="localhost", port=1883)
await messager.connect()

# Create and initialize agent
agent = Agent(
    llm=llm,
    messager=messager,
    role="research_assistant",
    description="AI assistant specialized in research and analysis",
    enable_dialogue_logging=True,
)
await agent.async_init()

# Use the agent
result = await agent.query("Research the latest developments in quantum computing")
print(result)
```

### Multi-Agent Production Setup

```python
# Production configuration for endless cycles
agent = Agent(
    llm=llm,
    messager=messager,
    role="production_agent",
    description="Production AI agent for customer support",
    
    # System prompt configuration
    system_prompt="You are a professional customer support agent...",
    override_default_prompts=False,  # Include utility rules
    
    # Endless cycle management
    max_dialogue_history_items=50,   # Limit memory usage
    max_query_history_items=10,      # Prevent context overflow
    adaptive_max_iterations=True,    # Smart iteration limits
    
    # Tool loop prevention
    max_consecutive_tool_calls=2,    # Aggressive loop detection
    enable_completion_analysis=True, # Detect task completion
    
    # Messaging control
    publish_to_supervisor=True,      # Multi-agent coordination
    publish_to_agent_topic=True,     # Monitoring
    enable_tool_result_publishing=False,  # Reduce message overhead
    
    # Logging
    enable_dialogue_logging=False,   # Disable for performance
)
```

### Development/Debug Setup

```python
# Full monitoring for development
agent = Agent(
    llm=llm,
    messager=messager,
    role="debug_agent",
    
    # Full logging and monitoring
    enable_dialogue_logging=True,
    enable_tool_result_publishing=True,
    
    # Relaxed limits for testing
    max_consecutive_tool_calls=5,
    max_dialogue_history_items=200,
    
    # All messaging enabled
    publish_to_supervisor=True,
    publish_to_agent_topic=True,
)
```

### Single Agent Mode

```python
# Minimal messaging for single-agent scenarios
agent = Agent(
    llm=llm,
    messager=messager,
    role="standalone_agent",
    
    # Disable multi-agent features
    publish_to_supervisor=False,
    publish_to_agent_topic=False,
    
    # Focus on direct interaction
    expected_output_format="text",
    enable_dialogue_logging=True,
)
```

## Advanced Features

### Tool Loop Detection

The agent automatically detects and prevents infinite tool calling loops:

```python
# Configure aggressive loop detection
agent = Agent(
    llm=llm,
    messager=messager,
    max_consecutive_tool_calls=2,    # Very strict
    tool_call_window_size=4,         # Small detection window
    enable_completion_analysis=True, # Auto-detect completion
)
```

### Task Completion Analysis

The agent analyzes tool results to determine task completion:

- Looks for success indicators: ✅, "successfully", "completed", "created", "saved", "sent"
- Analyzes multiple tool executions
- Provides completion-focused prompts to conclude tasks

### Context Management

Automatic context management prevents memory issues in endless cycles:

- **Dialogue History Cleanup**: Periodic cleanup of old dialogue entries
- **Query History Limits**: Per-query context truncation
- **Smart Truncation**: Preserves essential information while reducing size

### Messaging Architecture

Flexible messaging control for different deployment scenarios:

```python
# Full coordination (default)
agent = Agent(publish_to_supervisor=True, publish_to_agent_topic=True)

# Performance optimized
agent = Agent(publish_to_supervisor=True, publish_to_agent_topic=False)

# Debug mode
agent = Agent(enable_tool_result_publishing=True)
```

## Error Handling

The agent provides comprehensive error handling:

- **LLM Response Parsing**: Graceful handling of malformed responses
- **Tool Execution Errors**: Automatic retry and error reporting
- **Network Issues**: Connection resilience and reconnection
- **Resource Limits**: Memory and context management

## Performance Considerations

### For High Throughput

```python
agent = Agent(
    enable_dialogue_logging=False,      # Reduce overhead
    publish_to_agent_topic=False,       # Minimal messaging
    enable_tool_result_publishing=False, # No tool monitoring
    max_dialogue_history_items=20,      # Small memory footprint
)
```

### For Long-Running Operations

```python
agent = Agent(
    adaptive_max_iterations=True,       # Smart iteration limits
    enable_completion_analysis=True,    # Auto-detect completion
    max_consecutive_tool_calls=3,       # Prevent loops
    max_dialogue_history_items=100,     # Reasonable memory usage
)
```

## See Also

- [Supervisor API](supervisor.md) - Multi-agent coordination
- [BaseTool API](base-tool.md) - Tool development
- [Messaging Configuration](../messaging-configuration.md) - MQTT setup
- [System Prompt Configuration](../system-prompt-configuration.md) - Prompt customization 