# ToolManager API Reference

The `ToolManager` class handles tool registration, execution coordination, and result management in the Argentic framework with async support and comprehensive error handling.

## Class Definition

```python
class ToolManager:
    def __init__(
        self,
        messager: Messager,
        log_level: Union[LogLevel, str] = LogLevel.INFO,
        register_topic: str = "agent/tools/register",
        tool_call_topic_base: str = "agent/tools/call",
        tool_response_topic_base: str = "agent/tools/response", 
        status_topic: str = "agent/status/info",
        default_timeout: int = 30,
    )
```

## Parameters

#### `messager: Messager` (required)
The messaging system instance for MQTT communication.

#### `log_level: Union[LogLevel, str] = LogLevel.INFO`
Logging level for tool manager operations.

#### `register_topic: str = "agent/tools/register"`
MQTT topic for tool registration requests.

#### `tool_call_topic_base: str = "agent/tools/call"`
Base topic for tool execution requests. Tools subscribe to `{base}/{tool_id}`.

#### `tool_response_topic_base: str = "agent/tools/response"`
Base topic for tool responses. Tools publish to `{base}/{tool_id}`.

#### `status_topic: str = "agent/status/info"`
Topic for publishing tool registration confirmations.

#### `default_timeout: int = 30`
Default timeout in seconds for tool execution.

## Properties

#### `tools_by_id: Dict[str, Dict[str, Any]]`
Registry of tools indexed by unique tool ID.

#### `tools_by_name: Dict[str, Dict[str, Any]]`
Registry of tools indexed by tool name.

## Methods

### Core Methods

#### `async def async_init(self) -> None`
Initialize the ToolManager and subscribe to registration topics.

**Must be called** after creation and before use.

```python
tool_manager = ToolManager(messager)
await tool_manager.async_init()  # Required
```

#### `async def execute_tool(self, tool_name_or_id: str, arguments: Dict[str, Any], task_id_override: Optional[str] = None) -> Union[TaskResultMessage, TaskErrorMessage]`
Execute a tool by name or ID with given arguments.

**Parameters:**
- `tool_name_or_id`: Tool name or unique ID
- `arguments`: Tool arguments as dictionary
- `task_id_override`: Optional custom task ID

**Returns:** TaskResultMessage on success, TaskErrorMessage on failure

**Example:**
```python
result = await tool_manager.execute_tool(
    "Email Sender",
    {
        "recipient": "user@example.com",
        "subject": "Test",
        "message": "Hello World"
    }
)

if result.status == TaskStatus.SUCCESS:
    print(f"Success: {result.result}")
else:
    print(f"Error: {result.error}")
```

#### `async def get_tool_results(self, tool_call_requests: List[ToolCallRequest]) -> Tuple[List[Union[TaskResultMessage, TaskErrorMessage]], bool]`
Execute multiple tools concurrently.

**Parameters:**
- `tool_call_requests`: List of ToolCallRequest objects

**Returns:** Tuple of (results_list, any_errors_bool)

**Example:**
```python
requests = [
    ToolCallRequest(tool_id="email_tool", arguments={"recipient": "user1@example.com", "message": "Hi"}),
    ToolCallRequest(tool_id="file_tool", arguments={"filename": "report.txt", "content": "Data"})
]

results, has_errors = await tool_manager.get_tool_results(requests)

for result in results:
    if result.status == TaskStatus.SUCCESS:
        print(f"✅ {result.tool_name}: {result.result}")
    else:
        print(f"❌ {result.tool_name}: {result.error}")
```

### Information Methods

#### `def generate_tool_descriptions_for_prompt(self) -> str`
Generate JSON string describing all available tools for LLM prompts.

**Returns:** JSON string with tool definitions

#### `def get_tools_description(self) -> str`
Alias for `generate_tool_descriptions_for_prompt()`.

#### `def get_tool_names(self) -> List[str]`
Get list of all registered tool names.

**Returns:** List of tool names

#### `def set_log_level(self, level: Union[LogLevel, str]) -> None`
Update the logging level for the ToolManager.

### Internal Handler Methods

#### `async def _handle_register_tool(self, reg_msg: BaseMessage) -> None`
Handle tool registration requests (internal).

#### `async def _handle_result_message(self, msg: BaseMessage) -> None`
Handle tool execution results (internal).

#### `async def _handle_unregister_tool(self, unreg_msg: UnregisterToolMessage) -> None`
Handle tool unregistration requests (internal).

## Usage Examples

### Basic Setup

```python
from argentic.core.tools.tool_manager import ToolManager
from argentic.core.messager.messager import Messager

# Initialize messaging
messager = Messager(broker_address="localhost", port=1883)
await messager.connect()

# Create and initialize ToolManager
tool_manager = ToolManager(
    messager=messager,
    log_level="INFO",
    default_timeout=60  # 1 minute timeout
)
await tool_manager.async_init()
```

### Agent Integration

```python
from argentic.core.agent.agent import Agent

# ToolManager is created automatically by Agent
agent = Agent(
    llm=llm,
    messager=messager,
    # ToolManager created with these topic settings
    register_topic="agent/tools/register",
    tool_call_topic_base="agent/tools/call",
    tool_response_topic_base="agent/tools/response",
    status_topic="agent/status/info"
)
await agent.async_init()

# Access the agent's tool manager
tool_manager = agent._tool_manager
```

### Custom Topic Configuration

```python
# Custom MQTT topic structure
tool_manager = ToolManager(
    messager=messager,
    register_topic="custom/tools/register",
    tool_call_topic_base="custom/tools/execute",
    tool_response_topic_base="custom/tools/results",
    status_topic="custom/tools/status",
    default_timeout=45
)
await tool_manager.async_init()
```

### Multiple Tool Execution

```python
# Execute tools concurrently
from argentic.core.protocol.tool import ToolCallRequest

# Define multiple tool calls
tool_calls = [
    ToolCallRequest(
        tool_id="web_scraper",
        arguments={"url": "https://example.com", "selector": ".content"}
    ),
    ToolCallRequest(
        tool_id="email_sender", 
        arguments={"recipient": "admin@company.com", "subject": "Scrape Complete"}
    ),
    ToolCallRequest(
        tool_id="file_writer",
        arguments={"filename": "scrape_results.txt", "content": "Data from scraping"}
    )
]

# Execute all tools concurrently
results, any_errors = await tool_manager.get_tool_results(tool_calls)

# Process results
for i, result in enumerate(results):
    original_request = tool_calls[i]
    
    if result.status == TaskStatus.SUCCESS:
        print(f"✅ {result.tool_name} completed: {result.result}")
    else:
        print(f"❌ {result.tool_name} failed: {result.error}")
        
if any_errors:
    print("⚠️ Some tools failed - check individual results")
```

## Tool Registration Flow

The ToolManager handles tool registration automatically:

```
1. Tool publishes RegisterToolMessage to register_topic
2. ToolManager receives message → generates unique tool_id
3. ToolManager subscribes to tool's result topic: {response_base}/{tool_id}
4. ToolManager publishes ToolRegisteredMessage to status_topic
5. Tool receives confirmation → subscribes to task topic: {call_base}/{tool_id}
6. Tool is ready for execution
```

## Tool Information Structure

Each registered tool has the following information:

```python
tool_info = {
    "id": "550e8400-e29b-41d4-a716-446655440000",  # Unique UUID
    "name": "Email Sender",                         # Human-readable name  
    "description": "Send emails to recipients...",  # Tool manual/description
    "parameters": '{"type": "object", ...}',        # JSON schema for parameters
    "registered_at": "2024-01-01T12:00:00Z",       # Registration timestamp
    "source_client_id": "tool_service_123"         # Client that registered the tool
}
```

## Error Handling

The ToolManager provides comprehensive error handling:

### Tool Not Found

```python
result = await tool_manager.execute_tool("nonexistent_tool", {})
# Returns TaskErrorMessage with error: "Tool 'nonexistent_tool' not found."
```

### Tool Timeout

```python
# Tool takes longer than default_timeout
result = await tool_manager.execute_tool("slow_tool", {"data": "large_dataset"})
# Returns TaskErrorMessage with status: TaskStatus.TIMEOUT
```

### Tool Execution Errors

```python
# Tool raises exception during execution
result = await tool_manager.execute_tool("failing_tool", {"invalid": "data"})
# Returns TaskErrorMessage with tool's error message
```

### Concurrent Execution Errors

```python
results, any_errors = await tool_manager.get_tool_results(tool_calls)

# Check for system-level errors
for result in results:
    if hasattr(result, 'error') and "System exception" in result.error:
        print(f"System error in {result.tool_name}: {result.error}")
```

## Message Types

### Registration Messages

#### RegisterToolMessage
```python
{
    "tool_name": "Email Sender",
    "tool_manual": "Send emails to specified recipients...",
    "tool_api": '{"type": "object", "properties": {...}}',
    "source": "tool_client_id"
}
```

#### ToolRegisteredMessage  
```python
{
    "tool_id": "550e8400-e29b-41d4-a716-446655440000",
    "tool_name": "Email Sender", 
    "source": "tool_manager_client_id"
}
```

### Execution Messages

#### TaskMessage
```python
{
    "task_id": "task_123",
    "tool_id": "550e8400-e29b-41d4-a716-446655440000",
    "tool_name": "Email Sender",
    "arguments": {"recipient": "user@example.com", "message": "Hello"},
    "source": "agent_client_id"
}
```

#### TaskResultMessage
```python
{
    "task_id": "task_123",
    "tool_id": "550e8400-e29b-41d4-a716-446655440000", 
    "tool_name": "Email Sender",
    "status": "SUCCESS",
    "result": "✅ Email sent successfully to user@example.com",
    "arguments": {"recipient": "user@example.com", "message": "Hello"},
    "source": "tool_client_id"
}
```

#### TaskErrorMessage
```python
{
    "task_id": "task_123",
    "tool_id": "550e8400-e29b-41d4-a716-446655440000",
    "tool_name": "Email Sender", 
    "status": "FAILED",
    "error": "SMTP connection failed: Connection refused",
    "traceback": "Traceback (most recent call last)...",  # If debug mode
    "arguments": {"recipient": "invalid@email", "message": "Hello"},
    "source": "tool_client_id"
}
```

## Performance Considerations

### Concurrent Execution

The ToolManager executes multiple tools concurrently using `asyncio.gather()`:

```python
# All tools execute in parallel
results, has_errors = await tool_manager.get_tool_results([
    ToolCallRequest(tool_id="tool1", arguments={}),
    ToolCallRequest(tool_id="tool2", arguments={}), 
    ToolCallRequest(tool_id="tool3", arguments={})
])
```

### Timeout Management

```python
# Configure timeouts per use case
tool_manager = ToolManager(
    messager=messager,
    default_timeout=120  # 2 minutes for long-running tools
)

# Fast tools
result = await tool_manager.execute_tool("quick_lookup", {"key": "value"})

# Long-running tools use the default timeout
result = await tool_manager.execute_tool("data_processing", {"dataset": "large_file.csv"})
```

### Memory Management

The ToolManager uses locks and proper cleanup:

```python
# Thread-safe tool registration and result handling
async with self._result_lock:
    # Safe concurrent access to tool registry and pending tasks
    self.tools_by_id[tool_id] = tool_info
    self._pending_tasks[task_id] = future
```

## Monitoring and Debugging

### Tool Registry Inspection

```python
# List all registered tools
tool_names = tool_manager.get_tool_names()
print(f"Available tools: {tool_names}")

# Get detailed tool information
for tool_id, tool_info in tool_manager.tools_by_id.items():
    print(f"Tool: {tool_info['name']} (ID: {tool_id})")
    print(f"  Description: {tool_info['description']}")
    print(f"  Registered: {tool_info['registered_at']}")
```

### LLM Prompt Generation

```python
# Generate tool descriptions for LLM
tool_descriptions = tool_manager.generate_tool_descriptions_for_prompt()
print("Tools available to LLM:")
print(tool_descriptions)

# Example output:
# [
#   {
#     "tool_id": "550e8400-e29b-41d4-a716-446655440000",
#     "name": "Email Sender",
#     "description": "Send emails to specified recipients...",
#     "parameters": {"type": "object", "properties": {...}}
#   }
# ]
```

### Debug Logging

```python
# Enable debug logging
tool_manager.set_log_level("DEBUG")

# Detailed execution logging
result = await tool_manager.execute_tool("debug_tool", {"test": "data"})
# Logs: tool lookup, task creation, MQTT publishing, result waiting, etc.
```

## Best Practices

### 1. Proper Initialization

```python
# Always call async_init() before using ToolManager
tool_manager = ToolManager(messager)
await tool_manager.async_init()  # Required!
```

### 2. Error Handling

```python
result = await tool_manager.execute_tool("risky_tool", arguments)

if result.status == TaskStatus.SUCCESS:
    # Process successful result
    return result.result
elif result.status == TaskStatus.TIMEOUT:
    # Handle timeout specifically
    print(f"Tool timed out: {result.error}")
    return "Operation took too long"
elif result.status in [TaskStatus.FAILED, TaskStatus.ERROR]:
    # Handle execution errors
    print(f"Tool failed: {result.error}")
    return f"Error: {result.error}"
```

### 3. Concurrent Tool Execution

```python
# Use get_tool_results() for multiple tools
requests = [ToolCallRequest(...) for _ in tools_to_execute]
results, any_errors = await tool_manager.get_tool_results(requests)

# Don't use individual execute_tool() calls in sequence
# This is inefficient:
# for request in requests:
#     result = await tool_manager.execute_tool(request.tool_id, request.arguments)
```

### 4. Resource Management

```python
# ToolManager doesn't need explicit cleanup, but tools should unregister
# when shutting down their services

# In your tool service shutdown:
await my_tool.unregister()
await messager.disconnect()
```

## See Also

- [BaseTool API](base-tool.md) - Creating custom tools
- [Agent API](agent.md) - Using ToolManager in agents  
- [Messaging Configuration](../messaging-configuration.md) - MQTT setup
- [Tool Examples](../../examples/) - Complete tool implementations 