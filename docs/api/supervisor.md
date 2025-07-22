# Supervisor API Reference

The `Supervisor` class coordinates multi-agent workflows using pure messaging-based communication with advanced context management for endless cycle operation.

## Class Definition

```python
class Supervisor:
    def __init__(
        self,
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

## Parameters

### Core Parameters

#### `llm: ModelProvider` (required)
The LLM provider for making routing decisions.

#### `messager: Messager` (required) 
The messaging system for coordinating agents.

#### `log_level: Union[str, LogLevel] = LogLevel.INFO`
Logging level for supervisor operations.

#### `role: str = "supervisor"`
Unique identifier for the supervisor.

#### `system_prompt: Optional[str] = None`
Custom system prompt for routing decisions. Appended to default supervisor prompt.

#### `llm_config: Optional[Dict[str, Any]] = None`
Provider-specific LLM configuration parameters.

### Endless Cycle Management

#### `enable_dialogue_logging: bool = False`
Enable real-time dialogue logging for monitoring routing decisions.

#### `max_task_history_items: int = 10`
Maximum task history entries per workflow before truncation.

#### `max_dialogue_history_items: int = 50`
Maximum dialogue entries before cleanup.

#### `context_cleanup_threshold: int = 100`
Number of tasks processed before triggering deep cleanup.

## Methods

### Core Methods

#### `async def async_init(self) -> None`
Initialize supervisor messaging subscriptions.

**Must be called** after supervisor creation and before use.

```python
supervisor = Supervisor(llm=llm, messager=messager)
await supervisor.async_init()  # Required
```

#### `def add_agent(self, agent) -> None`
Register an agent with the supervisor.

**Parameters:**
- `agent`: Agent instance with `role` and `description` attributes

**Example:**
```python
researcher = Agent(llm=llm, messager=messager, role="researcher", description="Research specialist")
supervisor.add_agent(researcher)
```

#### `async def start_task(self, task: str, completion_callback=None) -> str`
Start a new multi-agent workflow.

**Parameters:**
- `task`: Task description to process
- `completion_callback`: Optional callback function for task completion

**Returns:** Unique task ID for tracking

**Example:**
```python
task_id = await supervisor.start_task(
    "Research quantum computing and email results to john@example.com",
    completion_callback=my_callback
)
```

### Handler Methods

#### `async def handle_initial_task(self, message: BaseMessage) -> None`
Handle tasks published to supervisor/tasks topic.

#### `async def handle_agent_result(self, message: BaseMessage) -> None`
Handle results from agents and determine next routing step.

### Internal Methods

#### `async def _route_task(self, task_id: str, task_content: str) -> None`
Route task to appropriate agent using LLM decision.

#### `async def _continue_or_complete_task(self, task_id: str, latest_result: str) -> None`
Analyze workflow progress and determine next action.

#### `async def _complete_task(self, task_id: str, success: bool, final_result: str = "", error: str = "") -> None`
Complete a workflow task and call completion callback.

#### `async def _llm_route_decision(self, content: str) -> str`
Use LLM to determine routing decision based on content.

### Context Management

#### `def _truncate_task_history(self, task_info: Dict[str, Any]) -> Dict[str, Any]`
Truncate task history to prevent context overflow.

#### `def _cleanup_context_if_needed(self) -> None`
Clean up context to prevent memory overflow.

#### `def _deep_cleanup(self) -> None`
Perform deep cleanup of accumulated data.

### Dialogue Management

#### `def _log_dialogue(self, role: str, content: str, message_type: str = "routing") -> None`
Log a dialogue entry for monitoring.

#### `def _get_content_preview(self, content: str, max_length: int = 200) -> str`
Get preview of content for logging.

#### `def _print_dialogue_entry(self, entry: Dict[str, Any]) -> None`
Print formatted dialogue entry.

## Usage Examples

### Basic Supervisor Setup

```python
from argentic.core.graph.supervisor import Supervisor
from argentic.core.agent.agent import Agent
from argentic.core.llm.providers.google_gemini import GoogleGeminiProvider
from argentic.core.messager.messager import Messager

# Initialize components
llm = GoogleGeminiProvider(config={"google_gemini_model_name": "gemini-2.0-flash-lite"})
messager = Messager(broker_address="localhost", port=1883)
await messager.connect()

# Create supervisor
supervisor = Supervisor(
    llm=llm,
    messager=messager,
    role="workflow_coordinator",
    system_prompt="Prioritize efficiency and accuracy in task routing.",
    enable_dialogue_logging=True,
)
await supervisor.async_init()

# Add agents
researcher = Agent(llm=llm, messager=messager, role="researcher", 
                  description="Research and information gathering specialist")
secretary = Agent(llm=llm, messager=messager, role="secretary",
                 description="Document management and communication specialist")

supervisor.add_agent(researcher)
supervisor.add_agent(secretary)

# Start workflow
task_id = await supervisor.start_task(
    "Research AI trends and create a summary report, then email it to stakeholders"
)
```

### Production Supervisor for Endless Cycles

```python
# Production configuration with context management
supervisor = Supervisor(
    llm=llm,
    messager=messager,
    role="production_supervisor",
    
    # Context management for endless cycles
    max_task_history_items=5,        # Limit task history
    max_dialogue_history_items=30,   # Limit dialogue history
    context_cleanup_threshold=50,    # Frequent cleanup
    
    # Monitoring
    enable_dialogue_logging=False,   # Disable for performance
)

# Add multiple specialized agents
agents = [
    Agent(role="researcher", description="Research and analysis specialist"),
    Agent(role="writer", description="Content creation and documentation"),
    Agent(role="reviewer", description="Quality assurance and editing"),
    Agent(role="publisher", description="Distribution and communication"),
]

for agent in agents:
    await agent.async_init()
    supervisor.add_agent(agent)
```

### Completion Callback Example

```python
# Define completion callback
def workflow_completed(task_id: str, success: bool, result: str = "", error: str = ""):
    if success:
        print(f"✅ Workflow {task_id} completed successfully!")
        print(f"Result: {result[:200]}...")
        # Log to database, notify users, etc.
    else:
        print(f"❌ Workflow {task_id} failed: {error}")
        # Handle error, retry, notify administrators, etc.

# Start workflow with callback
task_id = await supervisor.start_task(
    "Multi-step analysis and reporting workflow",
    completion_callback=workflow_completed
)

# Wait for completion
workflow_complete = asyncio.Event()

def callback_wrapper(task_id: str, success: bool, result: str = "", error: str = ""):
    workflow_completed(task_id, success, result, error)
    workflow_complete.set()

task_id = await supervisor.start_task(task, callback_wrapper)
await workflow_complete.wait()  # Block until complete
```

### Debug and Monitoring Setup

```python
# Full monitoring for development
supervisor = Supervisor(
    llm=llm,
    messager=messager,
    role="debug_supervisor",
    
    # Full dialogue logging
    enable_dialogue_logging=True,
    
    # Generous limits for debugging
    max_task_history_items=20,
    max_dialogue_history_items=100,
    context_cleanup_threshold=200,
    
    # Custom routing instructions
    system_prompt="""
    For debugging purposes, always explain your routing decisions.
    Consider agent workload and specialization when routing tasks.
    """,
)
```

## Workflow Patterns

### Sequential Processing

```python
# Task that requires sequential agent processing
task = """
1. Research the topic thoroughly (researcher)
2. Create a structured document (writer) 
3. Review and edit the content (reviewer)
4. Distribute the final document (publisher)
"""

task_id = await supervisor.start_task(task)
```

### Parallel Processing

```python
# Task with parallel work streams
task = """
Research project with parallel workstreams:
- Team A: Research technical specifications
- Team B: Analyze market conditions  
- Team C: Prepare executive summary
Coordinate results and prepare final report.
"""

task_id = await supervisor.start_task(task)
```

### Conditional Routing

```python
# Task with conditional logic
task = """
Analyze the quarterly report:
1. If revenue is down, route to financial_analyst for deep dive
2. If customer satisfaction is low, route to customer_success for analysis  
3. Always route to executive_summary for final report
"""

task_id = await supervisor.start_task(task)
```

## Message Flow

The supervisor manages the following message flow:

```
1. User/System → supervisor/tasks → start_task()
2. Supervisor → agent/{role}/tasks → Agent processes
3. Agent → supervisor/results → handle_agent_result()
4. Supervisor decides: continue routing OR complete task
5. If continue: goto step 2 with next agent
6. If complete: call completion_callback()
```

## Topic Structure

### Supervisor Topics

- **`supervisor/tasks`** - Incoming workflow requests
- **`supervisor/results`** - Agent task results

### Agent Communication

- **`agent/{role}/tasks`** - Task assignments to specific agents
- **`agent/{role}/results`** - Results from specific agents

## Context Management

The supervisor implements sophisticated context management for endless cycles:

### Task History Truncation

```python
# Automatically truncates task history when it exceeds limits
task_info = {
    "original_task": "Long workflow...",
    "history": [
        {"agent": "researcher", "result": "Research complete..."},
        {"agent": "writer", "result": "Document created..."},
        # ... many more entries
    ]
}

# Supervisor keeps first 2 + last N entries with truncation marker
truncated = supervisor._truncate_task_history(task_info)
```

### Automatic Cleanup

```python
# Periodic cleanup triggered by task count
supervisor._total_tasks_processed = 100  # Triggers cleanup
supervisor._cleanup_context_if_needed()

# Cleans up:
# - Excess dialogue history
# - Orphaned completion callbacks
# - Accumulated task data
```

## Error Handling

The supervisor provides robust error handling:

### Agent Failures

```python
# Automatic error handling for agent failures
if not message.success:
    error_msg = message.error or "Unknown error"
    await self._complete_task(task_id, success=False, error=error_msg)
```

### Unknown Agents

```python
# Graceful handling of unknown agent routing
if agent_role not in self._agents:
    await self._complete_task(
        task_id, success=False, error=f"Unknown agent: {agent_role}"
    )
```

### LLM Failures

```python
# Fallback routing when LLM fails
try:
    decision = await self._llm_route_decision(content)
except Exception as e:
    self.logger.error(f"LLM routing error: {e}")
    decision = "__end__"  # Fail gracefully
```

## Performance Considerations

### For High Throughput

```python
supervisor = Supervisor(
    enable_dialogue_logging=False,   # Reduce overhead
    max_task_history_items=3,        # Minimal memory
    max_dialogue_history_items=10,   # Small dialogue buffer
    context_cleanup_threshold=25,    # Frequent cleanup
)
```

### For Complex Workflows

```python
supervisor = Supervisor(
    max_task_history_items=15,       # More context
    max_dialogue_history_items=100,  # Full dialogue tracking
    context_cleanup_threshold=100,   # Less frequent cleanup
    enable_dialogue_logging=True,    # Full monitoring
)
```

## Best Practices

1. **Agent Registration**: Always register agents before starting workflows
2. **Completion Callbacks**: Use callbacks for workflow completion handling
3. **Context Limits**: Set appropriate limits based on memory constraints
4. **Error Handling**: Implement robust completion callback error handling
5. **Monitoring**: Enable dialogue logging for development, disable for production
6. **Cleanup**: Monitor task processing count for memory management

## See Also

- [Agent API](agent.md) - Worker agent implementation
- [Messaging Configuration](../messaging-configuration.md) - MQTT setup
- [Multi-Agent Examples](../../examples/) - Complete workflow examples 