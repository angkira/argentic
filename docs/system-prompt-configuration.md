# System Prompt Configuration

The AI Agent supports customizable system prompts through the configuration file. This allows you to specialize your agent for different domains or use cases.

## Configuration

### Basic Setup

Add an `agent` section to your `config.yaml` file:

```yaml
agent:
  system_prompt: |
    Your custom system prompt here...
```

### Using Default System Prompt

If you don't specify a `system_prompt` in the config, or if you leave the `agent` section empty, the agent will use the default system prompt:

```yaml
agent:
  # No system_prompt specified - uses default
```

Or you can omit the `agent` section entirely.

## Examples

### 1. Software Development Assistant

```yaml
agent:
  system_prompt: |
    You are a highly capable AI assistant specialized in software development.
    You excel at code review, debugging, architecture design, and providing
    technical guidance. Always provide practical, actionable advice.

    When using tools, prefer code analysis and debugging tools.
    When answering directly, provide clear explanations with code examples.
```

### 2. Data Analysis Specialist

```yaml
agent:
  system_prompt: |
    You are a data analysis expert with deep knowledge of statistics,
    machine learning, and data visualization. You help users understand
    their data, identify patterns, and make data-driven decisions.

    Prioritize using data analysis tools when available.
    Provide clear explanations of statistical concepts and methodologies.
```

### 3. Customer Support Agent

```yaml
agent:
  system_prompt: |
    You are a helpful customer support agent. You are patient, empathetic,
    and focused on solving customer problems. Always maintain a professional
    and friendly tone.

    Use available tools to look up information and resolve issues.
    When you can't solve a problem, clearly explain next steps.
```

## Dynamic System Prompt Updates

You can also update the system prompt programmatically:

```python
# Update system prompt at runtime
agent.set_system_prompt("Your new system prompt...")

# Get current system prompt
current_prompt = agent.get_system_prompt()
```

## Important Notes

### Response Format Requirements

If you provide a custom system prompt, you must include the required response format instructions for the agent to work properly. The agent expects responses in specific JSON formats:

- `tool_call` - When using tools
- `direct` - When answering directly
- `tool_result` - When providing final answers after tool execution

### Backward Compatibility

Existing configurations without a system prompt will continue to work unchanged, using the default system prompt that includes all necessary formatting instructions.

### Configuration Validation

The agent will log whether it's using a custom or default system prompt:

```
INFO: Using custom system prompt from config
```

or

```
INFO: Using default system prompt
```

## Best Practices

1. **Keep It Focused**: Make your system prompt specific to your use case
2. **Include Context**: Specify the domain expertise and personality
3. **Tool Usage Guidance**: Provide guidance on when and how to use tools
4. **Response Style**: Define the tone and style of responses
5. **Test Thoroughly**: Test your custom prompt with various scenarios

## Example Files

- `config.example.yaml` - Full configuration with custom system prompt
- `config.simple.yaml` - Simple configuration using default system prompt
