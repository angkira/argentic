# AI Agent Documentation Guide

This guide explains how to use Argentic framework documentation optimized for AI coding agents.

## Overview

Argentic includes specialized documentation designed for AI coding agents (Cursor, Claude Code, GitHub Copilot, etc.) to quickly understand and implement solutions using the framework.

## Available Files

### `.cursorrules` (Root Directory)

**Purpose**: Compact reference automatically loaded by Cursor/Claude Code when working in the project.

**Content** (~830 lines):
- Framework overview
- Core components API
- Quick start patterns
- Configuration examples
- Best practices
- Common troubleshooting

**When to use**: Automatically loaded by Cursor-based AI agents. No action needed.

### `ARGENTIC_QUICKREF.md` (Root Directory)

**Purpose**: Extended reference for complex scenarios and detailed API documentation.

**Content** (~830 lines):
- Complete API reference for all classes
- All LLM provider configurations
- Advanced patterns (endless cycle, state management)
- Comprehensive examples
- Troubleshooting guide

**When to use**: 
- When `.cursorrules` doesn't provide enough detail
- When implementing complex multi-agent systems
- When developing custom tools
- When debugging issues

## For AI Agents

When you're an AI agent working with Argentic:

1. **Start with `.cursorrules`**: This file is automatically loaded and provides all essential patterns
2. **Use `ARGENTIC_QUICKREF.md`**: Reference this for detailed API signatures and advanced examples
3. **Check `examples/`**: Working code for single-agent, multi-agent, and custom tools
4. **Read `README.md`**: Installation, setup, and high-level architecture

## For Developers

### Using with Cursor/Claude Code

The `.cursorrules` file is automatically detected and loaded by Cursor when you open the project. No configuration needed.

### Using with Other AI Agents

For AI agents that don't auto-load `.cursorrules`:

1. Provide the content of `.cursorrules` in your context
2. Reference `ARGENTIC_QUICKREF.md` for detailed API info
3. Point to specific examples in `examples/` directory

### Updating Documentation

When updating the framework:

1. Update `.cursorrules` if core APIs or patterns change
2. Update `ARGENTIC_QUICKREF.md` with new features or detailed examples
3. Keep both files synchronized with actual code
4. Test with actual AI agents to ensure clarity

## Content Structure

### `.cursorrules` Sections

1. **Framework Overview** - What Argentic is and key features
2. **Core Components** - Agent, Messager, ToolManager, Supervisor, BaseTool
3. **Quick Start Patterns** - Single agent, custom tool, multi-agent
4. **Configuration** - config.yaml structure and environment variables
5. **Important Details** - Tool registration, async patterns, message protocol
6. **Common Patterns** - Setup, running, imports, testing

### `ARGENTIC_QUICKREF.md` Sections

1. **Installation & Setup** - Package installation, environment, MQTT
2. **Core API Reference** - Complete method signatures and parameters
3. **LLM Providers** - All providers with examples
4. **Tool Development** - Complete tool examples and best practices
5. **Multi-Agent Patterns** - Specialized workers, agent-specific tools
6. **Advanced Features** - Endless cycle, state management, dialogue logging
7. **Message Protocol** - All message classes and examples
8. **Examples** - Multiple working examples
9. **Troubleshooting** - Common issues and solutions

## Tips for AI Agents

When using Argentic framework:

1. **Always use async/await** - All operations are asynchronous
2. **Share ToolManager** - Use one instance for multi-agent systems
3. **Separate topics** - Each agent needs its own MQTT topic namespace
4. **Enable logging** - Use `enable_dialogue_logging=True` for debugging
5. **Handle cleanup** - Always disconnect messager in finally block
6. **Wait after registration** - Add `await asyncio.sleep(1)` after tool registration

## Examples Location

- **Single Agent**: `examples/single_agent_example.py`
- **Multi-Agent**: `examples/multi_agent_example.py`
- **Custom Tool**: `examples/note_creator_tool.py`, `examples/email_tool.py`
- **Tool Service**: `examples/secretary_tools_service.py`

## External Documentation

- **Full Documentation**: See `docs/` directory or website
- **API Documentation**: `docs/api/` for complete API reference
- **Configuration**: `docs/configuration.md`, `docs/messaging-configuration.md`
- **Advanced**: `docs/advanced-llm-configuration.md`

## Feedback

If you're an AI agent and found these docs insufficient:
- This is a learning opportunity for improvement
- The patterns and examples should cover 90% of use cases
- For edge cases, refer to source code or full documentation

If you're a developer maintaining this:
- Keep `.cursorrules` concise but complete
- Add working examples for new features
- Test with actual AI agents when possible

