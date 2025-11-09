# Argentic Framework Context

Quick reference for Claude Code when working with this project.

## Framework: Argentic v0.11.x

Python microframework for AI agents with async MQTT messaging.

## Core Patterns

### 1. Single Agent
```python
from argentic import Agent, Messager, LLMFactory
from argentic.core.tools import ToolManager

llm = LLMFactory.create_from_config(config["llm"])
messager = Messager(broker_address="localhost", port=1883)
await messager.connect()

tool_manager = ToolManager(messager)
await tool_manager.async_init()

agent = Agent(llm, messager, tool_manager, role="assistant")
await agent.async_init()
response = await agent.query("question")
```

### 2. Custom Tool
```python
from argentic.core.tools.tool_base import BaseTool
from pydantic import BaseModel, Field

class MyInput(BaseModel):
    param: str = Field(description="Param description")

class MyTool(BaseTool):
    def __init__(self, messager):
        super().__init__(
            name="my_tool",
            manual="Tool description",
            api=json.dumps(MyInput.model_json_schema()),
            argument_schema=MyInput,
            messager=messager
        )
    
    async def _execute(self, **kwargs):
        return result
```

### 3. Multi-Agent
```python
from argentic.core.graph.supervisor import Supervisor

# ONE shared ToolManager!
tool_manager = ToolManager(messager)

researcher = Agent(llm, messager, tool_manager, 
                   role="researcher",
                   register_topic="agent/researcher/tools/register")
analyst = Agent(llm, messager, tool_manager,
                role="analyst",
                register_topic="agent/analyst/tools/register")

supervisor = Supervisor(llm=llm, messager=messager)
supervisor.add_agent(researcher)
supervisor.add_agent(analyst)
```

## Key Rules

✅ Always async/await
✅ Share ToolManager for multi-agent
✅ Separate MQTT topics per agent
✅ Enable dialogue_logging for debug
✅ Pydantic for tool validation

## Quick Commands

```bash
# Run agent
python -m argentic agent --config-path config.yaml

# MQTT broker
docker run -d -p 1883:1883 eclipse-mosquitto:2.0
```

## Full Documentation

See: `ARGENTIC_QUICKREF.md` for complete reference

