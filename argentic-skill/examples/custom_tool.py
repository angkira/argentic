"""
Example: Custom Tool Development

This example demonstrates how to create a custom tool for Argentic agents.
"""

import asyncio
import json
from typing import Any
from pydantic import BaseModel, Field
from argentic.core.tools.tool_base import BaseTool
from argentic.core.messager.messager import Messager


# Step 1: Define input schema with Pydantic
class WeatherInput(BaseModel):
    city: str = Field(description="City name to get weather for")
    units: str = Field(default="celsius", description="Temperature units: celsius or fahrenheit")


# Step 2: Implement tool class
class WeatherTool(BaseTool):
    """Tool for getting weather information."""

    def __init__(self, messager: Messager):
        super().__init__(
            name="weather_tool",
            manual="Get current weather for a city. Provide 'city' name and optionally 'units' (celsius/fahrenheit).",
            api=json.dumps(WeatherInput.model_json_schema()),
            argument_schema=WeatherInput,
            messager=messager,
        )

    async def _execute(self, **kwargs) -> Any:
        """
        Implement weather lookup logic.
        In real implementation, this would call a weather API.
        """
        city = kwargs["city"]
        units = kwargs.get("units", "celsius")

        # Simulate API call
        await asyncio.sleep(0.5)

        # Mock response
        temp = 22 if units == "celsius" else 72
        symbol = "°C" if units == "celsius" else "°F"

        result = f"Weather in {city}: {temp}{symbol}, partly cloudy"
        return result


# Step 3: Register and use tool
async def main():
    # Setup
    messager = Messager(broker_address="localhost", port=1883)
    await messager.connect()

    # Register tool
    tool = WeatherTool(messager)
    await tool.register(
        registration_topic="agent/tools/register",
        status_topic="agent/status/info",
        call_topic_base="agent/tools/call",
        response_topic_base="agent/tools/response",
    )

    print("Tool registered successfully!")
    print(f"Tool name: {tool.name}")
    print(f"Tool ID: {tool.id if hasattr(tool, 'id') else 'Pending...'}")

    # Wait for registration
    await asyncio.sleep(2)

    # Tool is now ready to be used by agents
    print("\nTool is ready. Create an agent to use it.")

    # Cleanup
    await messager.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
