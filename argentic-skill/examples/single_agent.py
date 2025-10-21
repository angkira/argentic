"""
Example: Single Agent with Direct Query

This example shows how to create a basic AI agent using Argentic.
"""

import asyncio
from argentic import Agent, Messager, LLMFactory
from argentic.core.tools import ToolManager
import yaml
from dotenv import load_dotenv


async def main():
    # Load environment variables
    load_dotenv()

    # Load configuration
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # Initialize LLM
    llm = LLMFactory.create_from_config(config["llm"])

    # Setup MQTT messaging
    messager = Messager(
        broker_address=config["messaging"]["broker_address"], port=config["messaging"]["port"]
    )
    await messager.connect()

    # Initialize ToolManager
    tool_manager = ToolManager(messager)
    await tool_manager.async_init()

    # Create Agent
    agent = Agent(
        llm=llm,
        messager=messager,
        tool_manager=tool_manager,
        role="assistant",
        system_prompt="You are a helpful AI assistant.",
        enable_dialogue_logging=True,
    )
    await agent.async_init()

    # Use the agent
    questions = [
        "What is the capital of France?",
        "Explain async/await in Python",
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        response = await agent.query(question)
        print(f"Answer: {response}")

    # Print dialogue summary
    agent.print_dialogue_summary()

    # Cleanup
    await messager.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
