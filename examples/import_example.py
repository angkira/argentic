#!/usr/bin/env python
"""Example demonstrating the simplified imports in Argentic"""

# Option 1: Import directly from the main package
from argentic import Agent, Messager, LLMFactory


# Create a simple example
async def main():
    """Simple example of using Argentic with simplified imports"""
    # Configure messager
    messager = Messager(
        protocol="mqtt", broker_address="localhost", port=1883, client_id="example_client"
    )

    # Use the LLMFactory to create a provider
    config = {"llm": {"provider": "ollama", "ollama_model_name": "gemma3:2b"}}
    llm_provider = LLMFactory.create(config, messager)

    # Create the agent
    agent = Agent(
        llm=llm_provider,
        messager=messager,
    )

    print("Successfully created Agent, Messager, and LLM provider using simplified imports")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
