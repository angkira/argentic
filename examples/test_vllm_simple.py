#!/usr/bin/env python3
"""Simple test for vLLM provider - text-only."""

import asyncio

from argentic import Messager
from argentic.core.llm.llm_factory import LLMFactory


async def main():
    print("=== vLLM Provider Simple Test ===\n")

    # Setup
    messager = Messager(broker_address="localhost", port=1883)
    await messager.connect()
    print("✓ Connected to MQTT\n")

    # Create vLLM provider
    config = {
        "llm": {
            "provider": "vllm",
            "vllm_base_url": "http://localhost:8000/v1",
            "vllm_model_name": "google/gemma-3n-E4B-it",
            "temperature": 0.7,
            "max_tokens": 100,
        }
    }

    print("Creating vLLM provider...")
    print(f"Server: {config['llm']['vllm_base_url']}")
    print(f"Model: {config['llm']['vllm_model_name']}")
    print()
    print("NOTE: vLLM server must be running:")
    print("  vllm serve google/gemma-3n-E4B-it --port 8000")
    print()

    try:
        llm = LLMFactory.create(config, messager)
        print(f"✓ Provider created: {llm.get_model_name()}\n")

        # Test chat
        from argentic.core.protocol.chat_message import UserMessage

        messages = [UserMessage(content="What is the capital of France? Answer briefly.")]

        print("Sending query...")
        response = await llm.achat(messages)

        print(f"Response: {response.content}")
        print(f"Finish reason: {response.finish_reason}\n")

        print("✓ Test passed!")

    except Exception as e:
        print(f"✗ Error: {e}\n")
        import traceback

        traceback.print_exc()

    await messager.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
