"""Integration test: ZeroMQ + Real LLM (requires API key).

This test validates the complete stack:
- ZeroMQ embedded proxy
- Messager over ZeroMQ driver  
- Agent with real LLM API

Requires: GOOGLE_GEMINI_API_KEY or GEMINI_API_KEY in environment
"""

import asyncio
import os

import pytest

try:
    import zmq  # noqa: F401

    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False

# Check for API key
api_key = os.getenv("GOOGLE_GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    pytest.skip("Gemini API key not found in environment", allow_module_level=True)

if not ZMQ_AVAILABLE:
    pytest.skip("pyzmq not installed", allow_module_level=True)

from argentic.core.agent.agent import Agent
from argentic.core.llm.llm_factory import LLMFactory
from argentic.core.messager.messager import Messager
from argentic.core.messager.protocols import MessagerProtocol


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.slow
async def test_zeromq_with_gemini():
    """Test complete stack: ZeroMQ + Gemini LLM + Agent."""
    # Create LLM
    config = {
        "llm": {
            "provider": "google_gemini",
            "google_gemini_model_name": "gemini-2.0-flash",
            "google_gemini_api_key": api_key,
            "google_gemini_parameters": {
                "temperature": 0.7,
                "max_output_tokens": 256,
            },
        }
    }
    llm = LLMFactory.create(config)

    # Create ZeroMQ messager with embedded proxy
    messager = Messager(
        protocol=MessagerProtocol.ZEROMQ,
        broker_address="127.0.0.1",
        port=19555,
        backend_port=19556,
        start_proxy=True,
        proxy_mode="embedded",
    )
    await messager.connect()
    await asyncio.sleep(0.3)

    # Create agent
    agent = Agent(
        llm=llm,
        messager=messager,
        role="test_agent",
        system_prompt="You are a helpful AI. Keep responses under 50 words.",
    )
    await agent.async_init()

    # Test query (simple math to avoid API issues)
    try:
        response = await agent.query("What is 5+3? Answer with just the number.", user_id="test")
        
        # Verify we got a response
        assert response is not None
        assert len(response) > 0
        print(f"✅ Agent response: {response[:100]}")
        
    finally:
        await messager.stop()


if __name__ == "__main__":
    # Allow running directly for manual testing
    asyncio.run(test_zeromq_with_gemini())
