#!/usr/bin/env python3

import asyncio
import os
import json
import yaml
import sys
import time
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage

from argentic.core.agent.agent import Agent
from argentic.core.graph.state import AgentState
from argentic.core.graph.supervisor import Supervisor
from argentic.core.llm.providers.google_gemini import GoogleGeminiProvider
from argentic.core.messager.messager import Messager
from argentic.core.protocol.tool import RegisterToolMessage
from argentic.core.tools.tool_manager import ToolManager


def log_with_flush(message):
    """Log with immediate flush to check for buffering issues"""
    print(f"⏰ {time.time():.3f}: {message}")
    sys.stdout.flush()  # Force immediate output


async def main():
    """Test the original multi_agent_example with forced log flushing"""
    # Load environment variables from .env file
    load_dotenv()

    # Create background logger to test for blocking
    async def background_logger():
        while True:
            await asyncio.sleep(1)
            log_with_flush("Background task still running...")

    # Load LLM configuration from a minimal config file
    llm_config_path = os.path.join(os.path.dirname(__file__), "config_gemini.yaml")
    with open(llm_config_path, "r") as f:
        llm_full_config = yaml.safe_load(f)
    if llm_full_config is None:
        llm_full_config = {}
    llm_config = llm_full_config.get("llm", {})
    if not isinstance(llm_config, dict):
        print(
            "Warning: 'llm' configuration not found or not a dictionary in config_gemini.yaml. Using defaults."
        )
        llm_config = {}

    # Load Messaging configuration from a minimal config file
    messaging_config_path = os.path.join(os.path.dirname(__file__), "config_messaging.yaml")
    with open(messaging_config_path, "r") as f:
        messaging_full_config = yaml.safe_load(f)
    if messaging_full_config is None:
        messaging_full_config = {}
    messaging_config_data = messaging_full_config.get("messaging", {})
    if not isinstance(messaging_config_data, dict):
        print(
            "Warning: 'messaging' configuration not found or not a dictionary in config_messaging.yaml. Using defaults."
        )
        messaging_config_data = {}

    # Initialize LLM (Google Gemini) using config (api_key will be read internally)
    llm = GoogleGeminiProvider(config=llm_config)

    # 1. Initialize Messager and ToolManager
    # Extract individual parameters from messaging_config_data
    broker_address = messaging_config_data.get("broker_address", "localhost")
    port = messaging_config_data.get("port", 1883)
    client_id = messaging_config_data.get("client_id", "")
    username = messaging_config_data.get("username")
    password = messaging_config_data.get("password")
    keepalive = messaging_config_data.get("keepalive", 60)

    messager = Messager(
        broker_address=broker_address,
        port=port,
        client_id=client_id,
        username=username,
        password=password,
        keepalive=keepalive,
    )
    tool_manager = ToolManager(messager)
    await messager.connect()
    await tool_manager.async_init()

    log_with_flush("--- Running Multi-Agent Example with Background Monitoring ---")

    # Start background monitoring
    logger_task = asyncio.create_task(background_logger())

    try:
        # 2. Initialize Agents (EXACT copy from multi_agent_example.py)
        # Supervisor Agent
        supervisor = Supervisor(
            llm=llm,
            messager=messager,
            tool_manager=tool_manager,
            role="supervisor",
            system_prompt="Route tasks: 'researcher' for info/data queries, 'coder' for programming. Be direct.",
            graph_id="my_multi_agent_system",
        )

        # Worker Agents
        researcher_prompt = "Research and provide factual information. No fluff."
        researcher = Agent(
            llm=llm,
            messager=messager,
            role="researcher",
            system_prompt=researcher_prompt,
            graph_id="my_multi_agent_system",
            expected_output_format="text",
        )
        await researcher.async_init()

        # Coder Agent
        coder_prompt = "Write code. Include brief comments only when necessary."
        coder = Agent(
            llm=llm,
            messager=messager,
            role="coder",
            system_prompt=coder_prompt,
            graph_id="my_multi_agent_system",
            expected_output_format="text",
        )
        await coder.async_init()

        # 3. Add Workers to the Supervisor
        supervisor.add_agent(researcher)
        supervisor.add_agent(coder)

        # 4. Compile the Supervisor's graph
        supervisor.compile()

        # Initial state for the graph
        initial_state: AgentState = {
            "messages": [HumanMessage(content="Research the current status of quantum computing.")],
            "next": None,
        }

        log_with_flush("Starting LangGraph streaming...")

        # Use supervisor.runnable to stream events
        if supervisor.runnable:
            async for event in supervisor.runnable.astream(initial_state):
                for key, value in event.items():
                    if key == "supervisor" or key == "researcher" or key == "coder":
                        # Print only relevant agent steps
                        log_with_flush(f"Node: {key}")
                        if value and "messages" in value and value["messages"]:
                            for msg in value["messages"]:
                                content = getattr(
                                    msg,
                                    "content",
                                    getattr(msg, "raw_content", str(msg)) or "No content",
                                )[:150]
                                log_with_flush(
                                    f"  Message Type: {type(msg).__name__}, Content: {content}..."
                                )
                    elif key == "END":
                        final_messages = value.get("messages", [])
                        for msg in final_messages:
                            content = getattr(
                                msg,
                                "content",
                                getattr(msg, "raw_content", str(msg)) or "No content",
                            )
                            log_with_flush(f"Final Answer: {content}")
                log_with_flush("---")
        else:
            log_with_flush("Error: Supervisor runnable is not compiled.")

    finally:
        # Stop background monitoring
        logger_task.cancel()
        try:
            await logger_task
        except asyncio.CancelledError:
            pass

    await messager.disconnect()
    log_with_flush("Multi-agent example completed!")


if __name__ == "__main__":
    asyncio.run(main())
