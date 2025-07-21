import asyncio
import os
import json
import yaml
from dotenv import load_dotenv  # Import load_dotenv

from langchain_core.messages import HumanMessage

from argentic.core.agent.agent import Agent
from argentic.core.graph.state import AgentState
from argentic.core.graph.supervisor import Supervisor
from argentic.core.llm.providers.google_gemini import GoogleGeminiProvider
from argentic.core.messager.messager import Messager
from argentic.core.protocol.tool import RegisterToolMessage
from argentic.core.tools.tool_manager import ToolManager  # Corrected import path


async def main():
    # Load environment variables from .env file
    load_dotenv()

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
    llm = GoogleGeminiProvider(config=llm_config)  # Corrected instantiation

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
    tool_manager = ToolManager(messager)  # Initialize ToolManager here
    await messager.connect()
    await tool_manager.async_init()

    print("\n--- Running Multi-Agent Example ---")

    # 2. Initialize Agents
    # Supervisor Agent
    supervisor = Supervisor(
        llm=llm,
        messager=messager,
        tool_manager=tool_manager,  # Pass tool_manager
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
        expected_output_format="text",  # Set for text output
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
        expected_output_format="text",  # Set for text output
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

    # Use supervisor.runnable to stream events
    if supervisor.runnable:
        async for event in supervisor.runnable.astream(initial_state):
            for key, value in event.items():
                if key == "supervisor" or key == "researcher" or key == "coder":
                    # Print only relevant agent steps
                    print(f"Node: {key}")
                    if value and "messages" in value and value["messages"]:
                        for msg in value["messages"]:
                            print(
                                f"  Message Type: {type(msg).__name__}, Content: {getattr(msg, 'content', getattr(msg, 'raw_content', str(msg)) or 'No content')[:150]}..."
                            )
                elif key == "END":
                    final_messages = value.get("messages", [])
                    for msg in final_messages:
                        print(
                            f"Final Answer: {getattr(msg, 'content', getattr(msg, 'raw_content', str(msg)) or 'No content')}"
                        )
            print("---")
    else:
        print("Error: Supervisor runnable is not compiled.")

    await messager.disconnect()


if __name__ == "__main__":
    # from dotenv import load_dotenv
    # load_dotenv()
    asyncio.run(main())
