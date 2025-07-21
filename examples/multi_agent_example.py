import asyncio
import os
import json
import yaml
import logging
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage

from argentic.core.agent.agent import Agent
from argentic.core.graph.state import AgentState
from argentic.core.graph.supervisor import Supervisor
from argentic.core.llm.providers.google_gemini import GoogleGeminiProvider
from argentic.core.messager.messager import Messager
from argentic.core.tools.tool_manager import ToolManager

from email_tool import EmailTool
from note_creator_tool import NoteCreatorTool


async def main():
    # Configure logging FIRST - before any components are initialized
    # Suppress noisy logs from infrastructure components
    logging.getLogger("tool_manager").setLevel(logging.ERROR)
    logging.getLogger("messager").setLevel(logging.ERROR)
    logging.getLogger("mqtt_driver").setLevel(logging.ERROR)
    logging.getLogger("google_gemini").setLevel(logging.ERROR)
    logging.getLogger("email_tool").setLevel(logging.ERROR)
    logging.getLogger("note_creator").setLevel(logging.ERROR)

    # Keep only agent and supervisor logs visible
    logging.getLogger("agent").setLevel(logging.INFO)
    logging.getLogger("supervisor").setLevel(logging.INFO)

    # Load environment variables
    load_dotenv()

    print("🔧 Logging configured - showing only agent/supervisor logs")

    # Load LLM configuration
    llm_config_path = os.path.join(os.path.dirname(__file__), "config_gemini.yaml")
    with open(llm_config_path, "r") as f:
        llm_full_config = yaml.safe_load(f)
    if llm_full_config is None:
        llm_full_config = {}
    llm_config = llm_full_config.get("llm", {})
    if not isinstance(llm_config, dict):
        print("Warning: LLM config not found. Using defaults.")
        llm_config = {}

    # Load Messaging configuration
    messaging_config_path = os.path.join(os.path.dirname(__file__), "config_messaging.yaml")
    with open(messaging_config_path, "r") as f:
        messaging_full_config = yaml.safe_load(f)
    if messaging_full_config is None:
        messaging_full_config = {}
    messaging_config_data = messaging_full_config.get("messaging", {})
    if not isinstance(messaging_config_data, dict):
        print("Warning: Messaging config not found. Using defaults.")
        messaging_config_data = {}

    # Initialize LLM
    llm = GoogleGeminiProvider(config=llm_config)

    # Initialize Messager and ToolManager
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

    # Register tools directly with the tool manager
    email_tool = EmailTool(messager=messager)
    note_tool = NoteCreatorTool(messager=messager)

    # Register tools
    register_topic = "agent/tools/register"
    call_topic_base = "agent/tools/call"
    response_topic_base = "agent/tools/response"
    status_topic = "agent/status/info"

    await email_tool.register(register_topic, status_topic, call_topic_base, response_topic_base)
    await note_tool.register(register_topic, status_topic, call_topic_base, response_topic_base)

    # Wait for tools to register
    await asyncio.sleep(2)

    print("\n🚀 Enhanced Multi-Agent Example")
    print("================================")

    # Initialize Supervisor with clear workflow prompt
    supervisor = Supervisor(
        llm=llm,
        messager=messager,
        tool_manager=tool_manager,
        role="supervisor",
        system_prompt=(
            "Route workflow: "
            "1. Research requests → 'researcher' "
            "2. Research results → 'secretary' to save/email "
            "3. Secretary tool confirmations → end "
            "No conversation. Just route."
        ),
        graph_id="enhanced_multi_agent_system",
    )

    # Create Researcher Agent - focused and complete
    researcher_prompt = (
        "You are a researcher. Provide information and analysis ONLY. "
        "Do NOT use any tools - just provide research content. "
        "Create complete report using format: TITLE, FINDINGS, CONCLUSION. "
        "Use your knowledge to create comprehensive research. No tools needed."
    )
    researcher = Agent(
        llm=llm,
        messager=messager,
        tool_manager=tool_manager,  # Share the tool manager
        role="researcher",
        system_prompt=researcher_prompt,
        graph_id="enhanced_multi_agent_system",
        expected_output_format="text",
    )
    await researcher.async_init()

    # Create Secretary Agent - handles tools explicitly
    secretary_prompt = (
        "Execute tools. No talk. "
        "Given research data: "
        "1. note_creator_tool - save file "
        "2. email_tool - send email "
        "Use tools. Confirm done."
    )
    secretary = Agent(
        llm=llm,
        messager=messager,
        tool_manager=tool_manager,  # Share the tool manager
        role="secretary",
        system_prompt=secretary_prompt,
        graph_id="enhanced_multi_agent_system",
        expected_output_format="text",
    )
    await secretary.async_init()

    # Add agents to supervisor
    supervisor.add_agent(researcher)
    supervisor.add_agent(secretary)

    # Compile the graph
    supervisor.compile()

    # Task: Research and document findings
    initial_state: AgentState = {
        "messages": [
            HumanMessage(
                content="Research the current status of quantum computing breakthroughs in 2024. "
                "Save a summary report and email it to john.doe@company.com with subject 'Quantum Computing Update'."
            )
        ],
        "next": None,
    }

    # Execute and stream results
    if supervisor.runnable:
        async for event in supervisor.runnable.astream(initial_state):
            for key, value in event.items():
                if key in ["supervisor", "researcher", "secretary"]:
                    print(f"\n📋 {key.upper()} WORKING...")
                    if value and "messages" in value and value["messages"]:
                        for msg in value["messages"]:
                            content = getattr(
                                msg,
                                "content",
                                getattr(msg, "raw_content", str(msg)) or "No content",
                            )
                            # Show truncated content for readability
                            if len(content) > 150:
                                content = content[:150] + "..."
                            print(f"   💬 {content}")
                elif key == "END":
                    final_messages = value.get("messages", [])
                    print(f"\n✅ TASK COMPLETED")
                    for msg in final_messages:
                        content = getattr(
                            msg, "content", getattr(msg, "raw_content", str(msg)) or "No content"
                        )
                        print(f"   🎯 {content}")
            print("   " + "-" * 40)
    else:
        print("❌ Error: Supervisor runnable not compiled.")

    await messager.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
