"""
Example: Multi-Agent System with Supervisor

This example shows how to create a multi-agent system where a supervisor
coordinates multiple specialized agents.
"""

import asyncio
from argentic import Agent, Messager, LLMFactory
from argentic.core.tools import ToolManager
from argentic.core.graph.supervisor import Supervisor
import yaml
from dotenv import load_dotenv


async def main():
    # Load environment and config
    load_dotenv()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # Initialize LLM and messaging
    llm = LLMFactory.create_from_config(config["llm"])
    messager = Messager(
        broker_address=config["messaging"]["broker_address"], port=config["messaging"]["port"]
    )
    await messager.connect()

    # IMPORTANT: One shared ToolManager for all agents
    tool_manager = ToolManager(messager)
    await tool_manager.async_init()

    # Create specialized agents
    researcher = Agent(
        llm=llm,
        messager=messager,
        tool_manager=tool_manager,
        role="researcher",
        description="Research and information gathering specialist",
        system_prompt="You are a researcher. Find and synthesize information.",
        expected_output_format="text",
        # Separate MQTT topics for this agent
        register_topic="agent/researcher/tools/register",
        tool_call_topic_base="agent/researcher/tools/call",
        tool_response_topic_base="agent/researcher/tools/response",
        status_topic="agent/researcher/status/info",
        graph_id="multi_agent_demo",
        enable_dialogue_logging=True,
    )
    await researcher.async_init()

    analyst = Agent(
        llm=llm,
        messager=messager,
        tool_manager=tool_manager,
        role="analyst",
        description="Data analysis and insights specialist",
        system_prompt="You are an analyst. Analyze data and provide insights.",
        expected_output_format="text",
        # Separate MQTT topics for this agent
        register_topic="agent/analyst/tools/register",
        tool_call_topic_base="agent/analyst/tools/call",
        tool_response_topic_base="agent/analyst/tools/response",
        status_topic="agent/analyst/status/info",
        graph_id="multi_agent_demo",
        enable_dialogue_logging=True,
    )
    await analyst.async_init()

    # Create supervisor
    supervisor = Supervisor(
        llm=llm,
        messager=messager,
        role="supervisor",
        system_prompt=(
            "You are a supervisor. Route tasks to the most appropriate agent: "
            "researcher for finding information, analyst for analyzing data."
        ),
        enable_dialogue_logging=True,
    )

    # Register agents with supervisor
    supervisor.add_agent(researcher)
    supervisor.add_agent(analyst)
    await supervisor.async_init()

    # Execute multi-agent workflow
    print("\n🚀 Starting multi-agent workflow...")

    workflow_complete = asyncio.Event()
    result_data = {}

    def completion_callback(task_id, success, result="", error=""):
        """Called when workflow completes."""
        result_data["success"] = success
        result_data["result"] = result
        result_data["error"] = error
        workflow_complete.set()

    # Start task
    task = (
        "Research the latest trends in artificial intelligence, "
        "then analyze the key findings and provide insights."
    )

    task_id = await supervisor.start_task(task, completion_callback)
    print(f"Task ID: {task_id}")

    # Wait for completion
    try:
        await asyncio.wait_for(workflow_complete.wait(), timeout=180)

        if result_data["success"]:
            print("\n✅ Workflow completed successfully!")
            print(f"\nResult:\n{result_data['result']}")
        else:
            print("\n❌ Workflow failed!")
            print(f"Error: {result_data['error']}")

    except asyncio.TimeoutError:
        print(f"\n⏱️ Workflow timeout! Task {task_id} did not complete.")

    # Print dialogue summaries
    print("\n" + "=" * 80)
    print("DIALOGUE SUMMARY")
    print("=" * 80)

    print("\n👨‍💼 SUPERVISOR:")
    supervisor.print_dialogue_summary()

    print("\n🔬 RESEARCHER:")
    researcher.print_dialogue_summary()

    print("\n📊 ANALYST:")
    analyst.print_dialogue_summary()

    # Cleanup
    await messager.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
