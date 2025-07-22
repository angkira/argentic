import asyncio
import os
import yaml
from dotenv import load_dotenv

from argentic.core.agent.agent import Agent
from argentic.core.llm.providers.google_gemini import GoogleGeminiProvider
from argentic.core.messager.messager import Messager
from argentic.core.tools.tool_manager import ToolManager
from argentic.core.protocol.message import AgentTaskMessage, AgentTaskResultMessage


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
    llm = GoogleGeminiProvider(config=llm_config)

    # Initialize Messager using config
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

    # Initialize ToolManager (required even if no tools are used)
    tool_manager = ToolManager(messager)
    await messager.connect()
    await tool_manager.async_init()

    # Create a single agent with proper parameters
    system_prompt = (
        "You are a helpful AI assistant. Provide clear, accurate, and informative responses. "
        "You can answer questions, explain concepts, and help with various tasks."
    )

    agent = Agent(
        llm=llm,
        messager=messager,
        tool_manager=tool_manager,
        role="research_agent",
        description="General-purpose research and information agent capable of answering questions, providing explanations, and helping with various tasks using available knowledge.",
        system_prompt=system_prompt,
        expected_output_format="text",
        enable_dialogue_logging=True,  # Enable dialogue logging
    )
    await agent.async_init()

    print("\n--- Single Agent Mode Examples ---")
    print("1. Direct query method")
    print("2. Messaging task system")
    print("====================================")

    # Method 1: Direct query (existing functionality)
    print("\n1️⃣ DIRECT QUERY METHOD:")
    question = "What is the capital of France?"
    print(f"Question: {question}")
    answer = await agent.query(question)
    print(f"Answer: {answer}")

    question = "Tell me a short story about a brave knight and a dragon."
    print(f"\nQuestion: {question}")
    answer = await agent.query(question)
    print(f"Answer: {answer}")

    # Method 2: Task messaging system (new functionality)
    print("\n\n2️⃣ MESSAGING TASK SYSTEM:")

    # Set up result listener
    results_received = {}
    result_event = asyncio.Event()

    async def handle_result(message):
        """Handle task results from the agent."""
        if hasattr(message, "task_id"):
            results_received[message.task_id] = {
                "success": message.success,
                "result": message.result,
                "error": getattr(message, "error", None),
            }
            result_event.set()

    # Subscribe to agent results
    result_topic = f"agent/{agent.role}/results"
    await messager.subscribe(result_topic, handle_result, AgentTaskResultMessage)

    # Send a task via messaging
    task_msg = AgentTaskMessage(
        task="Explain quantum computing in simple terms suitable for a high school student.",
        sender_id="single_agent_demo",
    )

    task_topic = f"agent/{agent.role}/tasks"
    await messager.publish(task_topic, task_msg)
    print(f"📤 Sent task via messaging to '{task_topic}'")
    print(f"Task ID: {task_msg.task_id}")

    # Wait for result
    print("⏳ Waiting for result...")
    try:
        await asyncio.wait_for(result_event.wait(), timeout=30.0)

        if task_msg.task_id in results_received:
            result_info = results_received[task_msg.task_id]
            if result_info["success"]:
                print(f"✅ Task completed successfully!")
                print(f"📄 Result: {result_info['result'][:300]}...")
            else:
                print(f"❌ Task failed: {result_info['error']}")
        else:
            print("⚠️ No result received")

    except asyncio.TimeoutError:
        print("⏱️ Timeout waiting for result")

    print("\n🏁 Single Agent Demo Complete!")
    print("Both direct queries and messaging tasks work in single-agent mode.")

    # Print dialogue summary
    if agent.enable_dialogue_logging:
        agent.print_dialogue_summary()

    await messager.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
