import asyncio
import os
import sys
import yaml
import logging
from dotenv import load_dotenv

# Fix for local imports: add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Remove LangGraph imports - using pure messaging now
from argentic.core.agent.agent import Agent
from argentic.core.graph.supervisor import Supervisor
from argentic.core.llm.providers.google_gemini import GoogleGeminiProvider
from argentic.core.messager.messager import Messager
from argentic.core.tools.tool_manager import ToolManager
from argentic.core.protocol.message import AgentTaskMessage

try:
    from email_tool import EmailTool  # local import when running from examples/
    from note_creator_tool import NoteCreatorTool
except ImportError:  # fallback when examples is treated as package
    from examples.email_tool import EmailTool
    from examples.note_creator_tool import NoteCreatorTool


async def main():
    # Load environment variables
    load_dotenv()

    # --- Boilerplate setup for LLM, Messager, ToolManager ---
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

    llm = GoogleGeminiProvider(config=llm_config)
    messager = Messager(**messaging_config_data)
    await messager.connect()

    email_tool = EmailTool(messager=messager)
    note_tool = NoteCreatorTool(messager=messager)

    # Define general topics for supervisor and overall system visibility
    register_topic = "agent/tools/register"
    call_topic_base = "agent/tools/call"
    response_topic_base = "agent/tools/response"
    status_topic = "agent/status/info"

    # Define agent-specific topics for their internal ToolManagers to listen to
    researcher_register_topic = "agent/researcher/tools/register"
    researcher_call_topic_base = "agent/researcher/tools/call"
    researcher_response_topic_base = "agent/researcher/tools/response"
    researcher_status_topic = "agent/researcher/status/info"

    secretary_register_topic = "agent/secretary/tools/register"
    secretary_call_topic_base = "agent/secretary/tools/call"
    secretary_response_topic_base = "agent/secretary/tools/response"
    secretary_status_topic = "agent/secretary/status/info"

    # 1. DEFINE WORKER AGENTS WITH THEIR CAPABILITIES

    common_conversation_rules = (
        "IMPORTANT: Follow these rules for all responses. Be direct and concise. Do not use polite phrases, introductions, or closing remarks. "
        "Do not offer opinions or extra information. Only provide the direct output required by the task. "
        "Keep your conversation with the user short and concise. Without the polite phrases, corporate speaking and other language garbage"
        "No gratitude, no extra words, no extra comments, no extra explanations, no extra answers, no extra anything. Just professional and concise answers."
        "No ratings the responses, no 'thank you', no 'I appreciate it', no feedback. Just straight and clear communication without any emotions."
    )

    researcher_description = "Research and information gathering agent: Conducts thorough research, analyzes complex topics, gathers information from various sources, and creates comprehensive reports and summaries on any subject matter."

    researcher_prompt = (
        f"{common_conversation_rules}\n"
        f"You are the powerful **Researcher** AI agent. Your responsibilities: {researcher_description}\n"
        "Your capabilities:\n"
        "  • Browsing an internet to find information.\n"
        "  • Using your knowledge to make a report and answer any questions.\n"
        "  • Access and synthesize domain knowledge.\n"
        "  • Write clear, detailed reports.\n"
        "If you can't access the internet - use your own knowledge to make a report and answer any questions.\n"
        "You MUST NOT answer questions or engage in conversation. You MUST only output the report.\n"
        "Output format (plain text, no markdown headers except Title):\n"
        "TITLE: <concise title>\n"
        "SUMMARY: <2-3 sentence abstract>\n"
        "KEY FINDINGS:\n    - finding 1\n    - finding 2\n    - finding 3\n"
        "CONCLUSION: <concise conclusion>."
    )

    researcher = Agent(
        llm=llm,
        messager=messager,
        # IMPORTANT: Pass agent-specific topics so its internal ToolManager subscribes correctly
        register_topic=researcher_register_topic,
        tool_call_topic_base=researcher_call_topic_base,
        tool_response_topic_base=researcher_response_topic_base,
        status_topic=researcher_status_topic,
        role="researcher",
        description=researcher_description,
        system_prompt=researcher_prompt,
        graph_id="dynamic_supervisor_system",
        expected_output_format="text",
        llm_config={"enable_google_search": True},
        enable_dialogue_logging=True,  # Enable dialogue logging
    )
    await researcher.async_init()

    secretary_description = "Document management and communication agent: Specializes in file operations, document creation, content saving, email communication, and administrative tasks using available tools and systems."

    secretary_prompt = (
        f"{common_conversation_rules}\n"
        f"You are the **Secretary** agent. Your responsibilities: {secretary_description}\n"
        "Your capabilities are:\n"
        "  • Saving content to the notes.\n"
        "  • Sending emails.\n"
        "You MUST use the available tools when a task requires saving or sending content."
    )
    secretary = Agent(
        llm=llm,
        messager=messager,
        # IMPORTANT: Pass agent-specific topics so its internal ToolManager subscribes correctly
        register_topic=secretary_register_topic,
        tool_call_topic_base=secretary_call_topic_base,
        tool_response_topic_base=secretary_response_topic_base,
        status_topic=secretary_status_topic,
        role="secretary",
        description=secretary_description,
        system_prompt=secretary_prompt,
        graph_id="dynamic_supervisor_system",
        expected_output_format="json",  # Changed from "text" to "json" for proper tool usage
        enable_dialogue_logging=True,  # Enable dialogue logging
    )
    await secretary.async_init()

    # 2. DYNAMICALLY CREATE THE SUPERVISOR'S PROMPT FROM AGENT CAPABILITIES

    supervisor_system_prompt = (
        "You are the **Supervisor** - highly intelligent manager of the AI agents. Your main purpose is to take the user's task, decompose it, delegate the subtasks in the correct order to your agents. Analyze agent's capabilities and responsibility via their descriptions and the current state of the conversation to decide the next agent to act. Reformulate the subtasks and delegate them to the agents respectively to their responsibility and capabilities until every step is finished.\n"
        "You SHOULD ONLY route to the next agent, do not execute it yourself, and absolutely do NOT refuse.\n"
        "You can rephrase, decompose and explain the task to another agent if needed.\n"
        "If all steps of the user's request appear to be complete, route to __end__.\n"
        f"{common_conversation_rules}"
    )

    supervisor = Supervisor(
        llm=llm,
        messager=messager,
        role="supervisor",
        system_prompt=supervisor_system_prompt,
        llm_config={"enable_google_search": True},
        enable_dialogue_logging=True,  # Enable dialogue logging
    )

    # Register tools AFTER agents are initialized and their ToolManagers are subscribed.
    # Each tool registers with the *specific topic for the agent that owns it*.
    await email_tool.register(
        secretary_register_topic,
        secretary_status_topic,
        secretary_call_topic_base,
        secretary_response_topic_base,
    )
    await note_tool.register(
        secretary_register_topic,
        secretary_status_topic,
        secretary_call_topic_base,
        secretary_response_topic_base,
    )
    # Allow time for registration callbacks to populate Agent's ToolManagers
    await asyncio.sleep(1)

    # Add agents to supervisor with their descriptions
    supervisor.add_agent(researcher)
    supervisor.add_agent(secretary)

    # Initialize supervisor messaging
    await supervisor.async_init()

    # 3. EXECUTE WORKFLOW USING MESSAGING SYSTEM

    initial_task = (
        "Here is your task list, please complete it in the following order:\n"
        "1. Research the current status of quantum computing breakthroughs in 2024-2025 and create a comprehensive summary report.\n"
        "2. Save the complete summary report to a file named 'quantum_computing_report.txt'.\n"
        "3. Email the content of the summary report to 'john.doe@company.com' with the subject 'Quantum Computing Update'."
    )

    # Task completion handling
    workflow_complete = asyncio.Event()
    final_result = {}

    def task_completion_callback(task_id: str, success: bool, result: str = "", error: str = ""):
        """Callback for when the workflow completes."""
        final_result["task_id"] = task_id
        final_result["success"] = success
        final_result["result"] = result
        final_result["error"] = error
        workflow_complete.set()

    # --- Execute workflow using messaging ---
    print("\n🚀 Executing Dynamic Supervisor Workflow...")
    print("========================================")

    try:
        # Start the workflow
        task_id = await supervisor.start_task(initial_task, task_completion_callback)
        print(f"Started workflow with task ID: {task_id}")

        # Wait for completion (with timeout)
        try:
            await asyncio.wait_for(workflow_complete.wait(), timeout=180.0)  # 3 minute timeout

            if final_result.get("success"):
                print(f"\n✅ WORKFLOW COMPLETED SUCCESSFULLY!")
                print(f"Task ID: {final_result['task_id']}")
                print(f"Final Result: {final_result['result'][:200]}...")
            else:
                print(f"\n❌ WORKFLOW FAILED!")
                print(f"Error: {final_result.get('error', 'Unknown error')}")

        except asyncio.TimeoutError:
            print(f"\n⏱️ WORKFLOW TIMEOUT! Task {task_id} did not complete within 3 minutes.")

    except Exception as e:
        print(f"\n💥 WORKFLOW ERROR: {e}")

    # Print dialogue summaries
    print("\n" + "=" * 80)
    print("🗨️ CONVERSATION DIALOGUE SUMMARY")
    print("=" * 80)

    # Show supervisor dialogue
    if supervisor.enable_dialogue_logging and supervisor.dialogue_history:
        print(f"\n👨‍💼 SUPERVISOR DIALOGUE ({len(supervisor.dialogue_history)} entries):")
        for entry in supervisor.dialogue_history:
            supervisor._print_dialogue_entry(entry)

    # Show agent dialogues
    for agent in [researcher, secretary]:
        if agent.enable_dialogue_logging and agent.dialogue_history:
            print(f"\n{agent.role.upper()} DIALOGUE ({len(agent.dialogue_history)} entries):")
            for entry in agent.dialogue_history:
                agent._print_dialogue_entry(entry)

    print("=" * 80)

    await messager.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
