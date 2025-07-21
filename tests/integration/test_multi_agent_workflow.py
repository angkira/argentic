import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

from langchain_core.messages import HumanMessage, AIMessage

from argentic.core.agent.agent import Agent
from argentic.core.graph.supervisor import Supervisor
from argentic.core.llm.providers.mock import MockLLMProvider, MockScenario
from argentic.core.messager.messager import Messager
from argentic.core.tools.tool_manager import ToolManager
from argentic.core.graph.state import AgentState
from tests.unit.test_mock_tools import (
    MockSearchTool,
    MockCalculatorTool,
    MockCodeExecutorTool,
    MockToolManager,
)


class TestMultiAgentWorkflowIntegration:
    """Integration tests for complete multi-agent workflows."""

    @pytest.fixture
    def mock_messager(self):
        """Create a mock messager for testing."""
        messager = MagicMock(spec=Messager)
        messager.connect = AsyncMock()
        messager.publish = AsyncMock()
        messager.subscribe = AsyncMock()
        messager.disconnect = AsyncMock()
        return messager

    @pytest.fixture
    def mock_tool_manager_with_tools(self):
        """Create a mock tool manager with registered tools."""
        manager = MockToolManager()

        # Register various tools
        search_tool = MockSearchTool()
        calc_tool = MockCalculatorTool()
        code_tool = MockCodeExecutorTool()

        manager.register_tools([search_tool, calc_tool, code_tool])
        return manager

    @pytest.mark.asyncio
    async def test_research_workflow(self, mock_messager, mock_tool_manager_with_tools):
        """Test a complete research workflow."""
        # Create supervisor that routes to researcher
        supervisor_scenario = MockScenario("research_workflow")
        supervisor_scenario.add_direct_response("researcher")
        supervisor_scenario.add_direct_response("__end__")

        supervisor_llm = MockLLMProvider({}).set_scenario(supervisor_scenario)

        supervisor = Supervisor(
            llm=supervisor_llm,
            messager=mock_messager,
            tool_manager=mock_tool_manager_with_tools,
            role="research_supervisor",
        )

        # Create researcher that uses search tool
        researcher_scenario = MockScenario("researcher_work")
        researcher_scenario.add_tool_call(
            "search_tool", {"query": "quantum computing"}, "I'll search for information."
        )
        researcher_scenario.add_direct_response(
            "Based on my research, quantum computing uses quantum mechanical phenomena..."
        )

        researcher_llm = MockLLMProvider({}).set_scenario(researcher_scenario)

        researcher = Agent(
            llm=researcher_llm,
            messager=mock_messager,
            role="researcher",
            system_prompt="You are a research assistant.",
            expected_output_format="text",
        )

        # Patch the researcher's tool manager to use our mock
        with patch.object(researcher, "_tool_manager", mock_tool_manager_with_tools):
            # Add agents and compile
            supervisor.add_agent(researcher)
            graph = supervisor.compile()

            # Test the workflow
            initial_state: AgentState = {
                "messages": [HumanMessage(content="Research quantum computing")],
                "next": None,
            }

            # Run the workflow step by step
            events = []
            async for event in graph.astream(initial_state):
                events.append(event)

            # Verify workflow completion
            assert len(events) > 0

            # Check that search tool was called
            search_tool = mock_tool_manager_with_tools.get_tool("search_tool")
            assert search_tool.call_count > 0
            assert any("quantum computing" in str(call) for call in search_tool.call_history)

    @pytest.mark.asyncio
    async def test_coding_workflow(self, mock_messager, mock_tool_manager_with_tools):
        """Test a complete coding workflow."""
        # Create supervisor that routes to coder
        supervisor_scenario = MockScenario("coding_workflow")
        supervisor_scenario.add_direct_response("coder")
        supervisor_scenario.add_direct_response("__end__")

        supervisor_llm = MockLLMProvider({}).set_scenario(supervisor_scenario)

        supervisor = Supervisor(
            llm=supervisor_llm,
            messager=mock_messager,
            tool_manager=mock_tool_manager_with_tools,
            role="coding_supervisor",
        )

        # Create coder that uses code execution tool
        coder_scenario = MockScenario("coder_work")
        coder_scenario.add_tool_call(
            "code_executor",
            {"language": "python", "code": "print('hello world')"},
            "I'll execute this code.",
        )
        coder_scenario.add_direct_response(
            "The code executed successfully and printed 'hello world'."
        )

        coder_llm = MockLLMProvider({}).set_scenario(coder_scenario)

        coder = Agent(
            llm=coder_llm,
            messager=mock_messager,
            role="coder",
            system_prompt="You are a coding assistant.",
            expected_output_format="text",
        )

        # Patch the coder's tool manager to use our mock
        with patch.object(coder, "_tool_manager", mock_tool_manager_with_tools):
            # Add agents and compile
            supervisor.add_agent(coder)
            graph = supervisor.compile()

            # Test the workflow
            initial_state: AgentState = {
                "messages": [HumanMessage(content="Write and run a Python hello world program")],
                "next": None,
            }

            # Run the workflow
            events = []
            async for event in graph.astream(initial_state):
                events.append(event)

            # Verify workflow completion
            assert len(events) > 0

            # Check that code executor was called
            code_tool = mock_tool_manager_with_tools.get_tool("code_executor")
            assert code_tool.call_count > 0
            assert any("python" in str(call) for call in code_tool.call_history)

    @pytest.mark.asyncio
    async def test_complex_multi_step_workflow(self, mock_messager, mock_tool_manager_with_tools):
        """Test a complex workflow involving multiple agents and tool calls."""
        # Create supervisor that routes between multiple agents
        supervisor_scenario = MockScenario("complex_workflow")
        supervisor_scenario.add_direct_response("researcher")  # First route to researcher
        supervisor_scenario.add_direct_response("coder")  # Then route to coder
        supervisor_scenario.add_direct_response("__end__")  # Finally end

        supervisor_llm = MockLLMProvider({}).set_scenario(supervisor_scenario)

        supervisor = Supervisor(
            llm=supervisor_llm,
            messager=mock_messager,
            tool_manager=mock_tool_manager_with_tools,
            role="complex_supervisor",
        )

        # Create researcher
        researcher_scenario = MockScenario("complex_research")
        researcher_scenario.add_tool_call(
            "search_tool", {"query": "sorting algorithms"}, "Researching sorting algorithms..."
        )
        researcher_scenario.add_direct_response(
            "I found information about various sorting algorithms including quicksort, mergesort, and bubblesort."
        )

        researcher_llm = MockLLMProvider({}).set_scenario(researcher_scenario)

        researcher = Agent(
            llm=researcher_llm,
            messager=mock_messager,
            role="researcher",
            expected_output_format="text",
        )

        # Create coder
        coder_scenario = MockScenario("complex_coding")
        coder_scenario.add_tool_call(
            "code_executor",
            {"language": "python", "code": "def quicksort(arr): pass"},
            "Implementing quicksort...",
        )
        coder_scenario.add_direct_response("I've implemented a quicksort algorithm in Python.")

        coder_llm = MockLLMProvider({}).set_scenario(coder_scenario)

        coder = Agent(
            llm=coder_llm, messager=mock_messager, role="coder", expected_output_format="text"
        )

        # Patch both agents' tool managers
        with (
            patch.object(researcher, "_tool_manager", mock_tool_manager_with_tools),
            patch.object(coder, "_tool_manager", mock_tool_manager_with_tools),
        ):

            # Add agents and compile
            supervisor.add_agent(researcher)
            supervisor.add_agent(coder)
            graph = supervisor.compile()

            # Test the complex workflow
            initial_state: AgentState = {
                "messages": [
                    HumanMessage(
                        content="Research sorting algorithms and then implement quicksort in Python"
                    )
                ],
                "next": None,
            }

            # Run the workflow
            events = []
            async for event in graph.astream(initial_state):
                events.append(event)

            # Verify both tools were used
            search_tool = mock_tool_manager_with_tools.get_tool("search_tool")
            code_tool = mock_tool_manager_with_tools.get_tool("code_executor")

            assert search_tool.call_count > 0
            assert code_tool.call_count > 0
            assert any("sorting" in str(call) for call in search_tool.call_history)
            assert any("quicksort" in str(call) for call in code_tool.call_history)

    @pytest.mark.asyncio
    async def test_error_handling_workflow(self, mock_messager, mock_tool_manager_with_tools):
        """Test workflow with tool failures and error recovery."""
        # Setup tools to fail
        search_tool = mock_tool_manager_with_tools.get_tool("search_tool")
        search_tool.set_failure_mode(True, "Search service is down")

        # Create supervisor
        supervisor_scenario = MockScenario("error_workflow")
        supervisor_scenario.add_direct_response("researcher")
        supervisor_scenario.add_direct_response("__end__")

        supervisor_llm = MockLLMProvider({}).set_scenario(supervisor_scenario)

        supervisor = Supervisor(
            llm=supervisor_llm,
            messager=mock_messager,
            tool_manager=mock_tool_manager_with_tools,
            role="error_supervisor",
        )

        # Create researcher that tries to use failing search tool
        researcher_scenario = MockScenario("error_research")
        researcher_scenario.add_tool_call("search_tool", {"query": "test"}, "Trying to search...")
        researcher_scenario.add_direct_response(
            "The search tool is currently unavailable, but I can provide general information about the topic."
        )

        researcher_llm = MockLLMProvider({}).set_scenario(researcher_scenario)

        researcher = Agent(
            llm=researcher_llm,
            messager=mock_messager,
            role="researcher",
            expected_output_format="text",
        )

        # Patch the researcher's tool manager
        with patch.object(researcher, "_tool_manager", mock_tool_manager_with_tools):
            # Add agents and compile
            supervisor.add_agent(researcher)
            graph = supervisor.compile()

            # Test the workflow with errors
            initial_state: AgentState = {
                "messages": [HumanMessage(content="Search for information about AI")],
                "next": None,
            }

            # Run the workflow - should handle errors gracefully
            events = []
            async for event in graph.astream(initial_state):
                events.append(event)

            # Verify that despite tool failure, workflow completed
            assert len(events) > 0

            # Check that search tool was called (even though it failed)
            assert search_tool.call_count > 0

    @pytest.mark.asyncio
    async def test_agent_messaging_integration(self, mock_messager):
        """Test agent messaging integration with the messager."""
        # Create simple agent
        mock_llm = MockLLMProvider({})
        mock_llm.add_direct_response("Hello! I'm ready to help you.")

        agent = Agent(
            llm=mock_llm,
            messager=mock_messager,
            role="messaging_agent",
            answer_topic="test/answers",
        )

        # Test async initialization
        await agent.async_init()

        # Test handling ask question
        from argentic.core.protocol.message import AskQuestionMessage
        from argentic.core.protocol.enums import MessageSource

        question_msg = AskQuestionMessage(
            question="Hello, how are you?", user_id="test_user", source=MessageSource.USER
        )

        await agent.handle_ask_question(question_msg)

        # Verify message was published
        mock_messager.publish.assert_called()

        # Check the published message
        call_args = mock_messager.publish.call_args
        topic, message = call_args[0]

        assert topic == "test/answers/test_user"  # User-specific topic
        assert message.question == "Hello, how are you?"
        assert message.answer == "Hello! I'm ready to help you."
        assert message.user_id == "test_user"

    @pytest.mark.asyncio
    async def test_performance_with_multiple_tool_calls(
        self, mock_messager, mock_tool_manager_with_tools
    ):
        """Test performance with multiple sequential tool calls."""
        # Create agent that makes multiple tool calls
        multi_tool_scenario = MockScenario("performance_test")

        # Add multiple tool calls
        for i in range(5):
            multi_tool_scenario.add_tool_call(
                "search_tool", {"query": f"query_{i}"}, f"Searching for query {i}..."
            )

        multi_tool_scenario.add_direct_response("Completed all searches successfully.")

        agent_llm = MockLLMProvider({}).set_scenario(multi_tool_scenario)

        agent = Agent(
            llm=agent_llm,
            messager=mock_messager,
            role="performance_agent",
            expected_output_format="text",
        )

        # Patch the agent's tool manager
        with patch.object(agent, "_tool_manager", mock_tool_manager_with_tools):
            # Measure execution time
            import time

            start_time = time.time()

            result = await agent.query("Perform multiple searches")

            end_time = time.time()
            execution_time = end_time - start_time

            # Verify execution completed
            assert "Completed all searches" in result

            # Verify all tool calls were made
            search_tool = mock_tool_manager_with_tools.get_tool("search_tool")
            assert search_tool.call_count == 5

            # Performance should be reasonable (less than 1 second for mock tools)
            assert execution_time < 1.0

    @pytest.mark.asyncio
    async def test_concurrent_agent_execution(self, mock_messager):
        """Test concurrent execution of multiple agents."""
        # Create multiple agents
        agents = []
        for i in range(3):
            mock_llm = MockLLMProvider({})
            mock_llm.add_direct_response(f"Response from agent {i}")

            agent = Agent(
                llm=mock_llm,
                messager=mock_messager,
                role=f"agent_{i}",
                expected_output_format="text",
            )
            agents.append(agent)

        # Execute all agents concurrently
        tasks = [agent.query(f"Question for agent {i}") for i, agent in enumerate(agents)]

        results = await asyncio.gather(*tasks)

        # Verify all agents responded
        assert len(results) == 3
        for i, result in enumerate(results):
            assert f"Response from agent {i}" in result


class TestEndToEndScenarios:
    """End-to-end test scenarios mimicking real usage."""

    @pytest.fixture
    def mock_messager(self):
        """Create a mock messager for testing."""
        messager = MagicMock(spec=Messager)
        messager.connect = AsyncMock()
        messager.publish = AsyncMock()
        messager.subscribe = AsyncMock()
        messager.disconnect = AsyncMock()
        return messager

    @pytest.mark.asyncio
    async def test_quantum_computing_research_scenario(self, mock_messager):
        """End-to-end test: Research quantum computing and write example code."""
        # This mimics the example from the multi_agent_example.py

        # Create tool manager with search and code tools
        tool_manager = MockToolManager()
        search_tool = MockSearchTool()
        code_tool = MockCodeExecutorTool()

        # Customize search results for quantum computing
        search_tool.set_search_results(
            [
                "Quantum computing leverages quantum mechanical phenomena like superposition and entanglement",
                "Key quantum algorithms include Shor's algorithm and Grover's algorithm",
                "Current quantum computers include IBM Quantum, Google Sycamore, and Rigetti systems",
            ]
        )

        tool_manager.register_tools([search_tool, code_tool])

        # Create supervisor
        supervisor_scenario = MockScenario("quantum_scenario")
        supervisor_scenario.add_direct_response("researcher")  # Route to researcher first
        supervisor_scenario.add_direct_response("__end__")  # Then end

        supervisor_llm = MockLLMProvider({}).set_scenario(supervisor_scenario)

        supervisor = Supervisor(
            llm=supervisor_llm,
            messager=mock_messager,
            tool_manager=tool_manager,
            role="quantum_supervisor",
            system_prompt="Route tasks: 'researcher' for info/data queries, 'coder' for programming. Be direct.",
        )

        # Create researcher
        researcher_scenario = MockScenario("quantum_research")
        researcher_scenario.add_tool_call(
            "search_tool",
            {"query": "quantum computing current status"},
            "Researching quantum computing...",
        )
        researcher_scenario.add_direct_response(
            "Based on current research, quantum computing is advancing rapidly with companies like IBM, Google, and others developing quantum processors."
        )

        researcher_llm = MockLLMProvider({}).set_scenario(researcher_scenario)

        researcher = Agent(
            llm=researcher_llm,
            messager=mock_messager,
            role="researcher",
            system_prompt="Research and provide factual information. No fluff.",
            expected_output_format="text",
        )

        # Patch tool managers
        with patch.object(researcher, "_tool_manager", tool_manager):
            # Add agents and compile
            supervisor.add_agent(researcher)
            graph = supervisor.compile()

            # Run the scenario
            initial_state: AgentState = {
                "messages": [
                    HumanMessage(content="Research the current status of quantum computing.")
                ],
                "next": None,
            }

            events = []
            final_messages = []

            async for event in graph.astream(initial_state):
                events.append(event)
                for key, value in event.items():
                    if key in ["researcher", "supervisor"] and "messages" in value:
                        final_messages.extend(value["messages"])

            # Verify the research was conducted
            assert len(events) > 0
            assert search_tool.call_count > 0

            # Verify research content
            search_calls = search_tool.call_history
            assert any("quantum" in str(call).lower() for call in search_calls)

            # Verify final response contains quantum computing information
            response_content = " ".join(
                [
                    getattr(msg, "content", "")
                    for msg in final_messages
                    if hasattr(msg, "content") and msg.content
                ]
            )
            assert "quantum" in response_content.lower()

    @pytest.mark.asyncio
    async def test_code_development_scenario(self, mock_messager):
        """End-to-end test: Develop and test a sorting algorithm."""

        # Create tool manager
        tool_manager = MockToolManager()
        code_tool = MockCodeExecutorTool()
        tool_manager.register_tools([code_tool])

        # Create supervisor that routes to coder
        supervisor_scenario = MockScenario("coding_scenario")
        supervisor_scenario.add_direct_response("coder")
        supervisor_scenario.add_direct_response("__end__")

        supervisor_llm = MockLLMProvider({}).set_scenario(supervisor_scenario)

        supervisor = Supervisor(
            llm=supervisor_llm,
            messager=mock_messager,
            tool_manager=tool_manager,
            role="coding_supervisor",
        )

        # Create coder
        coder_scenario = MockScenario("algorithm_coding")
        coder_scenario.add_tool_call(
            "code_executor",
            {
                "language": "python",
                "code": "def bubble_sort(arr):\n    for i in range(len(arr)):\n        for j in range(0, len(arr)-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr\n\nprint(bubble_sort([64, 34, 25, 12, 22, 11, 90]))",
            },
            "Implementing and testing bubble sort...",
        )
        coder_scenario.add_direct_response(
            "I've implemented a bubble sort algorithm and tested it with sample data."
        )

        coder_llm = MockLLMProvider({}).set_scenario(coder_scenario)

        coder = Agent(
            llm=coder_llm,
            messager=mock_messager,
            role="coder",
            system_prompt="Write code. Include brief comments only when necessary.",
            expected_output_format="text",
        )

        # Patch tool manager
        with patch.object(coder, "_tool_manager", tool_manager):
            # Add agents and compile
            supervisor.add_agent(coder)
            graph = supervisor.compile()

            # Run the scenario
            initial_state: AgentState = {
                "messages": [
                    HumanMessage(content="Implement a bubble sort algorithm in Python and test it.")
                ],
                "next": None,
            }

            events = []
            async for event in graph.astream(initial_state):
                events.append(event)

            # Verify code was executed
            assert len(events) > 0
            assert code_tool.call_count > 0

            # Verify bubble sort implementation
            code_calls = code_tool.call_history
            assert any("bubble_sort" in str(call) for call in code_calls)
            assert any("python" in str(call) for call in code_calls)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
