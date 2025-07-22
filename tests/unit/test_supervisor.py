import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

from langchain_core.messages import HumanMessage, AIMessage

from argentic.core.agent.agent import Agent
from argentic.core.graph.supervisor import Supervisor
from argentic.core.llm.providers.mock import (
    MockLLMProvider,
    MockScenario,
    MockResponse,
    MockResponseType,
)
from argentic.core.messager.messager import Messager


class TestSupervisor:
    """Unit tests for the Supervisor class."""

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
    def mock_llm_supervisor(self):
        """Create a mock LLM provider for supervisor testing - returns text responses."""
        mock = MockLLMProvider({})
        # Due to MockLLMProvider indexing: call_count increments before getting response
        # First call returns index min(1, len-1), so we need responses at index 1+
        mock.set_responses(
            [
                MockResponse(MockResponseType.DIRECT, content="dummy"),  # Index 0 - never returned
                MockResponse(
                    MockResponseType.DIRECT, content="researcher"
                ),  # Index 1 - first response
                MockResponse(MockResponseType.DIRECT, content="coder"),  # Index 2 - second response
                MockResponse(
                    MockResponseType.DIRECT, content="__end__"
                ),  # Index 3 - third response
            ]
        )
        return mock

    @pytest.fixture
    def supervisor(self, mock_messager, mock_llm_supervisor):
        """Create a basic supervisor for testing."""
        return Supervisor(
            llm=mock_llm_supervisor,
            messager=mock_messager,
            role="test_supervisor",
            system_prompt="Route tasks efficiently.",
        )

    @pytest.fixture
    def researcher_agent(self, mock_messager):
        """Create a mock researcher agent."""
        mock_llm = MockLLMProvider({})
        mock_llm.set_responses(
            [
                MockResponse(MockResponseType.DIRECT, content="dummy"),  # Index 0 - never returned
                MockResponse(
                    MockResponseType.DIRECT,
                    content="I found information about your research topic.",
                ),  # Index 1
            ]
        )

        agent = Agent(
            llm=mock_llm,
            messager=mock_messager,
            role="researcher",
            system_prompt="You are a research assistant.",
            expected_output_format="text",
        )
        return agent

    @pytest.fixture
    def coder_agent(self, mock_messager):
        """Create a mock coder agent."""
        mock_llm = MockLLMProvider({})
        mock_llm.set_responses(
            [
                MockResponse(MockResponseType.DIRECT, content="dummy"),  # Index 0 - never returned
                MockResponse(
                    MockResponseType.DIRECT, content="Here's the code you requested."
                ),  # Index 1
            ]
        )

        agent = Agent(
            llm=mock_llm,
            messager=mock_messager,
            role="coder",
            system_prompt="You are a coding assistant.",
            expected_output_format="text",
        )
        return agent

    @pytest.mark.asyncio
    async def test_supervisor_initialization(self, mock_messager, mock_llm_supervisor):
        """Test supervisor initialization."""
        supervisor = Supervisor(
            llm=mock_llm_supervisor,
            messager=mock_messager,
            role="test_supervisor",
        )

        assert supervisor.role == "test_supervisor"
        assert supervisor._agents == {}

    @pytest.mark.asyncio
    async def test_add_agent(self, supervisor, researcher_agent):
        """Test adding an agent to the supervisor."""
        supervisor.add_agent(researcher_agent)

        assert "researcher" in supervisor._agents
        assert supervisor._agents["researcher"] == researcher_agent.description

    @pytest.mark.asyncio
    async def test_add_multiple_agents(self, supervisor, researcher_agent, coder_agent):
        """Test adding multiple agents to the supervisor."""
        supervisor.add_agent(researcher_agent)
        supervisor.add_agent(coder_agent)

        assert len(supervisor._agents) == 2
        assert "researcher" in supervisor._agents
        assert "coder" in supervisor._agents

    @pytest.mark.asyncio
    async def test_start_task(self, supervisor):
        """Test starting a task with the supervisor."""
        # Create a supervisor with a fresh mock that won't auto-route to completion
        mock_llm = MockLLMProvider({})
        # For a simple task start test, we don't want it to immediately complete
        # So we'll patch the start_task to not actually route

        with patch.object(supervisor, "_route_task") as mock_route:
            task_id = await supervisor.start_task("Test task")

            assert task_id is not None
            assert task_id in supervisor._active_tasks
            assert supervisor._active_tasks[task_id]["original_task"] == "Test task"
            mock_route.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_route_decision_research(self, supervisor, mock_llm_supervisor):
        """Test LLM routing decision for research tasks."""
        # Add a researcher agent so supervisor knows it's valid
        supervisor.add_agent(MagicMock(role="researcher", description="Research specialist"))

        # Set fresh responses for this test
        mock_llm_supervisor.set_responses(
            [
                MockResponse(MockResponseType.DIRECT, content="dummy"),  # Index 0 - never returned
                MockResponse(
                    MockResponseType.DIRECT, content="researcher"
                ),  # Index 1 - first response
            ]
        )
        mock_llm_supervisor.call_count = 0  # Reset call count

        decision = await supervisor._llm_route_decision("Research quantum computing")
        assert decision == "researcher"

    @pytest.mark.asyncio
    async def test_llm_route_decision_coding(self, supervisor, mock_llm_supervisor):
        """Test LLM routing decision for coding tasks."""
        # Add a coder agent so supervisor knows it's valid
        supervisor.add_agent(MagicMock(role="coder", description="Coding specialist"))

        mock_llm_supervisor.set_responses(
            [
                MockResponse(MockResponseType.DIRECT, content="dummy"),  # Index 0 - never returned
                MockResponse(MockResponseType.DIRECT, content="coder"),  # Index 1 - first response
            ]
        )
        mock_llm_supervisor.call_count = 0  # Reset call count

        decision = await supervisor._llm_route_decision("Write Python code for sorting")
        assert decision == "coder"

    @pytest.mark.asyncio
    async def test_llm_route_decision_end_task(self, supervisor, mock_llm_supervisor):
        """Test LLM routing decision for task completion."""
        mock_llm_supervisor.set_responses(
            [
                MockResponse(MockResponseType.DIRECT, content="dummy"),  # Index 0 - never returned
                MockResponse(
                    MockResponseType.DIRECT, content="__end__"
                ),  # Index 1 - first response
            ]
        )
        mock_llm_supervisor.call_count = 0  # Reset call count

        decision = await supervisor._llm_route_decision("Task is complete")
        assert decision == "__end__"

    @pytest.mark.asyncio
    async def test_llm_route_decision_fallback(self, supervisor, mock_llm_supervisor):
        """Test LLM routing decision fallback behavior."""
        mock_llm_supervisor.set_responses(
            [
                MockResponse(MockResponseType.DIRECT, content="dummy"),  # Index 0 - never returned
                MockResponse(
                    MockResponseType.DIRECT, content="invalid_agent"
                ),  # Index 1 - first response
            ]
        )
        mock_llm_supervisor.call_count = 0  # Reset call count

        decision = await supervisor._llm_route_decision("Some task")
        assert decision == "__end__"  # Fallback behavior

    @pytest.mark.asyncio
    async def test_llm_route_decision_partial_match(self, supervisor, mock_llm_supervisor):
        """Test LLM routing decision with partial agent name match."""
        supervisor.add_agent(MagicMock(role="researcher", description="Research specialist"))

        mock_llm_supervisor.set_responses(
            [
                MockResponse(MockResponseType.DIRECT, content="dummy"),  # Index 0 - never returned
                MockResponse(
                    MockResponseType.DIRECT, content="I choose researcher for this task"
                ),  # Index 1
            ]
        )
        mock_llm_supervisor.call_count = 0  # Reset call count

        decision = await supervisor._llm_route_decision("Find information")
        assert decision == "researcher"

    @pytest.mark.asyncio
    async def test_default_supervisor_prompt(self, supervisor):
        """Test the default supervisor prompt generation."""
        prompt = supervisor._get_default_system_prompt()

        assert isinstance(prompt, str)
        assert "supervisor" in prompt.lower()
        assert "route" in prompt.lower()

    @pytest.mark.asyncio
    async def test_supervisor_error_handling(self, supervisor, mock_llm_supervisor):
        """Test supervisor error handling during routing."""
        # Make LLM throw an exception
        mock_llm_supervisor.reset()

        with patch.object(mock_llm_supervisor, "achat", side_effect=Exception("LLM Error")):
            decision = await supervisor._llm_route_decision("Some task")
            assert decision == "__end__"  # Should fallback gracefully

    @pytest.mark.asyncio
    async def test_context_management(self, supervisor):
        """Test supervisor context management features."""
        # Test dialogue logging
        supervisor.enable_dialogue_logging = True
        supervisor._log_dialogue("user", "Test message", "routing")

        assert len(supervisor.dialogue_history) == 1
        assert supervisor.dialogue_history[0]["content_preview"] == "Test message"

    @pytest.mark.asyncio
    async def test_task_history_truncation(self, supervisor):
        """Test task history truncation for context management."""
        # Create a task with long history
        task_info = {
            "original_task": "Test task",
            "history": [{"agent": f"agent_{i}", "result": f"result_{i}"} for i in range(20)],
        }

        truncated = supervisor._truncate_task_history(task_info)

        # Should be truncated to max_task_history_items or just slightly over due to truncation marker
        # The implementation keeps first 2 + recent (max-2) + 1 truncation marker = max+1 total
        assert len(truncated["history"]) <= supervisor.max_task_history_items + 1


class TestSupervisorIntegration:
    """Integration tests for Supervisor with agents."""

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
    async def test_full_multi_agent_workflow(self, mock_messager):
        """Test a complete multi-agent workflow."""
        # Create supervisor with specific routing scenario
        supervisor_llm = MockLLMProvider({})
        supervisor_llm.set_responses(
            [
                MockResponse(MockResponseType.DIRECT, content="dummy"),  # Index 0 - never returned
                MockResponse(
                    MockResponseType.DIRECT, content="researcher"
                ),  # Index 1 - first response
            ]
        )

        supervisor = Supervisor(
            llm=supervisor_llm,
            messager=mock_messager,
            role="workflow_supervisor",
            enable_dialogue_logging=True,
        )

        # Create researcher agent
        researcher_llm = MockLLMProvider({})
        researcher_llm.set_responses(
            [
                MockResponse(MockResponseType.DIRECT, content="dummy"),  # Index 0 - never returned
                MockResponse(
                    MockResponseType.DIRECT,
                    content="I found detailed information about quantum computing applications.",
                ),
            ]
        )

        researcher = Agent(
            llm=researcher_llm,
            messager=mock_messager,
            role="researcher",
            expected_output_format="text",
        )

        # Add agents to supervisor
        supervisor.add_agent(researcher)

        # Test task creation - patch _route_task to prevent immediate completion
        with patch.object(supervisor, "_route_task") as mock_route:
            task_id = await supervisor.start_task(
                "Research quantum computing and write code examples"
            )

            assert task_id is not None
            assert task_id in supervisor._active_tasks
            mock_route.assert_called_once()

    @pytest.mark.asyncio
    async def test_supervisor_agent_coordination(self, mock_messager):
        """Test coordination between supervisor and agents."""
        # Create a more complex routing scenario
        routing_scenario = MockScenario("multi_step_routing")
        routing_scenario.add_direct_response("researcher")  # First route to researcher
        routing_scenario.add_direct_response("coder")  # Then route to coder
        routing_scenario.add_direct_response("__end__")  # Finally end

        supervisor_llm = MockLLMProvider({}).set_scenario(routing_scenario)

        supervisor = Supervisor(
            llm=supervisor_llm,
            messager=mock_messager,
            role="coordination_supervisor",
            enable_dialogue_logging=True,
        )

        # Add mock agents using description strings
        researcher_agent = MagicMock()
        researcher_agent.role = "researcher"
        researcher_agent.description = "Research and information gathering specialist"

        coder_agent = MagicMock()
        coder_agent.role = "coder"
        coder_agent.description = "Code development and programming specialist"

        supervisor.add_agent(researcher_agent)
        supervisor.add_agent(coder_agent)

        # Test multiple routing decisions
        task_id = await supervisor.start_task("Complex multi-step task")

        assert len(supervisor._agents) == 2
        assert "researcher" in supervisor._agents
        assert "coder" in supervisor._agents
