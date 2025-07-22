import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

from langchain_core.messages import HumanMessage, AIMessage

from argentic.core.agent.agent import Agent
from argentic.core.llm.providers.mock import (
    MockLLMProvider,
    MockResponse,
    MockResponseType,
    MockScenario,
    MockToolCall,
)
from argentic.core.messager.messager import Messager
from argentic.core.protocol.message import (
    AgentLLMResponseMessage,
    AskQuestionMessage,
    AnswerMessage,
    BaseMessage,
)
from argentic.core.protocol.enums import MessageSource
from argentic.core.protocol.task import TaskResultMessage, TaskStatus
from argentic.core.graph.state import AgentState


class TestAgent:
    """Unit tests for the Agent class."""

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
    def mock_llm_simple(self):
        """Create a simple mock LLM provider with JSON responses."""
        mock = MockLLMProvider({})
        # Set responses directly instead of adding to defaults
        mock.set_responses(
            [
                MockResponse(
                    MockResponseType.DIRECT,
                    content='{"type": "direct", "content": "This is a direct answer."}',
                ),
                MockResponse(
                    MockResponseType.DIRECT,
                    content='{"type": "direct", "content": "Mock response 2"}',
                ),
                MockResponse(
                    MockResponseType.DIRECT,
                    content='{"type": "direct", "content": "Mock response 3"}',
                ),
            ]
        )
        return mock

    @pytest.fixture
    def mock_llm_tool_scenario(self):
        """Create a mock LLM provider with tool calling scenario."""
        mock = MockLLMProvider({})
        # Due to mock indexing bug where call_count increments before getting response:
        # - responses[1] is returned on first call (should be tool call)
        # - responses[1] is returned on second call too (because min(2, 1) = 1)
        # So we need to put the tool_result at index 1 and tool_call later to break the cycle
        mock.set_responses(
            [
                MockResponse(
                    MockResponseType.DIRECT,
                    content='{"type": "tool_result", "tool_id": "test_tool", "result": "Tool execution completed successfully."}',
                ),
                MockResponse(
                    MockResponseType.TOOL_CALL,
                    tool_calls=[MockToolCall(name="test_tool", args={"param1": "value1"})],
                ),
                MockResponse(
                    MockResponseType.DIRECT,
                    content='{"type": "tool_result", "tool_id": "test_tool", "result": "Tool execution completed successfully."}',
                ),
            ]
        )
        # Set to cycle mode so it can return different responses
        mock.response_mode = "cycle"
        return mock

    @pytest.fixture
    def agent(self, mock_messager, mock_llm_simple):
        """Create a basic agent for testing."""
        return Agent(
            llm=mock_llm_simple,
            messager=mock_messager,
            role="test_agent",
            system_prompt="You are a helpful test assistant.",
            expected_output_format="json",
        )

    @pytest.fixture
    def text_agent(self, mock_messager):
        """Create an agent with text output format."""
        mock_llm = MockLLMProvider({})
        # Fix for mock indexing bug - first response returned is index 1
        mock_llm.set_responses(
            [
                MockResponse(MockResponseType.DIRECT, content="Dummy response"),
                MockResponse(MockResponseType.DIRECT, content="Simple text response"),
            ]
        )
        return Agent(
            llm=mock_llm,
            messager=mock_messager,
            role="text_agent",
            system_prompt="You are a helpful test assistant.",
            expected_output_format="text",
        )

    @pytest.mark.asyncio
    async def test_agent_initialization(self, mock_messager, mock_llm_simple):
        """Test agent initialization with correct parameters."""
        agent = Agent(llm=mock_llm_simple, messager=mock_messager, role="test_agent")

        assert agent.role == "test_agent"
        assert agent.llm == mock_llm_simple
        assert agent.messager == mock_messager
        assert agent.expected_output_format == "json"  # default
        assert agent.max_tool_iterations == 10
        assert agent.history == []

    @pytest.mark.asyncio
    async def test_agent_async_init(self, agent):
        """Test agent async initialization."""
        with patch.object(agent._tool_manager, "async_init", new_callable=AsyncMock) as mock_init:
            await agent.async_init()
            mock_init.assert_called_once()

    @pytest.mark.asyncio
    async def test_agent_stop(self, agent):
        """Test agent stop functionality."""
        await agent.stop()
        # Should complete without errors

    @pytest.mark.asyncio
    async def test_query_direct_response(self, agent, mock_llm_simple):
        """Test agent query with direct response."""
        mock_llm_simple.set_responses(
            [
                MockResponse(
                    MockResponseType.DIRECT,
                    content='{"type": "direct", "content": "This is a direct answer."}',
                )
            ]
        )
        mock_llm_simple.call_count = 0  # Reset call count for clean test

        result = await agent.query("What is the capital of France?")

        assert result == "This is a direct answer."
        mock_llm_simple.assert_called(times=1)
        mock_llm_simple.assert_prompt_contains("capital of France")

    @pytest.mark.asyncio
    async def test_query_with_tools(self, mock_messager, mock_llm_tool_scenario):
        """Test agent query that involves tool calling."""
        # Mock the tool manager to return a successful result
        agent = Agent(llm=mock_llm_tool_scenario, messager=mock_messager, role="tool_agent")

        # Mock tool execution
        mock_tool_result = TaskResultMessage(
            tool_id="test_tool",
            tool_name="test_tool",
            task_id="task_123",
            status=TaskStatus.SUCCESS,
            result="Tool executed successfully",
        )

        with patch.object(
            agent._tool_manager, "get_tool_results", new_callable=AsyncMock
        ) as mock_get_results:
            mock_get_results.return_value = ([mock_tool_result], False)

            result = await agent.query("Use the test tool to help me.")

            # Should get the final response after tool execution
            assert "Tool execution completed successfully." in result
            mock_get_results.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_max_iterations(self, agent, mock_llm_simple):
        """Test agent query respects max iterations."""
        # Set up responses that would cause infinite loop without max iterations
        infinite_responses = []
        for _ in range(15):  # More than max_tool_iterations (10)
            infinite_responses.append(
                MockResponse(
                    MockResponseType.DIRECT,
                    content='{"type": "tool_call", "tool_calls": [{"tool_id": "infinite_tool", "arguments": {}}]}',
                )
            )

        mock_llm_simple.set_responses(infinite_responses)
        mock_llm_simple.call_count = 0  # Reset call count for clean test

        with patch.object(
            agent._tool_manager, "get_tool_results", new_callable=AsyncMock
        ) as mock_get_results:
            mock_get_results.return_value = ([], False)  # No results, no errors

            result = await agent.query("Start an infinite loop.")

            assert "Max iterations reached" in result
            # Should not exceed max iterations
            assert mock_llm_simple.call_count <= agent.max_tool_iterations + 1

    @pytest.mark.asyncio
    async def test_invoke_with_direct_response(self, agent, mock_llm_simple):
        """Test agent invoke method with direct response."""
        mock_llm_simple.set_responses(
            [
                MockResponse(
                    MockResponseType.DIRECT,
                    content='{"type": "direct", "content": "Direct response content"}',
                )
            ]
        )
        mock_llm_simple.call_count = 0  # Reset call count for clean test

        state: AgentState = {"messages": [HumanMessage(content="Test question")], "next": None}

        result = await agent.invoke(state)

        assert "messages" in result
        assert len(result["messages"]) == 2  # Input message + response message
        # The first message should be the original input
        assert isinstance(result["messages"][0], HumanMessage)
        # The response should be an AIMessage with the direct content
        response_msg = result["messages"][1]
        assert isinstance(response_msg, AIMessage)

    @pytest.mark.asyncio
    async def test_invoke_with_tool_calls(self, mock_messager, mock_llm_tool_scenario):
        """Test agent invoke method with tool calls."""
        agent = Agent(llm=mock_llm_tool_scenario, messager=mock_messager, role="tool_agent")

        # Mock tool execution
        mock_tool_result = TaskResultMessage(
            tool_id="test_tool",
            tool_name="test_tool",
            task_id="task_123",
            status=TaskStatus.SUCCESS,
            result="Tool result",
        )

        state: AgentState = {"messages": [HumanMessage(content="Use a tool")], "next": None}

        with patch.object(
            agent._tool_manager, "get_tool_results", new_callable=AsyncMock
        ) as mock_get_results:
            mock_get_results.return_value = ([mock_tool_result], False)

            result = await agent.invoke(state)

            assert "messages" in result
            # Should have the LLM response + tool result messages
            assert len(result["messages"]) >= 2

    @pytest.mark.asyncio
    async def test_handle_ask_question(self, agent, mock_llm_simple):
        """Test agent handling of ask question messages."""
        mock_llm_simple.set_responses(
            [
                MockResponse(
                    MockResponseType.DIRECT,
                    content='{"type": "direct", "content": "Question answered."}',
                )
            ]
        )
        mock_llm_simple.call_count = 0  # Reset call count for clean test

        ask_msg = AskQuestionMessage(
            question="What is AI?", user_id="test_user", source=MessageSource.USER
        )

        await agent.handle_ask_question(ask_msg)

        # Should publish answer
        agent.messager.publish.assert_called()

        # Check that the published message is an AnswerMessage
        call_args = agent.messager.publish.call_args
        topic, message = call_args[0]
        assert isinstance(message, AnswerMessage)
        assert message.question == "What is AI?"
        assert message.answer == "Question answered."
        assert message.user_id == "test_user"

    @pytest.mark.asyncio
    async def test_text_output_format(self, text_agent):
        """Test agent with text output format."""
        result = await text_agent.query("Simple question")

        assert result == "Simple text response"
        text_agent.llm.assert_called(times=1)

    @pytest.mark.asyncio
    async def test_system_prompt_usage(self, mock_messager, mock_llm_simple):
        """Test that system prompt is properly used."""
        custom_prompt = "You are a specialized test assistant."
        agent = Agent(
            llm=mock_llm_simple,
            messager=mock_messager,
            role="specialized_agent",
            system_prompt=custom_prompt,
        )

        mock_llm_simple.set_responses(
            [
                MockResponse(
                    MockResponseType.DIRECT,
                    content='{"type": "direct", "content": "Specialized response"}',
                )
            ]
        )
        mock_llm_simple.call_count = 0  # Reset call count for clean test

        await agent.query("Test question")

        # Check that the system prompt was included
        captured_prompt = mock_llm_simple.get_captured_prompt()
        assert custom_prompt in captured_prompt

    @pytest.mark.asyncio
    async def test_error_handling_in_query(self, agent, mock_llm_simple):
        """Test error handling during query processing."""
        mock_llm_simple.set_responses(
            [MockResponse(MockResponseType.ERROR, error_message="Simulated LLM error")]
        )
        mock_llm_simple.call_count = 0  # Reset call count for clean test

        with pytest.raises(Exception):
            await agent.query("This will cause an error")

    @pytest.mark.asyncio
    async def test_tool_execution_error_handling(self, mock_messager, mock_llm_tool_scenario):
        """Test handling of tool execution errors."""
        agent = Agent(llm=mock_llm_tool_scenario, messager=mock_messager, role="error_agent")

        # Mock tool execution error
        from argentic.core.protocol.task import TaskErrorMessage

        mock_error_result = TaskErrorMessage(
            tool_id="test_tool",
            tool_name="test_tool",
            task_id="task_123",
            error="Tool execution failed",
        )

        with patch.object(
            agent._tool_manager, "get_tool_results", new_callable=AsyncMock
        ) as mock_get_results:
            mock_get_results.return_value = ([mock_error_result], True)

            result = await agent.query("Use a failing tool")

            # Should handle the error gracefully
            assert isinstance(result, str)
            mock_get_results.assert_called_once()

    @pytest.mark.asyncio
    async def test_message_conversion_langchain_to_protocol(self, agent):
        """Test conversion from LangChain messages to protocol messages."""
        langchain_messages = [
            HumanMessage(content="Human message"),
            AIMessage(content="AI response"),
        ]

        protocol_messages = agent._convert_langchain_to_protocol_messages(langchain_messages)

        assert len(protocol_messages) == 2
        assert isinstance(protocol_messages[0], AskQuestionMessage)
        assert protocol_messages[0].question == "Human message"

    @pytest.mark.asyncio
    async def test_message_conversion_protocol_to_langchain(self, agent):
        """Test conversion from protocol messages to LangChain messages."""
        protocol_messages = [
            AskQuestionMessage(question="Test question", source=MessageSource.USER),
            AgentLLMResponseMessage(raw_content="Test response", source=MessageSource.LLM),
        ]

        langchain_messages = agent._convert_protocol_to_langchain_messages(protocol_messages)

        assert len(langchain_messages) == 2
        assert isinstance(langchain_messages[0], HumanMessage)
        assert langchain_messages[0].content == "Test question"
        assert isinstance(langchain_messages[1], AIMessage)

    @pytest.mark.asyncio
    async def test_set_log_level(self, agent):
        """Test setting log level."""
        from argentic.core.logger import LogLevel

        agent.set_log_level(LogLevel.DEBUG)
        assert agent.log_level == LogLevel.DEBUG

        agent.set_log_level("WARNING")
        assert agent.log_level == LogLevel.WARNING

    @pytest.mark.asyncio
    async def test_set_system_prompt(self, agent):
        """Test updating system prompt."""
        new_prompt = "New system prompt"
        agent.set_system_prompt(new_prompt)

        assert agent.system_prompt == new_prompt
        # Should rebuild prompt template
        assert agent.prompt_template is not None

    @pytest.mark.asyncio
    async def test_get_system_prompt(self, agent):
        """Test getting current system prompt."""
        prompt = agent.get_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0


class TestAgentScenarios:
    """Test Agent with different scenarios."""

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
    async def test_research_agent_scenario(self, mock_messager):
        """Test agent in a research scenario."""
        # Create a research-focused scenario
        scenario = MockScenario("research_task")
        scenario.add_direct_response(
            '{"type": "direct", "content": "I\'ll help you research that topic."}'
        )
        scenario.add_tool_call(
            "search_tool", {"query": "quantum computing"}
        )  # Use native tool call
        scenario.add_direct_response(
            '{"type": "tool_result", "tool_id": "search_tool", "result": "Based on my research, quantum computing is..."}'
        )

        mock_llm = MockLLMProvider({}).set_scenario(scenario)

        agent = Agent(
            llm=mock_llm,
            messager=mock_messager,
            role="researcher",
            system_prompt="You are a research assistant specialized in finding accurate information.",
        )

        # Mock successful tool execution
        with patch.object(
            agent._tool_manager, "get_tool_results", new_callable=AsyncMock
        ) as mock_get_results:
            mock_tool_result = TaskResultMessage(
                tool_id="search_tool",
                tool_name="search_tool",
                task_id="search_123",
                status=TaskStatus.SUCCESS,
                result="Quantum computing research results...",
            )
            mock_get_results.return_value = ([mock_tool_result], False)

            result = await agent.query("Research the current state of quantum computing")

            assert "quantum computing" in result.lower()
            mock_llm.assert_prompt_contains("research")

    @pytest.mark.asyncio
    async def test_coding_agent_scenario(self, mock_messager):
        """Test agent in a coding scenario."""
        # Create a coding-focused scenario
        scenario = MockScenario("coding_task")
        scenario.add_direct_response(
            '{"type": "direct", "content": "I\'ll help you write that code."}'
        )
        scenario.add_tool_call(
            "code_executor", {"language": "python", "code": "print('hello')"}
        )  # Use native tool call
        scenario.add_direct_response(
            '{"type": "tool_result", "tool_id": "code_executor", "result": "Here\'s the working code solution:"}'
        )

        mock_llm = MockLLMProvider({}).set_scenario(scenario)

        agent = Agent(
            llm=mock_llm,
            messager=mock_messager,
            role="coder",
            system_prompt="You are a coding assistant that writes clean, efficient code.",
        )

        # Mock successful code execution
        with patch.object(
            agent._tool_manager, "get_tool_results", new_callable=AsyncMock
        ) as mock_get_results:
            mock_tool_result = TaskResultMessage(
                tool_id="code_executor",
                tool_name="code_executor",
                task_id="code_123",
                status=TaskStatus.SUCCESS,
                result="hello\n",
            )
            mock_get_results.return_value = ([mock_tool_result], False)

            result = await agent.query("Write a Python script that prints hello world")

            assert "code" in result.lower()
            mock_llm.assert_prompt_contains("python")

    @pytest.mark.asyncio
    async def test_error_recovery_scenario(self, mock_messager):
        """Test agent error recovery."""
        scenario = MockScenario("error_recovery")
        scenario.add_direct_response("dummy")  # Dummy for indexing bug
        scenario.add_tool_call("failing_tool", {"param": "value"})  # Use native tool call
        scenario.add_direct_response(
            '{"type": "direct", "content": "The tool failed, but I can provide an alternative solution."}'
        )

        mock_llm = MockLLMProvider({}).set_scenario(scenario)

        agent = Agent(llm=mock_llm, messager=mock_messager, role="resilient_agent")

        # Mock tool failure then recovery
        with patch.object(
            agent._tool_manager, "get_tool_results", new_callable=AsyncMock
        ) as mock_get_results:
            from argentic.core.protocol.task import TaskErrorMessage

            mock_error_result = TaskErrorMessage(
                tool_id="failing_tool",
                tool_name="failing_tool",
                task_id="fail_123",
                error="Tool execution failed",
            )
            mock_get_results.return_value = ([mock_error_result], True)

            result = await agent.query("Try using the failing tool")

            assert "alternative solution" in result
            mock_get_results.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
