from unittest.mock import AsyncMock, MagicMock

import pytest

from argentic.core.agent.agent import Agent, AgentStateMode
from argentic.core.llm.providers.mock import (
    LcHumanMessage,
    LcSystemMessage,
    MockLLMProvider,
    MockResponse,
    MockResponseType,
)
from argentic.core.messager.messager import Messager


@pytest.fixture
def mock_messager():
    messager = MagicMock(spec=Messager)
    messager.connect = AsyncMock()
    messager.publish = AsyncMock()
    messager.subscribe = AsyncMock()
    messager.disconnect = AsyncMock()
    return messager


@pytest.fixture
def direct_llm():
    mock = MockLLMProvider({})
    mock.set_responses(
        [
            MockResponse(
                MockResponseType.DIRECT,
                content='{"type": "direct", "content": "ok"}',
            )
        ]
    )
    mock.call_count = 0
    return mock


@pytest.mark.asyncio
async def test_query_stateful_includes_history(mock_messager, direct_llm):
    agent = Agent(
        llm=direct_llm,
        messager=mock_messager,
        role="test_stateful",
        system_prompt="You are a helpful assistant.",
        expected_output_format="json",
        state_mode=AgentStateMode.STATEFUL,
    )

    question = "What is stateful mode?"
    await agent.query(question)

    captured = direct_llm.get_captured_messages()

    # Should include prior context: System + original question + formatted turn prompt
    assert any(isinstance(m, LcSystemMessage) for m in captured)
    assert any(isinstance(m, LcHumanMessage) and m.content == question for m in captured)
    assert len([m for m in captured if isinstance(m, LcHumanMessage)]) >= 2


@pytest.mark.asyncio
async def test_query_stateless_sends_single_formatted_prompt(mock_messager, direct_llm):
    agent = Agent(
        llm=direct_llm,
        messager=mock_messager,
        role="test_stateless",
        system_prompt="StatelessTestPrompt",
        expected_output_format="json",
        state_mode=AgentStateMode.STATELESS,
    )

    question = "What is stateless mode?"
    await agent.query(question)

    captured = direct_llm.get_captured_messages()
    # Stateless query() should not send previous messages as separate entries
    assert all(not isinstance(m, LcSystemMessage) for m in captured)
    assert len(captured) == 1
    assert isinstance(captured[0], LcHumanMessage)
    assert "QUESTION: " in captured[0].content
    assert question in captured[0].content
    # System prompt content should be embedded into the formatted user prompt
    assert "StatelessTestPrompt" in captured[0].content


@pytest.mark.asyncio
async def test_invoke_stateful_uses_full_history(mock_messager, direct_llm):
    agent = Agent(
        llm=direct_llm,
        messager=mock_messager,
        role="invoke_stateful",
        system_prompt="InvokeStatefulPrompt",
        expected_output_format="json",
        state_mode=AgentStateMode.STATEFUL,
    )

    state = {
        "messages": [LcHumanMessage(content="First"), LcHumanMessage(content="Second")],
        "next": None,
    }
    await agent.invoke(state)

    captured = direct_llm.get_captured_messages()
    contents = [getattr(m, "content", "") for m in captured]

    # Should include system + both human messages
    assert any(isinstance(m, LcSystemMessage) for m in captured)
    assert "First" in contents
    assert "Second" in contents


@pytest.mark.asyncio
async def test_invoke_stateless_uses_only_last_message(mock_messager, direct_llm):
    agent = Agent(
        llm=direct_llm,
        messager=mock_messager,
        role="invoke_stateless",
        system_prompt="InvokeStatelessPrompt",
        expected_output_format="json",
        state_mode=AgentStateMode.STATELESS,
    )

    state = {
        "messages": [LcHumanMessage(content="First"), LcHumanMessage(content="Second")],
        "next": None,
    }
    await agent.invoke(state)

    captured = direct_llm.get_captured_messages()
    # Stateless invoke() should include system + only the most recent user message
    assert any(isinstance(m, LcSystemMessage) for m in captured)
    human_msgs = [m for m in captured if isinstance(m, LcHumanMessage)]
    assert len(human_msgs) == 1
    assert human_msgs[0].content == "Second"
    assert all(getattr(m, "content", "") != "First" for m in captured)
