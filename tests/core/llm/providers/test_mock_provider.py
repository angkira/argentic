import pytest

from argentic.core.llm.providers.mock import (
    MockLLMProvider,
    MockResponse,
    MockResponseType,
    MockToolCall,
)
from argentic.core.protocol.chat_message import AssistantMessage, LLMChatResponse


@pytest.fixture
def mock_provider():
    config = {}
    provider = MockLLMProvider(config)
    provider.reset()
    return provider


@pytest.mark.asyncio
async def test_direct_response(mock_provider):
    mock_provider.add_direct_response("Test content")

    response = await mock_provider.ainvoke("Test prompt")
    assert isinstance(response, LLMChatResponse)
    assert isinstance(response.message, AssistantMessage)
    assert response.message.content == "Test content"
    assert response.message.tool_calls is None
    assert response.usage is None
    assert response.finish_reason is None


@pytest.mark.asyncio
async def test_tool_call_response(mock_provider):
    tool_call = MockToolCall(name="test_tool", args={"param": 42})
    mock_provider.add_response(
        MockResponse(MockResponseType.TOOL_CALL, content="Calling tool", tool_calls=[tool_call])
    )

    response = await mock_provider.ainvoke("Test prompt")
    assert isinstance(response, LLMChatResponse)
    assert isinstance(response.message, AssistantMessage)
    assert response.message.content == "Calling tool"
    assert response.message.tool_calls == [
        {"name": "test_tool", "args": {"param": 42}, "id": tool_call.id}
    ]


@pytest.mark.asyncio
async def test_error_response(mock_provider):
    mock_provider.add_error_response("Simulated error")

    with pytest.raises(Exception) as exc:
        await mock_provider.ainvoke("Test prompt")
    assert str(exc.value) == "Simulated error"


@pytest.mark.asyncio
async def test_chat_response(mock_provider):
    mock_provider.add_direct_response("Chat response")

    messages = [{"role": "user", "content": "Hello"}]
    response = await mock_provider.achat(messages)
    assert isinstance(response, LLMChatResponse)
    assert response.message.content == "Chat response"


@pytest.mark.asyncio
async def test_multiple_calls(mock_provider):
    mock_provider.add_direct_response("First")
    mock_provider.add_direct_response("Second")

    resp1 = await mock_provider.ainvoke("Prompt 1")
    assert resp1.message.content == "First"

    resp2 = await mock_provider.ainvoke("Prompt 2")
    assert resp2.message.content == "Second"

    assert mock_provider.call_count == 2
    assert len(mock_provider.captured_prompts) == 2


def test_sync_invoke(mock_provider):
    mock_provider.add_direct_response("Sync response")

    response = mock_provider.invoke("Sync prompt")
    assert isinstance(response, LLMChatResponse)
    assert response.message.content == "Sync response"


def test_sync_chat(mock_provider):
    mock_provider.add_direct_response("Sync chat")

    messages = [{"role": "user", "content": "Sync hello"}]
    response = mock_provider.chat(messages)
    assert isinstance(response, LLMChatResponse)
    assert response.message.content == "Sync chat"
