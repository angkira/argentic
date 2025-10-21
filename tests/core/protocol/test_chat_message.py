import pytest

from argentic.core.protocol.chat_message import (
    AssistantMessage,
    ChatMessage,
    LLMChatResponse,
    SystemMessage,
    ToolMessage,
    UserMessage,
)


def test_system_message():
    msg = SystemMessage(role="system", content="Test system")
    assert msg.role == "system"
    assert msg.content == "Test system"
    assert msg.model_dump() == {"role": "system", "content": "Test system"}


def test_user_message():
    msg = UserMessage(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"


def test_assistant_message():
    tool_calls = [{"name": "tool1", "args": {"param": 1}, "id": "123"}]
    msg = AssistantMessage(role="assistant", content="Response", tool_calls=tool_calls)
    assert msg.role == "assistant"
    assert msg.content == "Response"
    assert msg.tool_calls == tool_calls

    # Without tool calls
    msg2 = AssistantMessage(role="assistant", content="Simple response")
    assert msg2.tool_calls is None


def test_tool_message():
    msg = ToolMessage(role="tool", content="Result: success", tool_call_id="123")
    assert msg.role == "tool"
    assert msg.content == "Result: success"
    assert msg.tool_call_id == "123"

    # Without tool_call_id
    msg2 = ToolMessage(role="tool", content="Error")
    assert msg2.tool_call_id is None


def test_llm_chat_response():
    assistant = AssistantMessage(content="Answer")
    response = LLMChatResponse(
        message=assistant, usage={"prompt_tokens": 10, "completion_tokens": 5}, finish_reason="stop"
    )
    assert response.message.content == "Answer"
    assert response.usage == {"prompt_tokens": 10, "completion_tokens": 5}
    assert response.finish_reason == "stop"

    # Minimal
    minimal = LLMChatResponse(message=AssistantMessage(content="Minimal"))
    assert minimal.usage is None
    assert minimal.finish_reason is None


def test_validation_errors():
    with pytest.raises(ValueError):  # Pydantic validation error
        SystemMessage(role="user", content="Invalid")  # role should be "system"

    with pytest.raises(ValueError):
        UserMessage(role="assistant", content="Invalid")

    with pytest.raises(ValueError):
        AssistantMessage(role="tool", content="Invalid")

    with pytest.raises(ValueError):
        ToolMessage(role="system", content="Invalid")

    with pytest.raises(ValueError):
        ChatMessage(
            role="invalid", content="Test"
        )  # invalid role not caught by base, but we can add base validator if needed


@pytest.mark.parametrize(
    "msg_class, valid_role",
    [
        (SystemMessage, "system"),
        (UserMessage, "user"),
        (AssistantMessage, "assistant"),
        (ToolMessage, "tool"),
    ],
)
def test_serialization(msg_class, valid_role):
    kwargs = {"role": valid_role, "content": "Test"}
    if msg_class == AssistantMessage:
        kwargs["tool_calls"] = []
    elif msg_class == ToolMessage:
        kwargs["tool_call_id"] = "id"

    msg = msg_class(**kwargs)

    json_data = msg.model_dump_json()
    assert isinstance(json_data, str)
    reconstructed = msg_class.model_validate_json(json_data)
    assert reconstructed.content == "Test"
    assert reconstructed.role == valid_role
