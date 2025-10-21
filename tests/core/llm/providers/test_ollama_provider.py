from unittest.mock import AsyncMock, patch

import httpx
import pytest

from argentic.core.llm.providers.ollama import OllamaProvider
from argentic.core.protocol.chat_message import AssistantMessage, LLMChatResponse


@pytest.fixture
def ollama_config():
    return {
        "llm": {
            "ollama_model_name": "test_model",
            "ollama_base_url": "http://test:11434",
            "ollama_parameters": {"temperature": 0.7},
        }
    }


@pytest.fixture
def ollama_provider(ollama_config):
    return OllamaProvider(ollama_config)


@pytest.mark.asyncio
async def test_ainvoke(ollama_provider):
    mock_response = {
        "message": {"role": "assistant", "content": "Test response"},
        "prompt_eval_count": 10,
        "eval_count": 5,
        "done": True,
    }

    async def mock_post(*args, **kwargs):
        return httpx.Response(200, json=mock_response)

    with patch("httpx.AsyncClient.post", new=AsyncMock(side_effect=mock_post)):
        response = await ollama_provider.ainvoke("Test prompt")

        assert isinstance(response, LLMChatResponse)
        assert isinstance(response.message, AssistantMessage)
        assert response.message.content == "Test response"
        assert response.usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        assert response.finish_reason == "stop"


@pytest.mark.asyncio
async def test_achat(ollama_provider):
    messages = [{"role": "user", "content": "Hello"}]

    mock_response = {"message": {"role": "assistant", "content": "Hi there"}, "done": True}

    async def mock_post(*args, **kwargs):
        return httpx.Response(200, json=mock_response)

    with patch("httpx.AsyncClient.post", new=AsyncMock(side_effect=mock_post)):
        response = await ollama_provider.achat(messages)

        assert response.message.content == "Hi there"
        assert response.usage is None
        assert response.finish_reason == "stop"


def test_invoke_sync(ollama_provider):
    with patch("asyncio.run") as mock_run:
        mock_run.return_value = {"message": {"content": "Sync test"}}

        response = ollama_provider.invoke("Sync prompt")
        assert response.message.content == "Sync test"


def test_chat_sync(ollama_provider):
    with patch("asyncio.run") as mock_run:
        mock_run.return_value = {"message": {"content": "Sync chat test"}}

        messages = [{"role": "user", "content": "Sync hello"}]
        response = ollama_provider.chat(messages)
        assert response.message.content == "Sync chat test"


@pytest.mark.asyncio
async def test_error_handling(ollama_provider):
    async def mock_post(*args, **kwargs):
        raise httpx.HTTPError("API error")

    with patch("httpx.AsyncClient.post", new=AsyncMock(side_effect=mock_post)):
        with pytest.raises(httpx.HTTPError):
            await ollama_provider.ainvoke("Error prompt")
