import asyncio
import json
import uuid
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage

from argentic.core.llm.providers.base import ModelProvider
from argentic.core.logger import get_logger, LogLevel


class MockResponseType(Enum):
    """Types of responses the mock can generate."""

    DIRECT = "direct"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ERROR = "error"


@dataclass
class MockToolCall:
    """Represents a mock tool call."""

    name: str
    args: Dict[str, Any]
    id: Optional[str] = None

    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())


@dataclass
class MockResponse:
    """Represents a predefined mock response."""

    response_type: MockResponseType
    content: str = ""
    tool_calls: List[MockToolCall] = field(default_factory=list)
    error_message: Optional[str] = None

    def to_langchain_message(self) -> AIMessage:
        """Convert to LangChain AIMessage."""
        if self.response_type == MockResponseType.ERROR:
            raise Exception(self.error_message or "Mock LLM error")

        if self.response_type == MockResponseType.TOOL_CALL and self.tool_calls:
            # Convert to LangChain tool call format
            lc_tool_calls = [
                {"name": tc.name, "args": tc.args, "id": tc.id, "type": "tool_call"}
                for tc in self.tool_calls
            ]
            return AIMessage(content=self.content, tool_calls=lc_tool_calls)

        return AIMessage(content=self.content)


@dataclass
class MockScenario:
    """Represents a testing scenario with multiple exchanges."""

    name: str
    exchanges: List[MockResponse] = field(default_factory=list)
    expected_prompts: List[str] = field(default_factory=list)

    def add_response(self, response: MockResponse):
        """Add a response to this scenario."""
        self.exchanges.append(response)
        return self

    def add_direct_response(self, content: str):
        """Add a direct text response."""
        return self.add_response(MockResponse(MockResponseType.DIRECT, content=content))

    def add_tool_call(self, tool_name: str, tool_args: Dict[str, Any], content: str = ""):
        """Add a tool call response."""
        tool_call = MockToolCall(name=tool_name, args=tool_args)
        return self.add_response(
            MockResponse(MockResponseType.TOOL_CALL, content=content, tool_calls=[tool_call])
        )

    def add_error(self, error_message: str):
        """Add an error response."""
        return self.add_response(MockResponse(MockResponseType.ERROR, error_message=error_message))


class MockLLMProvider(ModelProvider):
    """
    Mock LLM provider for testing that supports:
    - Predefined responses and tool calls
    - Prompt capture and validation
    - Scenario-based testing
    - Error simulation
    """

    def __init__(self, config: Dict[str, Any], messager: Optional[Any] = None):
        super().__init__(config, messager)
        self.logger = get_logger("mock_llm", LogLevel.INFO)

        # State tracking
        self.call_count = 0
        self.captured_prompts: List[str] = []
        self.captured_messages: List[List[BaseMessage]] = []
        self.captured_tools: List[List[Any]] = []

        # Response configuration
        self.responses: List[MockResponse] = []
        self.current_scenario: Optional[MockScenario] = None
        self.response_mode = "sequential"  # "sequential", "cycle", "random"

        # Behavior configuration
        self.simulate_delay = False
        self.delay_range = (0.1, 0.5)  # seconds
        self.failure_rate = 0.0  # 0.0 to 1.0

        # Validation functions
        self.prompt_validators: List[Callable[[str], bool]] = []

        # Initialize with default responses if none provided
        if not self.responses:
            self._setup_default_responses()

    def _setup_default_responses(self):
        """Setup default responses for basic testing."""
        self.responses = [
            MockResponse(MockResponseType.DIRECT, content="Mock response 1"),
            MockResponse(MockResponseType.DIRECT, content="Mock response 2"),
            MockResponse(MockResponseType.DIRECT, content="Mock response 3"),
        ]

    def set_responses(self, responses: List[MockResponse]):
        """Set predefined responses."""
        self.responses = responses
        return self

    def set_scenario(self, scenario: MockScenario):
        """Set a testing scenario."""
        self.current_scenario = scenario
        self.responses = scenario.exchanges
        return self

    def add_response(self, response: MockResponse):
        """Add a single response."""
        self.responses.append(response)
        return self

    def add_direct_response(self, content: str):
        """Add a direct text response."""
        return self.add_response(MockResponse(MockResponseType.DIRECT, content=content))

    def add_tool_call_response(self, tool_name: str, tool_args: Dict[str, Any], content: str = ""):
        """Add a tool call response."""
        tool_call = MockToolCall(name=tool_name, args=tool_args)
        return self.add_response(
            MockResponse(MockResponseType.TOOL_CALL, content=content, tool_calls=[tool_call])
        )

    def add_error_response(self, error_message: str):
        """Add an error response."""
        return self.add_response(MockResponse(MockResponseType.ERROR, error_message=error_message))

    def add_prompt_validator(self, validator: Callable[[str], bool]):
        """Add a prompt validation function."""
        self.prompt_validators.append(validator)
        return self

    def validate_prompt_contains(self, text: str):
        """Add a validator to check if prompt contains specific text."""
        return self.add_prompt_validator(lambda prompt: text.lower() in prompt.lower())

    def validate_prompt_not_contains(self, text: str):
        """Add a validator to check if prompt doesn't contain specific text."""
        return self.add_prompt_validator(lambda prompt: text.lower() not in prompt.lower())

    def get_next_response(self) -> MockResponse:
        """Get the next response based on the current mode."""
        if not self.responses:
            self.logger.warning("No responses configured, returning default")
            return MockResponse(MockResponseType.DIRECT, content="Default mock response")

        if self.response_mode == "sequential":
            idx = min(self.call_count, len(self.responses) - 1)
        elif self.response_mode == "cycle":
            idx = self.call_count % len(self.responses)
        else:  # random
            import random

            idx = random.randint(0, len(self.responses) - 1)

        return self.responses[idx]

    def _capture_call(self, messages: List[BaseMessage], tools: Optional[List[Any]] = None):
        """Capture call details for validation."""
        self.call_count += 1
        self.captured_messages.append(messages.copy() if messages else [])
        self.captured_tools.append(tools.copy() if tools else [])

        # Capture prompt text
        if messages:
            prompt_text = "\n".join(
                [
                    f"{msg.__class__.__name__}: {getattr(msg, 'content', str(msg))}"
                    for msg in messages
                ]
            )
            self.captured_prompts.append(prompt_text)

            # Run prompt validators
            for validator in self.prompt_validators:
                if not validator(prompt_text):
                    self.logger.warning(f"Prompt validation failed for: {prompt_text[:100]}...")

    def _simulate_delay(self):
        """Simulate processing delay if configured."""
        if self.simulate_delay:
            import random

            delay = random.uniform(*self.delay_range)
            return asyncio.sleep(delay)
        return asyncio.sleep(0)

    def _check_failure_simulation(self):
        """Check if we should simulate a failure."""
        if self.failure_rate > 0:
            import random

            if random.random() < self.failure_rate:
                raise Exception("Simulated LLM failure")

    async def _make_call(
        self, messages: List[BaseMessage], tools: Optional[List[Any]] = None
    ) -> AIMessage:
        """Internal method to make a mock call."""
        await self._simulate_delay()
        self._check_failure_simulation()

        self._capture_call(messages, tools)

        response = self.get_next_response()
        return response.to_langchain_message()

    # Implementation of abstract methods from ModelProvider

    def invoke(self, prompt: str, **kwargs: Any) -> BaseMessage:
        """Synchronously invoke with a single prompt."""
        messages: List[BaseMessage] = [HumanMessage(content=prompt)]
        try:
            return asyncio.run(self._make_call(messages, **kwargs))
        except RuntimeError:
            # If we're already in an event loop
            raise RuntimeError(
                "Cannot use invoke() from within an async context. Use ainvoke() instead."
            )

    async def ainvoke(self, prompt: str, **kwargs: Any) -> BaseMessage:
        """Asynchronously invoke with a single prompt."""
        messages: List[BaseMessage] = [HumanMessage(content=prompt)]
        return await self._make_call(messages, **kwargs)

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> BaseMessage:
        """Synchronously invoke with chat messages."""
        langchain_messages = self._convert_dict_messages_to_langchain(messages)
        try:
            return asyncio.run(self._make_call(langchain_messages, **kwargs))
        except RuntimeError:
            raise RuntimeError(
                "Cannot use chat() from within an async context. Use achat() instead."
            )

    async def achat(
        self, messages: List[Dict[str, str]], tools: Optional[List[Any]] = None, **kwargs: Any
    ) -> BaseMessage:
        """Asynchronously invoke with chat messages."""
        langchain_messages = self._convert_dict_messages_to_langchain(messages)
        return await self._make_call(langchain_messages, tools=tools, **kwargs)

    def _convert_dict_messages_to_langchain(
        self, messages: List[Dict[str, str]]
    ) -> List[BaseMessage]:
        """Convert dictionary messages to LangChain BaseMessage objects."""
        langchain_messages = []

        for msg in messages:
            role = msg.get("role", "user").lower()
            content = msg.get("content", "")

            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role in ["assistant", "model"]:
                langchain_messages.append(AIMessage(content=content))
            elif role == "tool":
                tool_call_id = msg.get("tool_call_id")
                if tool_call_id:
                    langchain_messages.append(
                        ToolMessage(content=content, tool_call_id=tool_call_id)
                    )
                else:
                    langchain_messages.append(HumanMessage(content=f"Tool output: {content}"))
            else:
                langchain_messages.append(HumanMessage(content=f"{role}: {content}"))

        return langchain_messages

    # Validation and assertion methods for testing

    def assert_called(self, times: Optional[int] = None):
        """Assert that the mock was called."""
        if times is None:
            assert (
                self.call_count > 0
            ), f"Expected mock to be called, but it was called {self.call_count} times"
        else:
            assert self.call_count == times, f"Expected {times} calls, but got {self.call_count}"

    def assert_prompt_contains(self, text: str, call_index: int = -1):
        """Assert that a captured prompt contains specific text."""
        if not self.captured_prompts:
            raise AssertionError("No prompts captured")

        prompt = self.captured_prompts[call_index]
        assert (
            text.lower() in prompt.lower()
        ), f"Expected prompt to contain '{text}', but got: {prompt[:200]}..."

    def assert_prompt_not_contains(self, text: str, call_index: int = -1):
        """Assert that a captured prompt doesn't contain specific text."""
        if not self.captured_prompts:
            raise AssertionError("No prompts captured")

        prompt = self.captured_prompts[call_index]
        assert (
            text.lower() not in prompt.lower()
        ), f"Expected prompt to not contain '{text}', but got: {prompt[:200]}..."

    def assert_tools_provided(self, call_index: int = -1):
        """Assert that tools were provided in a call."""
        if not self.captured_tools:
            raise AssertionError("No tool calls captured")

        tools = self.captured_tools[call_index]
        assert tools is not None and len(tools) > 0, "Expected tools to be provided"

    def get_captured_prompt(self, call_index: int = -1) -> str:
        """Get a captured prompt by index."""
        if not self.captured_prompts:
            raise IndexError("No prompts captured")
        return self.captured_prompts[call_index]

    def get_captured_messages(self, call_index: int = -1) -> List[BaseMessage]:
        """Get captured messages by index."""
        if not self.captured_messages:
            raise IndexError("No messages captured")
        return self.captured_messages[call_index]

    def reset(self):
        """Reset all captured state."""
        self.call_count = 0
        self.captured_prompts.clear()
        self.captured_messages.clear()
        self.captured_tools.clear()
        return self

    # Utility methods for common test scenarios

    @classmethod
    def create_simple_agent_scenario(cls) -> "MockLLMProvider":
        """Create a mock provider for simple agent testing."""
        mock = cls({})
        mock.add_direct_response("I'll help you with that task.")
        mock.add_tool_call_response("search_tool", {"query": "test query"}, "I'll search for that.")
        mock.add_direct_response("Based on the search results, here's the answer.")
        return mock

    @classmethod
    def create_supervisor_scenario(cls) -> "MockLLMProvider":
        """Create a mock provider for supervisor testing."""
        mock = cls({})
        mock.add_direct_response("researcher")  # Route to researcher
        mock.add_direct_response("coder")  # Route to coder
        mock.add_direct_response("__end__")  # End conversation
        return mock

    @classmethod
    def create_error_scenario(cls) -> "MockLLMProvider":
        """Create a mock provider for error testing."""
        mock = cls({})
        mock.add_error_response("Rate limit exceeded")
        mock.add_error_response("Connection timeout")
        mock.add_direct_response("Recovery response")
        return mock
