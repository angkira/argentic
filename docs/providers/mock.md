# Mock LLM Provider

`MockLLMProvider` is a fully in-memory fake model intended for unit-tests and offline demos.

---

## Why it exists
* Removes all external dependencies → no network, no model weights.
* Provides deterministic, programmable responses to validate agent logic.
* Captures prompts, messages, tool usage and call counts for assertions.

---

## Features
* Predefined response list (direct text, tool call, tool result, error).
* Scenario helper to script multi-turn interactions.
* Prompt validators (`assert_prompt_contains`, etc.).
* Failure simulation (`failure_rate`, artificial delay).

---

## Requirements
None – ships with Argentic.

---

## Configuration
No special keys; you typically instantiate `MockLLMProvider({})` and then programmatic methods configure behaviour.

```python
mock = MockLLMProvider({})
mock.add_direct_response("Hello!")
mock.add_tool_call_response("search", {"query": "test"})
```

If you still need to use the config-dict style:
```yaml
llm: {}
```

---

## Environment variables
None.

---

## Example (pytest)
```python
from argentic.core.llm.providers.mock import MockLLMProvider

provider = MockLLMProvider.create_simple_agent_scenario()

reply = provider.invoke("Hi")
assert reply.content == "I'll help you with that task."

# Validate prompt captured
provider.assert_called()
provider.assert_prompt_contains("Hi")
``` 