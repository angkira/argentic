# Argentic Testing Framework

This directory contains comprehensive tests for the Argentic multi-agent system, including unit tests, integration tests, and end-to-end tests.

## Overview

The testing framework is designed to provide:

- **Fast, reliable feedback** during development
- **Deterministic test results** without expensive LLM API calls  
- **Comprehensive coverage** of all system components
- **Realistic scenarios** that mirror production usage

## Test Structure

```
tests/
├── unit/                    # Unit tests for individual components
│   ├── test_agent.py       # Agent class tests
│   ├── test_supervisor.py  # Supervisor class tests
│   └── test_mock_tools.py  # Mock tools and utilities tests
├── integration/             # Integration tests for component interaction
│   └── test_multi_agent_workflow.py  # Multi-agent workflow tests
└── README.md               # This file
```

## Mock LLM Provider

The core of our testing strategy is the `MockLLMProvider` which simulates LLM behavior without making real API calls:

### Features

- **Predefined responses**: Set up specific responses for different scenarios
- **Tool call simulation**: Mock tool calling behavior 
- **Prompt capture**: Capture and validate prompts sent to the LLM
- **Error simulation**: Test error handling with simulated failures
- **Scenario-based testing**: Define complex multi-step interactions

### Example Usage

```python
from argentic.core.llm.providers.mock import MockLLMProvider, MockScenario

# Simple direct response
mock_llm = MockLLMProvider({})
mock_llm.add_direct_response("I can help you with that task.")

# Tool calling scenario
mock_llm.add_tool_call_response("search_tool", {"query": "test"}, "I'll search for that.")

# Complex scenario
scenario = MockScenario("research_task")
scenario.add_direct_response("I'll help you research that topic.")
scenario.add_tool_call("search_tool", {"query": "quantum computing"})
scenario.add_direct_response("Based on my research...")

mock_llm.set_scenario(scenario)
```

## Mock Tools

We provide comprehensive mock tools that simulate real tool behavior:

- `MockSearchTool`: Simulates web search functionality
- `MockCalculatorTool`: Simulates mathematical calculations
- `MockCodeExecutorTool`: Simulates code execution
- `MockFileSystemTool`: Simulates file system operations
- `MockToolManager`: Manages and orchestrates mock tools

### Example Usage

```python
from tests.unit.test_mock_tools import MockSearchTool, MockToolManager

# Create and configure mock tools
search_tool = MockSearchTool()
search_tool.set_search_results(["Result 1", "Result 2"])

# Use with tool manager
tool_manager = MockToolManager()
tool_manager.register_tool(search_tool)

# Execute tools
result = await tool_manager.execute_tool("search_tool", {"query": "test"})
```

## Running Tests

### Quick Start

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests
python run_tests.py all

# Run specific test types
python run_tests.py unit           # Unit tests only
python run_tests.py integration    # Integration tests only
python run_tests.py fast          # Fast tests only
```

### Advanced Usage

```bash
# Run with coverage
python run_tests.py all --coverage --html-report

# Run specific test file
python run_tests.py unit --file tests/unit/test_agent.py

# Run tests matching pattern
python run_tests.py unit --function test_agent_initialization

# Run in parallel
python run_tests.py all --parallel

# Run only failed tests from last run
python run_tests.py unit --last-failed
```

### Direct pytest Usage

```bash
# Run all tests
pytest tests/

# Run unit tests with coverage
pytest tests/unit/ --cov=src/argentic --cov-report=html

# Run specific test
pytest tests/unit/test_agent.py::TestAgent::test_agent_initialization -v

# Run tests with markers
pytest -m "not slow" tests/
```

## Test Categories

### Unit Tests (`tests/unit/`)

Test individual components in isolation using mocks:

- **Agent tests**: LLM interaction, tool calling, message handling
- **Supervisor tests**: Routing logic, agent coordination, graph compilation
- **Mock tool tests**: Tool behavior, error handling, state management

**Characteristics:**
- Fast execution (< 1 second per test)
- No external dependencies
- Deterministic results
- High code coverage

### Integration Tests (`tests/integration/`)

Test component interactions and workflows:

- **Multi-agent workflows**: Complete supervisor + agent interactions
- **Tool integration**: Agent + tool manager + tools working together
- **Message flow**: End-to-end message passing and handling
- **Error recovery**: Error handling across component boundaries

**Characteristics:**
- Moderate execution time (1-10 seconds per test)
- Test realistic scenarios
- Use mock external services
- Focus on component interaction

### End-to-End Tests

Test complete user scenarios:

- **Research workflows**: Question → research → answer
- **Coding workflows**: Request → code generation → execution
- **Complex workflows**: Multi-step tasks with multiple agents

**Characteristics:**
- Longer execution time (10+ seconds per test)
- Mirror real user interactions
- Test the entire system stack
- Validate user-facing functionality

## Test Markers

Tests are organized using pytest markers:

- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.integration`: Integration tests  
- `@pytest.mark.e2e`: End-to-end tests
- `@pytest.mark.slow`: Tests that take longer to run
- `@pytest.mark.agent`: Agent-specific tests
- `@pytest.mark.supervisor`: Supervisor-specific tests
- `@pytest.mark.tools`: Tool-related tests
- `@pytest.mark.workflow`: Workflow tests

## Best Practices

### Writing Tests

1. **Use descriptive test names**: `test_agent_handles_tool_execution_errors`
2. **Test one thing at a time**: Each test should verify a single behavior
3. **Use appropriate mocks**: Mock external dependencies, not internal logic
4. **Include both happy path and error cases**
5. **Test edge cases and boundary conditions**

### Mock Configuration

1. **Reset mocks between tests**: Use `mock.reset()` in test setup
2. **Capture and validate inputs**: Verify prompts and tool arguments
3. **Use realistic responses**: Mimic actual LLM/tool behavior
4. **Test error conditions**: Include failure scenarios

### Performance

1. **Keep unit tests fast**: < 1 second per test
2. **Use `@pytest.mark.slow` for longer tests**
3. **Run fast tests during development**
4. **Run full suite in CI/CD**

## Test Data and Fixtures

### Common Fixtures

- `mock_messager`: Mock messager for testing
- `mock_llm_*`: Various LLM mock configurations
- `mock_tool_manager`: Tool manager with registered tools
- `*_agent`: Pre-configured agent instances

### Test Scenarios

Predefined scenarios for common testing patterns:

- `MockLLMProvider.create_simple_agent_scenario()`
- `MockLLMProvider.create_supervisor_scenario()`
- `MockLLMProvider.create_error_scenario()`

## Debugging Tests

### Verbose Output

```bash
# Extra verbose
python run_tests.py unit --verbose

# Show print statements
pytest tests/ -s

# Show full tracebacks
pytest tests/ --tb=long
```

### Debugging Tools

```bash
# Run with pdb on failures
pytest tests/ --pdb

# Time slow tests
pytest tests/ --durations=10

# Run only failed tests
pytest tests/ --lf
```

### Mock Debugging

```python
# Check mock calls
mock_llm.assert_called(times=2)
mock_llm.assert_prompt_contains("search query")

# Inspect captured data
print(mock_llm.get_captured_prompt())
print(mock_tool.call_history)
```

## Continuous Integration

The testing framework is designed to work well in CI/CD environments:

### GitHub Actions Example

```yaml
- name: Run tests
  run: |
    pip install -r requirements-test.txt
    python run_tests.py all --coverage
    
- name: Upload coverage
  uses: codecov/codecov-action@v3
```

### Test Strategy for CI

1. **Fast feedback**: Run unit tests on every commit
2. **Full validation**: Run all tests on pull requests
3. **Coverage reporting**: Track test coverage over time
4. **Performance monitoring**: Watch for slow tests

## Adding New Tests

### For New Components

1. Create test file: `tests/unit/test_new_component.py`
2. Add comprehensive unit tests
3. Create integration tests if needed
4. Update this README if introducing new patterns

### For New Scenarios

1. Define test scenarios in integration tests
2. Create new mock tools if needed
3. Add end-to-end tests for user-facing features
4. Document new test patterns

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure `PYTHONPATH` includes `src/`
2. **Async test failures**: Use `@pytest.mark.asyncio`
3. **Mock not working**: Check mock reset and configuration
4. **Slow tests**: Use markers to separate fast/slow tests

### Getting Help

1. Check existing test patterns for examples
2. Review mock provider documentation
3. Run tests with verbose output for debugging
4. Use pytest's built-in debugging tools

## Future Improvements

- [ ] Property-based testing with Hypothesis
- [ ] Performance benchmarking tests
- [ ] Visual regression testing for UI components
- [ ] Contract testing for API boundaries
- [ ] Chaos engineering tests for resilience 