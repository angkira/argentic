# Argentic
![Argentic Logo](./assets/logo.jpg){: .styled-logo }
[![Python application](https://github.com/angkira/argentic/actions/workflows/python-app.yml/badge.svg)](https://github.com/angkira/argentic/actions/workflows/python-app.yml)

A microframework for building and running local AI agents.

Argentic provides a lightweight, configurable framework designed to simplify the setup and operation of local AI agents. It integrates with various Large Language Model (LLM) backends and utilizes a messaging protocol (currently MQTT) for flexible communication between the core agent, tools, and clients.

## Features

- **Modular Design**: Core components include an `Agent`, a `Messager` for communication, and an `LLMProvider` for interacting with language models.
- **Multiple LLM Backends**: Supports various LLMs through a factory pattern, including:
  - Ollama (via `ollama` Python library)
  - Llama.cpp (via HTTP server or direct CLI interaction)
  - Google Gemini (via API)
- **Configuration Driven**: Easily configure LLM providers, messaging brokers (MQTT), communication topics, and logging via `config.yaml`.
- **Command-Line Interface**: Start different components (agent, example tools, CLI client) using `start.sh`. Configure config path and log level via CLI arguments (`--config-path`, `--log-level`) or environment variables (`CONFIG_PATH`, `LOG_LEVEL`).
- **Messaging Protocol**: Uses MQTT for decoupled communication between the agent and potential tools or clients. Includes message classes for defined interactions (e.g., `AskQuestionMessage`).
- **Extensible Tool System**: Designed to integrate external tools via messaging. Includes an example RAG (Retrieval-Augmented Generation) tool (`src/services/rag_tool_service.py`) demonstrating this capability.
- **CLI Client**: A simple command-line client (`src/cli_client.py`) for interacting with the agent.
- **Graceful Shutdown**: Handles termination signals for proper cleanup.

## Getting Started

1. **Clone the repository:**

   ```bash
   git clone https://github.com/angkira/argentic.git
   cd argentic
   ```

2. **Set up Python environment:**
   You have two options:

   **Option 1: Using the installation script**

   ```bash
   # This will create a virtual environment and install the package in development mode
   ./install.sh
   source .venv/bin/activate
   ```

   **Option 2: Manual setup**
   It's recommended to use a virtual environment. The project uses `uv` (or `pip`) and `pyproject.toml`.

   ```bash
   # Using uv
   uv venv
   uv sync
   source .venv/bin/activate # Or your environment's activation script

   # Or using pip
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```

3. **Configure:**
   - Copy or rename `config.yaml.example` to `config.yaml` (if an example exists) or edit `config.yaml` directly.
   - Set up your desired LLM provider (`llm` section).
   - Configure the MQTT broker details (`messaging` section).
   - Set any required API keys or environment variables (e.g., `GOOGLE_GEMINI_API_KEY` if using Gemini). Refer to `.env.example` if provided.
4. **Run Components:**
   Use the `start.sh` script:

   - Run the main agent:

     ```bash
     ./start.sh agent [--config-path path/to/config.yaml] [--log-level DEBUG]
     ```

   - Run the example RAG tool service (optional, in a separate terminal):

     ```bash
     ./start.sh rag
     ```

   - Run the CLI client to interact (optional, in a separate terminal):

     ```bash
     ./start.sh cli
     ```

## Using as a Python Package

After installation, you can import the Argentic components in your Python code using simplified imports:

```python
# Option 1: Import directly from the main package
from argentic import Agent, Messager, LLMFactory

# Option 2: Import from specific modules with reduced nesting
from argentic.core import Agent, Messager, LLMFactory

# Option 3: Import specific tools
from argentic.tools import BaseTool, ToolManager
```

You can also create custom tools or extend the core functionality by subclassing the base classes.

## Configuration (`config.yaml`)

The `config.yaml` file controls the application's behavior:

- `llm`: Defines the LLM provider to use and its specific settings. Set the `provider` key to one of the supported names below:
  - `provider: ollama`
    - `ollama_model_name`: (Required) The name of the model served by Ollama (e.g., `gemma3:12b-it-qat`).
    - `ollama_use_chat_model`: (Optional, boolean, default: `true`) Whether to use Ollama's chat completion endpoint.
  - `provider: llama_cpp_server`
    - `llama_cpp_server_binary`: (Optional) Path to the `llama-server` executable (needed if `auto_start` is true).
    - `llama_cpp_server_args`: (Optional, list) Arguments to pass when auto-starting the server (e.g., model path, host, port).
    - `llama_cpp_server_host`: (Required) Hostname or IP address of the running llama.cpp server (e.g., `127.0.0.1`).
    - `llama_cpp_server_port`: (Required) Port number of the running llama.cpp server (e.g., `5000`).
    - `llama_cpp_server_auto_start`: (Optional, boolean, default: `false`) Whether Argentic should try to start the `llama-server` process itself.
  - `provider: llama_cpp_cli`
    - `llama_cpp_cli_binary`: (Required) Path to the `llama.cpp` main CLI executable (e.g., `~/llama.cpp/build/bin/llama-gemma3-cli`).
    - `llama_cpp_cli_model_path`: (Required) Path to the GGUF model file.
    - `llama_cpp_cli_args`: (Optional, list) Additional arguments to pass to the CLI (e.g., `--temp 0.7`, `--n-predict 128`).
  - `provider: google_gemini`
    - `google_gemini_api_key`: (Required) Your Google Gemini API key. **It is strongly recommended to set this via the `GOOGLE_GEMINI_API_KEY` environment variable instead of directly in the file.** Argentic uses `python-dotenv` to load variables from a `.env` file.
    - `google_gemini_model_name`: (Required) The specific Gemini model to use (e.g., `gemini-2.0-flash`).
- `messaging`: Configures the messaging protocol (e.g., `mqtt`) and connection details (broker address, port, credentials).
- `topics`: Defines the MQTT topics used for commands, responses, logging, and tool communication.

## Tools

Argentic supports interaction with external tools via the configured messaging system. Tools run as independent services and communicate with the main agent.

**Tool Registration Process:**

1. **Tool-Side (`BaseTool`):**
   - A tool service (like `rag_tool_service.py`) instantiates a tool class derived from `core.tools.tool_base.BaseTool`.
   - It calls the `tool.register()` method, providing the relevant messaging topics from the configuration (`register`, `status`, `call`, `response_base`).
   - The tool publishes a `RegisterToolMessage` (containing its name, description/manual, and Pydantic schema for arguments) to the agent's registration topic (e.g., `agent/tools/register`).
   - The tool simultaneously subscribes to the agent's status topic (e.g., `agent/status/info`) to await a `ToolRegisteredMessage` confirmation.
2. **Agent-Side (`ToolManager`):**
   - The `ToolManager` (within the main agent) listens on the registration topic.
   - Upon receiving a `RegisterToolMessage`, it generates a unique `tool_id` for the tool.
   - It stores the tool's metadata (ID, name, description, API schema).
   - The `ToolManager` subscribes to the tool's specific result topic (e.g., `agent/tools/response/<generated_tool_id>`) to listen for task outcomes.
   - It publishes the `ToolRegisteredMessage` (including the `tool_id`) back to the agent's status topic, confirming registration with the tool.
3. **Tool-Side (Confirmation):**
   - The tool receives the `ToolRegisteredMessage`, stores its assigned `tool_id`.
   - It then subscribes to its dedicated task topic (e.g., `agent/tools/call/<generated_tool_id>`) to listen for incoming tasks.

**Task Execution Flow:**

1. **Agent Needs Tool:** The agent (likely prompted by the LLM) decides to use a tool.
2. **Agent Executes Task (`ToolManager.execute_tool`):**
   - The agent calls `tool_manager.execute_tool(tool_name_or_id, arguments)`.
   - The `ToolManager` creates a `TaskMessage` (containing a unique `task_id`, the `tool_id`, and the arguments).
   - It publishes this `TaskMessage` to the specific tool's task topic (e.g., `agent/tools/call/<tool_id>`).
   - It waits asynchronously for a response message associated with the `task_id` on the tool's result topic.
3. **Tool Executes Task (`BaseTool._handle_task_message`):**
   - The tool service receives the `TaskMessage` on its task topic.
   - It validates the arguments using the tool's Pydantic schema.
   - It executes the tool's core logic (`_execute` method).
   - It creates a `TaskResultMessage` (on success) or `TaskErrorMessage` (on failure), including the original `task_id`.
   - It publishes this result message to its result topic (e.g., `agent/tools/response/<tool_id>`).
4. **Agent Receives Result (`ToolManager._handle_result_message`):**
   - The `ToolManager` receives the result message on the tool's result topic.
   - It matches the `task_id` to the pending asynchronous task and delivers the result (or error) back to the agent's logic that initiated the call.

An example `rag_tool_service.py` demonstrates how a tool (`KnowledgeBaseTool`) can be built and run independently, registering and communicating with the agent using this messaging pattern.

## Testing

The project includes a comprehensive test suite organized into categories:

### Test Structure

- **Unit Tests**: Located in `tests/core/messager/unit/`, these tests verify individual components in isolation.
- **Integration Tests**: Located in `tests/core/messager/test_messager_integration.py`, these tests verify how components work together.
- **End-to-End Tests**: Located in `tests/core/messager/e2e/`, these tests verify the system behavior using actual message brokers via Docker.

### Running Tests

Several scripts are available in the `bin/` directory to run different types of tests:

- **All Tests**: Run the complete test suite with the main test script:

  ```bash
  ./bin/run_tests.sh
  ```

- **Unit Tests Only**: Run only the unit tests:

  ```bash
  ./bin/run_unit_tests.sh
  ```

- **E2E Tests Only**: Run only the end-to-end tests (requires Docker):

  ```bash
  ./bin/run_e2e_tests.sh
  ```

  The E2E test script supports Docker container management:

  ```bash
  # Start Docker containers before running tests
  ./bin/run_e2e_tests.sh --start-docker

  # Start Docker, run tests, and stop containers afterward
  ./bin/run_e2e_tests.sh --start-docker --stop-docker

  # Only start Docker containers without running tests
  ./bin/run_e2e_tests.sh --docker-only --start-docker

  # Only stop Docker containers
  ./bin/run_e2e_tests.sh --docker-only --stop-docker

  # Pass additional arguments to pytest after --
  ./bin/run_e2e_tests.sh --start-docker -- -v
  ```

- **Integration Tests Only**: Run only the integration tests:
  ```bash
  ./bin/run_integration_tests.sh
  ```

Each script accepts additional pytest arguments. For example, to run tests with higher verbosity:

```bash
./bin/run_unit_tests.sh -v
```

### Test Markers

The tests use markers to categorize different test types:

- `@pytest.mark.e2e`: Marks tests that require external dependencies (Docker containers)
- `@pytest.mark.slow`: Marks tests that take a long time to execute
- `@pytest.mark.kafka`: Marks Kafka-specific tests

You can use these markers with pytest's `-m` option to run specific test categories:

```bash
python -m pytest -m "e2e and not kafka"
```

## Documentation

More detailed documentation is planned, potentially hosted on GitHub Pages, covering architecture, API details, and tool development guidelines.

For development, clone the repository and install in editable mode with development dependencies:
```bash
git clone https://github.com/yourusername/argentic.git
cd argentic
# Create and activate a virtual environment (recommended)
# python -m venv .venv 
# source .venv/bin/activate  # On Windows use .venv\Scripts\activate

# Install the project in editable mode with development dependencies
uv pip install -e .[dev]
```
