# Configuration

This guide covers the configuration system in Argentic, which uses a YAML-based configuration file to control the behavior of all components.

## Overview

Argentic uses a `config.yaml` file as the primary configuration mechanism. This file controls:

- LLM provider selection and settings
- Messaging system configuration
- Tool system settings
- Logging configuration
- API keys and credentials

## Configuration File Location

By default, Argentic looks for `config.yaml` in the current working directory. You can specify a custom path using:

- Command line argument: `--config-path path/to/config.yaml`
- Environment variable: `CONFIG_PATH=path/to/config.yaml`

## Basic Configuration Structure

```yaml
# LLM Provider Configuration
llm:
  provider: "ollama" # Which LLM provider to use
  # Provider-specific settings...

# Messaging System Configuration
messaging:
  protocol: "mqtt"
  broker_address: "localhost"
  port: 1883
  # Additional messaging settings...

# Tool System Configuration
tools:
  registration:
    topic: "agent/tools/register"
  # Tool-specific settings...

# Logging Configuration
logging:
  level: "INFO"
  format: "detailed"
  # Additional logging settings...
```

## Configuration Sections

### LLM Configuration

The `llm` section defines which Large Language Model provider to use and its configuration:

```yaml
llm:
  provider: "ollama" # Options: ollama, llama_cpp_server, llama_cpp_cli, google_gemini
  # Provider-specific parameters follow...
```

**Supported Providers:**

- `ollama` - Local Ollama server
- `llama_cpp_server` - llama.cpp HTTP server
- `llama_cpp_cli` - llama.cpp command-line interface
- `llama_cpp_langchain` - llama.cpp via Langchain
- `google_gemini` - Google Gemini API

For detailed provider configuration, see [Advanced LLM Configuration](advanced-llm-configuration.md).

### Messaging Configuration

The `messaging` section configures the communication system between components:

```yaml
messaging:
  protocol: "mqtt" # Messaging protocol (currently MQTT only)
  broker_address: "localhost" # MQTT broker hostname/IP
  port: 1883 # MQTT broker port
  client_id: "argentic-agent" # Unique client identifier
  username: null # Optional authentication
  password: null # Optional authentication
  keepalive: 60 # Connection keepalive interval

  # Optional TLS configuration
  tls_params:
    ca_certs: "/path/to/ca.crt"
    certfile: "/path/to/client.crt"
    keyfile: "/path/to/client.key"
```

For detailed messaging configuration, see [Messaging Configuration](messaging-configuration.md).

### Tool System Configuration

The `tools` section configures the external tool integration system:

```yaml
tools:
  registration:
    topic: "agent/tools/register" # Topic for tool registration
    timeout: 30 # Registration timeout (seconds)

  execution:
    topic_base: "agent/tools/call" # Base topic for tool execution
    response_base: "agent/tools/response" # Base topic for tool responses
    timeout: 300 # Execution timeout (seconds)

  status:
    topic: "agent/status/info" # Topic for status updates
```

### Logging Configuration

The `logging` section controls logging behavior:

```yaml
logging:
  level: "INFO" # Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
  format: "detailed" # Log format: simple, detailed, json
  file: null # Optional log file path
  pub_log_topic: null # Optional MQTT topic for log publishing
```

**Log Levels:**

- `DEBUG`: Detailed debugging information
- `INFO`: General information messages
- `WARNING`: Warning messages for potential issues
- `ERROR`: Error messages for handled exceptions
- `CRITICAL`: Critical errors that may cause application failure

You can also set the log level via:

- Command line: `--log-level DEBUG`
- Environment variable: `LOG_LEVEL=DEBUG`

## Environment Variables

Sensitive configuration like API keys should be stored in environment variables rather than the config file. Argentic supports loading environment variables from a `.env` file.

Create a `.env` file in your project root:

```bash
# Google Gemini API Key (if using google_gemini provider)
GOOGLE_GEMINI_API_KEY=your_api_key_here

# Custom config path (optional)
CONFIG_PATH=/path/to/custom/config.yaml

# Log level override (optional)
LOG_LEVEL=DEBUG
```

For detailed environment variable configuration, see [Environment Variables](environment-variables.md).

## Configuration Validation

Argentic validates the configuration on startup and will:

- Report missing required settings
- Warn about deprecated options
- Provide default values for optional settings
- Validate parameter ranges and types

Check the startup logs for any configuration warnings or errors.

## Example Configurations

### Minimal Configuration

```yaml
llm:
  provider: "ollama"
  ollama_model_name: "llama3.2:3b"

messaging:
  protocol: "mqtt"
  broker_address: "localhost"
```

### Production Configuration

```yaml
llm:
  provider: "google_gemini"
  google_gemini_model_name: "gemini-2.0-flash"
  google_gemini_parameters:
    temperature: 0.7
    max_output_tokens: 2048

messaging:
  protocol: "mqtt"
  broker_address: "mqtt.production.com"
  port: 8883
  client_id: "argentic-prod-agent"
  username: "argentic"
  tls_params:
    ca_certs: "/etc/ssl/certs/ca.crt"
    certfile: "/etc/ssl/certs/client.crt"
    keyfile: "/etc/ssl/private/client.key"

logging:
  level: "INFO"
  format: "json"
  file: "/var/log/argentic/agent.log"

tools:
  execution:
    timeout: 600 # 10 minutes for long-running tools
```

### Development Configuration

```yaml
llm:
  provider: "ollama"
  ollama_model_name: "llama3.2:3b"
  ollama_parameters:
    temperature: 0.8
    num_predict: 256

messaging:
  protocol: "mqtt"
  broker_address: "localhost"
  port: 1883
  client_id: "argentic-dev"

logging:
  level: "DEBUG"
  format: "detailed"
  pub_log_topic: "logs/debug"

tools:
  registration:
    timeout: 10 # Faster timeout for development
```

## Best Practices

1. **Use Environment Variables**: Store sensitive information like API keys in environment variables or `.env` files.

2. **Version Control**: Include `config.yaml.example` in version control, but exclude actual `config.yaml` and `.env` files.

3. **Documentation**: Comment your configuration files to explain custom settings.

4. **Validation**: Always check startup logs for configuration warnings.

5. **Backup**: Keep backups of working configurations, especially for production.

6. **Gradual Changes**: When modifying configurations, change one section at a time and test.

## Troubleshooting

### Common Issues

1. **Configuration Not Found**: Ensure `config.yaml` exists in the current directory or specify the path correctly.

2. **Invalid YAML**: Use a YAML validator to check syntax. Common issues include incorrect indentation and missing quotes.

3. **Missing Required Settings**: Check startup logs for required configuration that wasn't provided.

4. **Connection Issues**: Verify broker addresses, ports, and network connectivity for messaging configuration.

5. **Permission Issues**: Ensure the application has read access to configuration files and write access to log directories.

### Configuration Debugging

Enable debug logging to see detailed configuration loading:

```bash
./start.sh agent --log-level DEBUG
```

This will show:

- Configuration file loading process
- Environment variable resolution
- Default value application
- Validation results
