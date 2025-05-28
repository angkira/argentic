# Environment Variables

This guide covers environment variable configuration in Argentic, including the use of `.env` files for managing sensitive information and configuration overrides.

## Overview

Environment variables provide a secure way to configure Argentic without hardcoding sensitive information like API keys, passwords, and connection strings in configuration files. Argentic supports both system environment variables and `.env` files.

## Using .env Files

Argentic automatically loads environment variables from a `.env` file in the project root directory using the `python-dotenv` library.

### Creating a .env File

Create a `.env` file in your project root:

```bash
# .env
# =============================================================================
# Argentic Environment Configuration
# =============================================================================

# LLM Provider API Keys
# -----------------------------------------------------------------------------
GOOGLE_GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# MQTT Messaging Configuration
# -----------------------------------------------------------------------------
MQTT_BROKER_ADDRESS=mqtt.production.com
MQTT_PORT=8883
MQTT_USERNAME=argentic_user
MQTT_PASSWORD=secure_mqtt_password
MQTT_CLIENT_ID=argentic-prod-agent-001

# TLS/SSL Certificates
# -----------------------------------------------------------------------------
MQTT_CA_CERT=/etc/ssl/certs/mqtt-ca.crt
MQTT_CLIENT_CERT=/etc/ssl/certs/mqtt-client.crt
MQTT_CLIENT_KEY=/etc/ssl/private/mqtt-client.key

# Application Configuration
# -----------------------------------------------------------------------------
CONFIG_PATH=/etc/argentic/config.yaml
LOG_LEVEL=INFO
LOG_FILE=/var/log/argentic/agent.log

# Database Configuration (if applicable)
# -----------------------------------------------------------------------------
DATABASE_URL=postgresql://user:password@localhost:5432/argentic
REDIS_URL=redis://localhost:6379/0

# Development/Production Environment
# -----------------------------------------------------------------------------
ENVIRONMENT=production
DEBUG=false
```

### Environment Variable Types

#### Required Variables

These environment variables are required for specific providers or features:

```bash
# Google Gemini API (required when using google_gemini provider)
GOOGLE_GEMINI_API_KEY=your_api_key

# MQTT Authentication (required for authenticated MQTT brokers)
MQTT_USERNAME=your_username
MQTT_PASSWORD=your_password
```

#### Optional Variables

These variables override default configuration values:

```bash
# Configuration overrides
CONFIG_PATH=/custom/path/to/config.yaml
LOG_LEVEL=DEBUG

# Performance tuning
WORKER_THREADS=4
MAX_CONCURRENT_TASKS=10
```

## Core Environment Variables

### Application Configuration

| Variable      | Description                | Default         | Example                                |
| ------------- | -------------------------- | --------------- | -------------------------------------- |
| `CONFIG_PATH` | Path to configuration file | `./config.yaml` | `/etc/argentic/config.yaml`            |
| `LOG_LEVEL`   | Logging level              | `INFO`          | `DEBUG`, `INFO`, `WARNING`, `ERROR`    |
| `LOG_FILE`    | Log file path              | None (stdout)   | `/var/log/argentic/agent.log`          |
| `ENVIRONMENT` | Environment name           | `development`   | `production`, `staging`, `development` |

### LLM Provider Configuration

| Variable                | Description           | Required | Provider             |
| ----------------------- | --------------------- | -------- | -------------------- |
| `GOOGLE_GEMINI_API_KEY` | Google Gemini API key | Yes      | google_gemini        |
| `OPENAI_API_KEY`        | OpenAI API key        | Yes      | openai (future)      |
| `ANTHROPIC_API_KEY`     | Anthropic API key     | Yes      | anthropic (future)   |
| `HUGGINGFACE_API_KEY`   | HuggingFace API key   | Yes      | huggingface (future) |

### Messaging Configuration

| Variable              | Description             | Default          | Example               |
| --------------------- | ----------------------- | ---------------- | --------------------- |
| `MQTT_BROKER_ADDRESS` | MQTT broker hostname    | `localhost`      | `mqtt.production.com` |
| `MQTT_PORT`           | MQTT broker port        | `1883`           | `8883` (TLS)          |
| `MQTT_USERNAME`       | MQTT username           | None             | `argentic_user`       |
| `MQTT_PASSWORD`       | MQTT password           | None             | `secure_password`     |
| `MQTT_CLIENT_ID`      | MQTT client identifier  | `argentic-agent` | `argentic-prod-001`   |
| `MQTT_KEEPALIVE`      | MQTT keepalive interval | `60`             | `300`                 |

### TLS/SSL Configuration

| Variable           | Description             | Example                       |
| ------------------ | ----------------------- | ----------------------------- |
| `MQTT_CA_CERT`     | CA certificate path     | `/etc/ssl/certs/ca.crt`       |
| `MQTT_CLIENT_CERT` | Client certificate path | `/etc/ssl/certs/client.crt`   |
| `MQTT_CLIENT_KEY`  | Client private key path | `/etc/ssl/private/client.key` |
| `MQTT_CERT_REQS`   | Certificate requirement | `CERT_REQUIRED`               |
| `MQTT_TLS_VERSION` | TLS version             | `PROTOCOL_TLSv1_2`            |

## Configuration File Integration

Reference environment variables in `config.yaml` using the `${VARIABLE_NAME}` syntax:

```yaml
llm:
  provider: "google_gemini"
  google_gemini_api_key: "${GOOGLE_GEMINI_API_KEY}"
  google_gemini_model_name: "gemini-2.0-flash"

messaging:
  protocol: "mqtt"
  broker_address: "${MQTT_BROKER_ADDRESS}"
  port: "${MQTT_PORT}"
  username: "${MQTT_USERNAME}"
  password: "${MQTT_PASSWORD}"
  client_id: "${MQTT_CLIENT_ID}"

  tls_params:
    ca_certs: "${MQTT_CA_CERT}"
    certfile: "${MQTT_CLIENT_CERT}"
    keyfile: "${MQTT_CLIENT_KEY}"

logging:
  level: "${LOG_LEVEL}"
  file: "${LOG_FILE}"
```

## Environment-Specific Configuration

### Development Environment

```bash
# .env.development
ENVIRONMENT=development
LOG_LEVEL=DEBUG
DEBUG=true

# Local services
MQTT_BROKER_ADDRESS=localhost
MQTT_PORT=1883

# Development API keys (use test keys)
GOOGLE_GEMINI_API_KEY=dev_api_key_here
```

### Production Environment

```bash
# .env.production
ENVIRONMENT=production
LOG_LEVEL=INFO
DEBUG=false

# Production services
MQTT_BROKER_ADDRESS=mqtt.production.com
MQTT_PORT=8883
MQTT_USERNAME=argentic_prod
MQTT_PASSWORD=secure_production_password

# Production API keys
GOOGLE_GEMINI_API_KEY=prod_api_key_here

# TLS configuration
MQTT_CA_CERT=/etc/ssl/certs/prod-ca.crt
MQTT_CLIENT_CERT=/etc/ssl/certs/prod-client.crt
MQTT_CLIENT_KEY=/etc/ssl/private/prod-client.key
```

### Staging Environment

```bash
# .env.staging
ENVIRONMENT=staging
LOG_LEVEL=INFO
DEBUG=false

# Staging services
MQTT_BROKER_ADDRESS=mqtt.staging.com
MQTT_PORT=8883

# Staging API keys
GOOGLE_GEMINI_API_KEY=staging_api_key_here
```

## Security Best Practices

### 1. Never Commit Secrets

Add sensitive files to `.gitignore`:

```gitignore
# Environment files
.env
.env.local
.env.production
.env.staging

# API keys and certificates
*.key
*.pem
api_keys.txt
secrets/
```

### 2. Use Example Files

Provide example environment files for reference:

```bash
# .env.example
# =============================================================================
# Argentic Environment Configuration Example
# =============================================================================

# LLM Provider API Keys
GOOGLE_GEMINI_API_KEY=your_gemini_api_key_here

# MQTT Configuration
MQTT_BROKER_ADDRESS=localhost
MQTT_PORT=1883
MQTT_USERNAME=your_username
MQTT_PASSWORD=your_password

# Application Configuration
LOG_LEVEL=INFO
CONFIG_PATH=./config.yaml
```

### 3. Environment Variable Validation

Validate required environment variables:

```python
import os
from typing import List

def validate_environment() -> List[str]:
    """Validate required environment variables."""
    required_vars = [
        'GOOGLE_GEMINI_API_KEY',
        'MQTT_BROKER_ADDRESS',
    ]

    missing = []
    for var in required_vars:
        if not os.getenv(var):
            missing.append(var)

    return missing
```

### 4. Use Strong Passwords

Generate secure passwords for production:

```bash
# Generate secure MQTT password
openssl rand -base64 32

# Generate secure client ID
openssl rand -hex 16
```

## Container Deployment

### Docker Environment

```dockerfile
# Dockerfile
FROM python:3.12-slim

# Copy application
COPY . /app
WORKDIR /app

# Install dependencies
RUN pip install -e .

# Environment configuration
ENV CONFIG_PATH=/app/config/production.yaml
ENV LOG_LEVEL=INFO

# Run application
CMD ["python", "-m", "argentic.main"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: "3.8"

services:
  argentic-agent:
    build: .
    environment:
      - GOOGLE_GEMINI_API_KEY=${GOOGLE_GEMINI_API_KEY}
      - MQTT_BROKER_ADDRESS=mosquitto
      - MQTT_PORT=1883
      - LOG_LEVEL=${LOG_LEVEL:-INFO}
    env_file:
      - .env.production
    depends_on:
      - mosquitto

  mosquitto:
    image: eclipse-mosquitto:2.0
    ports:
      - "1883:1883"
    volumes:
      - ./mosquitto.conf:/mosquitto/config/mosquitto.conf
```

### Kubernetes Secrets

```yaml
# k8s-secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: argentic-secrets
type: Opaque
stringData:
  google-gemini-api-key: "your_api_key_here"
  mqtt-username: "argentic_user"
  mqtt-password: "secure_password"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: argentic-agent
spec:
  template:
    spec:
      containers:
        - name: argentic
          image: argentic:latest
          env:
            - name: GOOGLE_GEMINI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: argentic-secrets
                  key: google-gemini-api-key
            - name: MQTT_USERNAME
              valueFrom:
                secretKeyRef:
                  name: argentic-secrets
                  key: mqtt-username
```

## Environment Variable Loading Order

Argentic loads environment variables in the following order (later sources override earlier ones):

1. System environment variables
2. `.env` file in current directory
3. Command-line arguments
4. Configuration file defaults

## Debugging Environment Issues

### 1. Check Variable Loading

```bash
# Print all environment variables
env | grep ARGENTIC

# Check specific variable
echo $GOOGLE_GEMINI_API_KEY
```

### 2. Validate .env File

```bash
# Check .env file syntax
cat .env | grep -E '^[A-Z_]+=.*$'

# Source .env manually for testing
source .env
echo $GOOGLE_GEMINI_API_KEY
```

### 3. Enable Debug Logging

```bash
# Enable debug logging to see configuration loading
LOG_LEVEL=DEBUG ./start.sh agent
```

## Common Environment Variables

### Complete Example

```bash
# .env - Complete configuration example
# =============================================================================

# Application
CONFIG_PATH=./config.yaml
LOG_LEVEL=INFO
ENVIRONMENT=production

# LLM Providers
GOOGLE_GEMINI_API_KEY=your_gemini_api_key_here

# MQTT Messaging
MQTT_BROKER_ADDRESS=mqtt.production.com
MQTT_PORT=8883
MQTT_USERNAME=argentic_prod
MQTT_PASSWORD=secure_mqtt_password
MQTT_CLIENT_ID=argentic-prod-agent-001
MQTT_KEEPALIVE=300

# TLS Configuration
MQTT_CA_CERT=/etc/ssl/certs/ca.crt
MQTT_CLIENT_CERT=/etc/ssl/certs/client.crt
MQTT_CLIENT_KEY=/etc/ssl/private/client.key

# Performance
MAX_CONCURRENT_TASKS=10
WORKER_THREADS=4
TIMEOUT_SECONDS=300

# Monitoring
HEALTH_CHECK_INTERVAL=60
METRICS_ENABLED=true
METRICS_PORT=8080
```
