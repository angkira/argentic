# Messaging Configuration

This guide covers the messaging system configuration in Argentic, which enables communication between the agent, tools, and clients.

## Overview

Argentic uses a messaging system to enable decoupled communication between components. The primary messaging protocol is MQTT, with support for other protocols planned.

## Messaging Protocols

### MQTT (Message Queuing Telemetry Transport)

MQTT is the currently supported messaging protocol in Argentic. It's lightweight, publish-subscribe based, and ideal for distributed applications.

#### Basic MQTT Configuration

```yaml
messaging:
  protocol: "mqtt"
  broker_address: "localhost" # MQTT broker hostname or IP
  port: 1883 # MQTT broker port (1883 = standard, 8883 = TLS)
  client_id: "argentic-agent" # Unique client identifier
  keepalive: 60 # Keepalive interval in seconds
```

#### Authentication

```yaml
messaging:
  protocol: "mqtt"
  broker_address: "mqtt.example.com"
  port: 1883
  client_id: "argentic-agent"
  username: "your_username" # MQTT username
  password: "your_password" # MQTT password (consider using env vars)
```

#### TLS/SSL Configuration

For secure connections, configure TLS parameters:

```yaml
messaging:
  protocol: "mqtt"
  broker_address: "secure-mqtt.example.com"
  port: 8883 # Standard TLS port
  client_id: "argentic-agent"
  username: "your_username"
  password: "your_password"

  tls_params:
    ca_certs: "/path/to/ca-certificate.crt" # CA certificate file
    certfile: "/path/to/client-certificate.crt" # Client certificate
    keyfile: "/path/to/client-private-key.key" # Client private key
    cert_reqs: "CERT_REQUIRED" # Certificate requirement level
    tls_version: "PROTOCOL_TLS" # TLS protocol version
    ciphers: "HIGH:!aNULL:!MD5" # Allowed cipher suites
```

**TLS Configuration Options:**

- **`ca_certs`**: Path to the Certificate Authority (CA) certificate file
- **`certfile`**: Path to the client certificate file (for mutual TLS)
- **`keyfile`**: Path to the client private key file (for mutual TLS)
- **`cert_reqs`**: Certificate requirement level:
  - `CERT_NONE`: No certificate verification
  - `CERT_OPTIONAL`: Certificate verification if provided
  - `CERT_REQUIRED`: Certificate verification required
- **`tls_version`**: TLS protocol version (e.g., `PROTOCOL_TLS`, `PROTOCOL_TLSv1_2`)
- **`ciphers`**: Allowed cipher suites for encryption

## Topic Structure

Argentic uses a structured topic hierarchy for organized communication:

### Agent Topics

```
agent/
├── status/
│   └── info              # Agent status updates
├── query/
│   ├── ask               # Questions to the agent
│   └── response          # Agent responses
└── tools/
    ├── register          # Tool registration requests
    ├── call/<tool_id>    # Tool execution requests
    └── response/<tool_id> # Tool execution responses
```

### Tool Topics

```
tools/
├── <tool_name>/
│   ├── register         # Tool registration
│   ├── status           # Tool status
│   ├── call/<task_id>   # Task execution
│   └── response/<task_id> # Task results
```

### Log Topics

```
logs/
├── debug               # Debug messages
├── info                # Information messages
├── warning             # Warning messages
├── error               # Error messages
└── agent/<component>   # Component-specific logs
```

## Configuration Examples

### Local Development

```yaml
messaging:
  protocol: "mqtt"
  broker_address: "localhost"
  port: 1883
  client_id: "argentic-dev"
  keepalive: 60
```

### Docker Compose Setup

```yaml
messaging:
  protocol: "mqtt"
  broker_address: "mosquitto" # Docker service name
  port: 1883
  client_id: "argentic-agent"
  keepalive: 60
```

### Cloud MQTT Service

```yaml
messaging:
  protocol: "mqtt"
  broker_address: "mqtt.aws.example.com"
  port: 8883
  client_id: "argentic-prod-agent"
  username: "argentic_user"
  password: "secure_password" # Use environment variable
  keepalive: 300

  tls_params:
    ca_certs: "/etc/ssl/certs/aws-iot-ca.crt"
    cert_reqs: "CERT_REQUIRED"
    tls_version: "PROTOCOL_TLSv1_2"
```

### High Availability Setup

```yaml
messaging:
  protocol: "mqtt"
  broker_address: "mqtt-cluster.example.com"
  port: 8883
  client_id: "argentic-ha-agent-01"
  username: "argentic_ha"
  password: "ha_password"
  keepalive: 60

  # Connection retry settings
  connect_retry_delay: 5 # Seconds between connection attempts
  max_reconnect_attempts: 10

  tls_params:
    ca_certs: "/etc/ssl/certs/cluster-ca.crt"
    cert_reqs: "CERT_REQUIRED"
    tls_version: "PROTOCOL_TLSv1_2"
```

## Advanced Configuration

### Quality of Service (QoS)

MQTT supports different QoS levels for message delivery:

```yaml
messaging:
  protocol: "mqtt"
  broker_address: "mqtt.example.com"

  # Default QoS levels for different message types
  qos_levels:
    tool_calls: 1 # At least once delivery
    responses: 1 # At least once delivery
    status: 0 # At most once delivery
    logs: 0 # At most once delivery
```

**QoS Levels:**

- **0**: At most once (fire and forget)
- **1**: At least once (acknowledged delivery)
- **2**: Exactly once (assured delivery)

### Message Retention

Configure message retention for persistent communication:

```yaml
messaging:
  protocol: "mqtt"
  broker_address: "mqtt.example.com"

  # Message retention settings
  retain_messages:
    status: true # Retain status messages
    tool_registry: true # Retain tool registration
    responses: false # Don't retain responses
```

### Clean Session

Control session persistence:

```yaml
messaging:
  protocol: "mqtt"
  broker_address: "mqtt.example.com"
  client_id: "argentic-persistent-agent"
  clean_session: false # Maintain session across disconnections
```

## Environment Variables

Store sensitive messaging configuration in environment variables:

```bash
# .env file
MQTT_BROKER_ADDRESS=mqtt.production.com
MQTT_USERNAME=argentic_user
MQTT_PASSWORD=secure_password
MQTT_CLIENT_ID=argentic-prod-001

# TLS Certificate paths
MQTT_CA_CERT=/etc/ssl/certs/ca.crt
MQTT_CLIENT_CERT=/etc/ssl/certs/client.crt
MQTT_CLIENT_KEY=/etc/ssl/private/client.key
```

Reference in configuration:

```yaml
messaging:
  protocol: "mqtt"
  broker_address: "${MQTT_BROKER_ADDRESS}"
  username: "${MQTT_USERNAME}"
  password: "${MQTT_PASSWORD}"
  client_id: "${MQTT_CLIENT_ID}"

  tls_params:
    ca_certs: "${MQTT_CA_CERT}"
    certfile: "${MQTT_CLIENT_CERT}"
    keyfile: "${MQTT_CLIENT_KEY}"
```

## MQTT Broker Setup

### Mosquitto (Local Development)

Install and run Mosquitto locally:

```bash
# Install Mosquitto
brew install mosquitto  # macOS
sudo apt-get install mosquitto mosquitto-clients  # Ubuntu

# Start Mosquitto
mosquitto -v  # Verbose mode for debugging
```

### Docker Mosquitto

```yaml
# docker-compose.yml
version: "3.8"
services:
  mosquitto:
    image: eclipse-mosquitto:2.0
    ports:
      - "1883:1883"
      - "9001:9001"
    volumes:
      - ./mosquitto.conf:/mosquitto/config/mosquitto.conf
      - ./data:/mosquitto/data
      - ./logs:/mosquitto/log
```

### Cloud MQTT Services

Popular cloud MQTT services:

- **AWS IoT Core**: Managed MQTT with device management
- **Azure IoT Hub**: Enterprise IoT messaging platform
- **Google Cloud IoT Core**: Scalable IoT device connectivity
- **HiveMQ Cloud**: Professional MQTT cloud service
- **CloudMQTT**: Simple MQTT hosting service

## Troubleshooting

### Connection Issues

1. **Broker Unreachable**:

   ```bash
   # Test broker connectivity
   mosquitto_sub -h mqtt.example.com -p 1883 -t test/topic
   ```

2. **Authentication Failures**:

   - Verify username/password
   - Check client ID uniqueness
   - Validate certificate paths and permissions

3. **TLS/SSL Issues**:
   ```bash
   # Test TLS connection
   openssl s_client -connect mqtt.example.com:8883 -CAfile ca.crt
   ```

### Message Delivery Issues

1. **Messages Not Received**:

   - Check topic subscription patterns
   - Verify QoS levels
   - Check client connection status

2. **Duplicate Messages**:
   - Review QoS settings
   - Check for multiple subscribers
   - Verify clean session settings

### Performance Issues

1. **High Latency**:

   - Reduce keepalive interval
   - Use appropriate QoS levels
   - Optimize topic structure

2. **Connection Drops**:
   - Increase keepalive interval
   - Implement reconnection logic
   - Check network stability

## Monitoring and Debugging

### Enable Debug Logging

```yaml
logging:
  level: "DEBUG"
  pub_log_topic: "logs/debug"
```

### MQTT Client Tools

Useful tools for debugging MQTT:

```bash
# Subscribe to all topics
mosquitto_sub -h localhost -t '#' -v

# Publish test message
mosquitto_pub -h localhost -t test/topic -m "Hello World"

# Monitor specific topic pattern
mosquitto_sub -h localhost -t 'agent/+/status' -v
```

### Message Inspection

Monitor Argentic message flow:

```bash
# Monitor tool registration
mosquitto_sub -h localhost -t 'agent/tools/register' -v

# Monitor agent responses
mosquitto_sub -h localhost -t 'agent/query/response' -v

# Monitor all agent activity
mosquitto_sub -h localhost -t 'agent/#' -v
```
