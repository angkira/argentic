# Messaging Configuration

This guide covers the messaging system configuration in Argentic, which enables communication between the agent, tools, and clients.

## Overview

Argentic uses a messaging system to enable decoupled communication between components. Multiple messaging protocols are supported:

- **MQTT** - Lightweight pub/sub for distributed systems
- **Redis** - In-memory messaging with pub/sub
- **Kafka** - High-throughput event streaming
- **RabbitMQ** - Enterprise message queuing
- **ZeroMQ** - Ultra-low latency brokerless messaging

## Messaging Protocols

### MQTT (Message Queuing Telemetry Transport)

MQTT is a lightweight, publish-subscribe protocol ideal for distributed applications and IoT deployments.

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

### ZeroMQ (High-Performance Brokerless Messaging)

ZeroMQ provides ultra-low latency messaging (~50-100μs) without requiring an external broker. Ideal for high-performance local multi-agent scenarios.

#### Basic ZeroMQ Configuration (Embedded Proxy)

```yaml
messaging:
  protocol: zeromq
  broker_address: 127.0.0.1
  port: 5555              # Frontend port (XSUB - for publishers)
  backend_port: 5556      # Backend port (XPUB - for subscribers)
  start_proxy: true       # Auto-start embedded proxy
  proxy_mode: embedded    # Proxy runs in agent process
```

#### External Proxy Configuration

For distributed setups, run the proxy separately:

```yaml
messaging:
  protocol: zeromq
  broker_address: 192.168.1.100  # Remote proxy host
  port: 5555
  backend_port: 5556
  start_proxy: false      # Connect to external proxy
  proxy_mode: external
```

#### Advanced ZeroMQ Configuration

```yaml
messaging:
  protocol: zeromq
  broker_address: 127.0.0.1
  port: 5555
  backend_port: 5556
  start_proxy: true
  proxy_mode: embedded

  # Socket options
  high_water_mark: 1000   # Message queue limit (prevents memory overflow)
  linger: 1000            # Socket close wait time (ms)
  connect_timeout: 5000   # Connection timeout (ms)
  topic_encoding: utf-8   # Topic encoding (default: utf-8)
```

**ZeroMQ Configuration Options:**

- **`broker_address`**: Hostname or IP for proxy (default: 127.0.0.1)
- **`port`**: Frontend port for publishers (default: 5555)
- **`backend_port`**: Backend port for subscribers (default: 5556)
- **`start_proxy`**: Auto-start proxy if not running (default: true)
- **`proxy_mode`**: `embedded` (in-process) or `external` (separate process)
- **`high_water_mark`**: Max queued messages per socket (default: 1000)
- **`linger`**: How long to wait for pending messages on close (ms)
- **`connect_timeout`**: Connection timeout in milliseconds
- **`topic_encoding`**: Character encoding for topics (default: utf-8)

#### ZeroMQ Performance Characteristics

- **Latency**: ~50-100μs (10-20x faster than MQTT)
- **Throughput**: 1M+ messages/second
- **Memory**: Lightweight, bounded by `high_water_mark`
- **Architecture**: Brokerless (XPUB/XSUB proxy pattern)

#### ZeroMQ Limitations

- ❌ No message persistence (fire-and-forget only)
- ❌ No QoS levels (all messages are QoS 0 equivalent)
- ❌ No retention (messages not stored)
- ❌ Topics cannot contain spaces (wire format uses space delimiter)
- ❌ Best for local/LAN deployments (not internet-scale)

#### When to Use ZeroMQ

**✅ Use ZeroMQ when:**
- Ultra-low latency is critical (<1ms)
- High message throughput needed (100K+ msg/sec)
- Local/LAN deployment only
- No persistence requirements
- Development/benchmarking

**❌ Use MQTT/Kafka when:**
- Message persistence required
- QoS guarantees needed
- Distributed across internet
- Long-term message retention
- Production reliability critical

#### Running External ZeroMQ Proxy

For external proxy mode, start a standalone proxy:

```python
# zeromq_proxy.py
import zmq

def run_proxy(frontend_port=5555, backend_port=5556):
    context = zmq.Context()

    # Frontend socket (XSUB) - publishers connect here
    frontend = context.socket(zmq.XSUB)
    frontend.bind(f"tcp://*:{frontend_port}")

    # Backend socket (XPUB) - subscribers connect here
    backend = context.socket(zmq.XPUB)
    backend.bind(f"tcp://*:{backend_port}")

    # Run proxy (blocks)
    zmq.proxy(frontend, backend)

if __name__ == "__main__":
    print("Starting ZeroMQ proxy...")
    run_proxy()
```

Run with:
```bash
python zeromq_proxy.py
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

### Agent Messaging Control (New Feature)

Agents now support fine-grained messaging control for different deployment scenarios:

```python
# Production agent with minimal messaging overhead
agent = Agent(
    llm=llm,
    messager=messager,
    publish_to_supervisor=True,        # Multi-agent coordination
    publish_to_agent_topic=False,      # Disable monitoring topic
    enable_tool_result_publishing=False, # No individual tool results
)

# Development agent with full monitoring
agent = Agent(
    llm=llm,
    messager=messager,
    publish_to_supervisor=True,        # Full coordination
    publish_to_agent_topic=True,       # Enable monitoring
    enable_tool_result_publishing=True, # Detailed tool monitoring
)

# Single-agent mode (no supervisor)
agent = Agent(
    llm=llm,
    messager=messager,
    publish_to_supervisor=False,       # No supervisor coordination
    publish_to_agent_topic=True,       # Local monitoring only
)
```

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
