# End-to-End Testing for Messaging System

This directory contains end-to-end tests for the messaging system that use actual message brokers running in Docker containers.

## Requirements

- Docker and docker compose installed on your system
- Python dependencies installed (same as main project)

## Running the Tests

To run the end-to-end tests:

```bash
# Navigate to the project root
cd /path/to/argentic

# Run the e2e tests with the e2e marker
pytest tests/core/messager/test_messager_e2e.py -v -m e2e
```

Note: The first run may take some time as Docker images are downloaded.

## Test Details

The e2e tests verify:

1. Publishing and subscribing with the same protocol (MQTT, Redis, RabbitMQ, Kafka)
2. Cross-protocol messaging (publishing with one protocol and subscribing with another)

### Docker Containers

The tests use the following containers:

- Mosquitto MQTT broker (port 1883)
- Redis (port 6379)
- RabbitMQ (port 5672)
- Kafka (port 9092) with Zookeeper

## Troubleshooting

If tests fail:

1. Check that Docker is running
2. Ensure no other services are using the required ports
3. Check container logs:
   ```bash
   docker compose -f tests/core/messager/docker-compose.yml logs
   ```
4. Increase timeouts in the test code if necessary

### Skipping Specific Tests

To run only tests for specific protocols:

```bash
# Run only the MQTT tests
pytest tests/core/messager/test_messager_e2e.py -v -k "mqtt"
```
