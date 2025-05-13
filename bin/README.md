# Testing Scripts for Argentic Messaging System

This directory contains scripts to help with testing the Argentic messaging system.

## Running Tests

### E2E Tests with Docker Containers

The `run_e2e_tests.sh` script helps manage Docker containers for end-to-end testing.

```bash
# Basic usage - use existing containers if running, or start new ones
./bin/run_e2e_tests.sh --start-docker

# Force restart of containers even if they're running
./bin/run_e2e_tests.sh --force-restart-docker

# Only manage containers, don't run tests
./bin/run_e2e_tests.sh --docker-only --start-docker

# Start containers, run tests, then stop containers
./bin/run_e2e_tests.sh --start-docker --stop-docker

# Pass arguments to pytest
./bin/run_e2e_tests.sh --start-docker -- -v
```

#### Container Reuse

The script now robustly detects if the required containers (mosquitto, redis, rabbitmq, zookeeper, kafka) are already running and reuses them by default. The detection works with:

- Containers started by our docker-compose file
- Containers started by other means but with matching names
- Custom-built images like our mosquitto container

To force a fresh start of containers:

```bash
./bin/run_e2e_tests.sh --force-restart-docker
```

## Serialization Helper

The `fix_serialization.py` script provides a universal JSON encoder that properly handles:

- UUID objects
- Datetime objects
- Custom BaseMessage objects
- TestMessage objects

This is automatically applied by the test scripts.

## Environment Setup

All test scripts ensure:

1. The necessary Python dependencies are installed
2. The proper PYTHONPATH is set
3. Docker containers are properly configured

## RabbitMQ Configuration

When RabbitMQ containers are running, the scripts automatically:

1. Create a test vhost if it doesn't exist
2. Set proper permissions for the guest user
3. Verify the RabbitMQ management interface is accessible

## Port Configuration

To avoid conflicts with locally running services:

- MQTT (Mosquitto): Port 1884 (instead of default 1883)
- Redis: Port 6380 (instead of default 6379)
- RabbitMQ: Standard ports (5672, 15672)
- Kafka: Standard port (9092)
- Zookeeper: Standard port (2181)
