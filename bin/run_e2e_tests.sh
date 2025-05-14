#!/bin/bash

# Define Docker Compose path
DOCKER_COMPOSE_PATH="tests/core/messager/e2e/docker-compose.yml"

# Required services for testing
REQUIRED_SERVICES=("mosquitto" "redis" "rabbitmq" "kafka" "zookeeper")

# Wait time in seconds
WAIT_TIME=20

# Wait for services to be ready
function wait_for_services {
    echo "Waiting for services to start ($WAIT_TIME seconds)..."
    
    # Initial delay to let containers start
    sleep 5
    
    # Check if mosquitto is ready using connection test
    echo "Checking MQTT broker health..."
    # Try to use mosquitto_sub if available
    if command -v mosquitto_sub &> /dev/null; then
        if timeout 3 mosquitto_sub -h localhost -p 1884 -t test/health -C 1 -W 1 &> /dev/null; then
            echo "MQTT broker is accepting connections"
        else
            echo "Warning: MQTT broker is not responding to connection tests. MQTT tests may fail."
            echo "Waiting longer for MQTT to initialize..."
            sleep 5
        fi
    else
        echo "mosquitto_sub not available, can't verify MQTT connectivity"
        # If can't test directly, wait longer
        sleep 5
    fi
    
    # Check RabbitMQ status
    echo "Checking RabbitMQ status..."
    if ! curl -s http://localhost:15672/api/vhosts -u guest:guest > /dev/null; then
        echo "Warning: RabbitMQ management interface is not responding. Some tests may fail."
        echo "Waiting longer for RabbitMQ to initialize..."
        sleep 5
    else
        echo "RabbitMQ is ready for connections."
        # Ensure RabbitMQ is properly configured for tests
        configure_rabbitmq_for_tests
    fi
    
    # Show status for debugging
    echo "Docker services status:"
    docker-compose -f "$DOCKER_COMPOSE_PATH" ps
}

function show_help {
    echo "Usage: $0 [options] [-- pytest_args]"
    echo "Options:"
    echo "  --start-docker         Start Docker containers before running tests (only if not already running)"
    echo "  --force-restart-docker Stop and restart Docker containers even if they're already running"
    echo "  --stop-docker          Stop Docker containers after running tests"
    echo "  --docker-only          Only manage Docker containers, don't run tests"
    echo "  --help                 Show this help message"
    echo
    echo "Examples:"
    echo "  $0 --start-docker                 # Use existing containers or start new ones, then run tests"
    echo "  $0 --force-restart-docker         # Force restart containers, then run tests"
    echo "  $0 --start-docker -- -v           # Use/start containers and run tests with verbose output"
    echo "  $0 --docker-only --start-docker   # Only ensure containers are running, don't run tests"
}

# Check if required services are already running
function check_running_services {
    echo "Checking for running services..."
    
    # First check if containers exist with names matching our service names
    for service in "${REQUIRED_SERVICES[@]}"; do
        container_name="e2e-${service}-1"
        if ! docker ps -q --filter "name=$container_name" | grep -q .; then
            echo "Service container $container_name is not running"
            return 1
        else
            echo "Service container $container_name is running"
        fi
    done
    
    echo "All required services are already running"
    return 0
}

# Function to configure RabbitMQ for tests
function configure_rabbitmq_for_tests {
    echo "Configuring RabbitMQ for tests..."
    
    # Create test vhost if it doesn't exist
    if ! curl -s -u guest:guest -X GET http://localhost:15672/api/vhosts/test | grep -q "name"; then
        echo "Creating test vhost..."
        curl -s -u guest:guest -X PUT http://localhost:15672/api/vhosts/test
    fi
    
    # Ensure guest user has permissions on test vhost
    echo "Setting permissions for guest user on test vhost..."
    curl -s -u guest:guest -X PUT \
        -H "Content-Type: application/json" \
        -d '{"configure":".*","write":".*","read":".*"}' \
        http://localhost:15672/api/permissions/test/guest
    
    echo "RabbitMQ configuration complete."
}

# Parse command line arguments
START_DOCKER=false
FORCE_RESTART=false
STOP_DOCKER=false
DOCKER_ONLY=false
PYTEST_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --start-docker)
            START_DOCKER=true
            shift
            ;;
        --force-restart-docker)
            START_DOCKER=true
            FORCE_RESTART=true
            shift
            ;;
        --stop-docker)
            STOP_DOCKER=true
            shift
            ;;
        --docker-only)
            DOCKER_ONLY=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        --)
            shift
            PYTEST_ARGS=("$@")
            break
            ;;
        *)
            PYTEST_ARGS+=("$1")
            shift
            ;;
    esac
done

# Ensure the script stops if any command fails
set -e

# Install dependencies
# python -m pip install -e ".[dev,kafka,redis,rabbitmq]"
uv sync --extra dev --extra kafka --extra redis --extra rabbitmq

# Set Python path to include the src directory
PROJECT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT

# Start Docker containers if requested
if [ "$START_DOCKER" = true ]; then
    # Check if services are already running
    if [ "$FORCE_RESTART" = false ] && check_running_services; then
        echo "Using existing Docker containers..."
    else
        # If forcing restart or containers not running, start new ones
        echo "Stopping any existing containers..."
        docker-compose -f "$DOCKER_COMPOSE_PATH" down -v --remove-orphans

        echo "Starting Docker containers..."
        docker-compose -f "$DOCKER_COMPOSE_PATH" up -d
        
        if [ $? -ne 0 ]; then
            echo "Failed to start Docker containers"
            exit 1
        fi
        
        wait_for_services
    fi
    
    # Show running containers
    docker-compose -f "$DOCKER_COMPOSE_PATH" ps
    
    # Check RabbitMQ status specifically
    echo "Checking RabbitMQ status..."
    if ! curl -s http://localhost:15672/api/vhosts -u guest:guest > /dev/null; then
        echo "Warning: RabbitMQ management interface is not responding. Some tests may fail."
    else
        echo "RabbitMQ is ready for connections."
        # Ensure RabbitMQ is properly configured for tests
        configure_rabbitmq_for_tests
    fi
fi

# Run tests if not in docker-only mode
if [ "$DOCKER_ONLY" = false ]; then
    echo "Running E2E tests..."
    
    # Fix JSON serialization issues
    echo "Applying serialization fixes..."
    python bin/fix_serialization.py
    
    # Run the tests
    python -m pytest tests/core/messager/e2e -m "e2e" "${PYTEST_ARGS[@]}"
    TEST_EXIT_CODE=$?
else
    echo "Skipping tests as requested by --docker-only"
    TEST_EXIT_CODE=0
fi

# Stop Docker containers if requested
if [ "$STOP_DOCKER" = true ]; then
    echo "Stopping Docker containers..."
    docker-compose -f "$DOCKER_COMPOSE_PATH" down -v --remove-orphans
fi

exit $TEST_EXIT_CODE 