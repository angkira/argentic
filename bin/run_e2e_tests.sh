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
    
    # Check RabbitMQ status and readiness for configuration
    echo "Checking RabbitMQ management API readiness for configuration..."
    # Loop until RabbitMQ management API is responsive or timeout
    RETRY_COUNT=0
    MAX_RETRIES=12 # Wait up to 60 seconds (12 * 5s)
    SUCCESS=false
    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        if curl -f -u guest:guest http://localhost:15672/api/aliveness-test/%2F > /dev/null 2>&1; then
            echo "RabbitMQ management API is responsive on default vhost."
            SUCCESS=true
            break
        else
            echo "RabbitMQ management API not yet responsive. Retrying in 5 seconds... ($((RETRY_COUNT+1))/$MAX_RETRIES)"
            sleep 5
            RETRY_COUNT=$((RETRY_COUNT+1))
        fi
    done

    if [ "$SUCCESS" = false ]; then
        echo "Error: RabbitMQ management API did not become responsive. Exiting."
        exit 1
    fi

    # If responsive, then proceed to configure
    echo "RabbitMQ management API is up, proceeding with vhost/permissions configuration."
    configure_rabbitmq_for_tests
    
    # Show status for debugging
    echo "Docker services status after initial wait_for_services:"
    docker compose -f "$DOCKER_COMPOSE_PATH" ps
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
    
    echo "Attempting to create test vhost if it doesn't exist..."
    # Check if vhost 'test' exists
    if curl -f -u guest:guest -X GET http://localhost:15672/api/vhosts/test > /dev/null 2>&1; then
        echo "Vhost 'test' already exists."
    else
        echo "Vhost 'test' does not exist or not accessible, attempting to create..."
        curl -f -u guest:guest -X PUT http://localhost:15672/api/vhosts/test
        CREATE_VHOST_EC=$?
        if [ $CREATE_VHOST_EC -ne 0 ]; then
            echo "Error: Failed to create vhost 'test'. Exit code: $CREATE_VHOST_EC. Output above."
            # Attempt to get more info on failure
            curl -v -u guest:guest -X GET http://localhost:15672/api/vhosts
            exit 1
        else
            echo "Successfully sent command to create vhost 'test'."
        fi
    fi
    
    echo "Setting permissions for guest user on test vhost..."
    curl -f -u guest:guest -X PUT \
        -H "Content-Type: application/json" \
        -d '{"configure":".*","write":".*","read":".*"}' \
        http://localhost:15672/api/permissions/test/guest
    SET_PERMISSIONS_EC=$?
    if [ $SET_PERMISSIONS_EC -ne 0 ]; then
        echo "Error: Failed to set permissions for user 'guest' on vhost 'test'. Exit code: $SET_PERMISSIONS_EC. Output above."
        # Attempt to get more info on failure
        echo "Current permissions for vhost test:"
        curl -v -u guest:guest http://localhost:15672/api/vhosts/test/permissions
        echo "Current permissions for user guest:"
        curl -v -u guest:guest http://localhost:15672/api/users/guest/permissions
        exit 1
    else
        echo "Successfully set permissions for user 'guest' on vhost 'test'."
    fi
    
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
uv sync --all-extras

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
        docker compose -f "$DOCKER_COMPOSE_PATH" down -v --remove-orphans

        echo "Starting Docker containers..."
        docker compose -f "$DOCKER_COMPOSE_PATH" up -d
        
        if [ $? -ne 0 ]; then
            echo "Failed to start Docker containers"
            exit 1
        fi
        
        wait_for_services
    fi
    
    # Show running containers
    docker compose -f "$DOCKER_COMPOSE_PATH" ps
    
    # This block is now somewhat redundant if wait_for_services fully handles configuration,
    # but keep it as a final check and explicit sleep before tests.
    echo "Final check of RabbitMQ status before tests..."
    if ! curl -f -u guest:guest http://localhost:15672/api/aliveness-test/%2F > /dev/null 2>&1; then
        echo "Error: RabbitMQ management interface became unresponsive before tests. Exiting."
        exit 1
    else
        echo "RabbitMQ management API still responsive."
        # Re-ensure configuration one last time, or rely on wait_for_services.
        # For safety, let's call it again, it has checks for existence.
        echo "Re-confirming RabbitMQ configuration..."
        configure_rabbitmq_for_tests 
        echo "Waiting 5 seconds for RabbitMQ configuration to apply fully before tests..."
        sleep 5 # Increased sleep slightly
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
    docker compose -f "$DOCKER_COMPOSE_PATH" down -v --remove-orphans
fi

exit $TEST_EXIT_CODE 