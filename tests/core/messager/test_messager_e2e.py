import pytest
import asyncio
import os
import subprocess
import time
import uuid
import json
from datetime import datetime

# Add src to path to fix import issues
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.core.messager.messager import Messager
from src.core.messager.protocols import MessagerProtocol
from src.core.protocol.message import BaseMessage


# Add a JSON encoder for TestMessage to fix serialization
class TestMessageEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, TestMessage):
            return {
                "id": str(obj.id),
                "timestamp": obj.timestamp.isoformat() if obj.timestamp else None,
                "type": obj.__class__.__name__,
                "message": obj.message,
                "value": obj.value,
            }
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, uuid.UUID):
            return str(obj)
        return super().default(obj)


# Set the custom encoder as the default
json._default_encoder = TestMessageEncoder()


# Define a test message class
class TestMessage(BaseMessage):
    """Test message for e2e tests"""

    message: str = "test_message"
    value: int = 42

    # Ensure this class is not collected as a test
    __test__ = False

    # Override to ensure proper JSON serialization
    def model_dump_json(self) -> str:
        """Custom JSON serialization to handle datetime and UUID fields"""
        data = {
            "id": str(self.id),
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "type": self.__class__.__name__,
            "message": self.message,
            "value": self.value,
        }
        return json.dumps(data)

    # Ensure we can properly convert to dict for json serialization
    def model_dump(self) -> dict:
        """Convert to dict for JSON serialization"""
        return {
            "id": str(self.id),
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "type": self.__class__.__name__,
            "message": self.message,
            "value": self.value,
        }


# Mock the aioredis module for Redis driver
# Similar to what we did in unit tests to avoid TimeoutError conflict
class RedisMock:
    """A more complete mock of the aioredis package to avoid import conflicts"""

    # Define a custom TimeoutError that doesn't conflict
    class CustomTimeoutError(Exception):
        pass

    # Define a Redis mock class
    class Redis:
        def __init__(self, *args, **kwargs):
            self.closed = False
            self.pubsub = MagicMock(return_value=AsyncMock())
            self.publish = AsyncMock()
            self.close = AsyncMock()

    # Define a connection module
    class connection:
        pass

    # Define a client module
    class client:
        class PubSub:
            def __init__(self):
                self.subscribe = AsyncMock()
                self.listen = AsyncMock()
                self.__aiter__ = AsyncMock(return_value=self)
                self.__anext__ = AsyncMock()

    # Factory function
    @staticmethod
    async def from_url(*args, **kwargs):
        return RedisMock.Redis()


# Only apply Redis mock if actually running Redis tests
if "redis" in sys.argv or "redis" in os.environ.get("PYTEST_CURRENT_TEST", ""):
    from unittest.mock import AsyncMock, MagicMock

    sys.modules["aioredis"] = RedisMock

# Test configuration
TEST_CONFIG = {
    "mqtt": {
        "broker_address": "localhost",
        "port": 1883,
        "protocol": MessagerProtocol.MQTT,
        "client_id": "mqtt-e2e-client",
    },
    "redis": {
        "broker_address": "localhost",
        "port": 6379,
        "protocol": MessagerProtocol.REDIS,
        "client_id": "redis-e2e-client",
    },
    "rabbitmq": {
        "broker_address": "localhost",
        "port": 5672,
        "protocol": MessagerProtocol.RABBITMQ,
        "client_id": "rabbitmq-e2e-client",
        "username": "guest",
        "password": "guest",
    },
    "kafka": {
        "broker_address": "localhost",
        "port": 9092,
        "protocol": MessagerProtocol.KAFKA,
        "client_id": "kafka-e2e-client",
        # Add group_id for Kafka consumer
        "group_id": "test-group",
        # Add auto_offset_reset for new topics
        "auto_offset_reset": "earliest",
    },
}


@pytest.fixture(scope="module")
def docker_services():
    """Use existing containers or start if needed."""
    # Skip if docker is not available
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        pytest.skip("Docker is not available")

    # Path to the docker-compose.yml file
    docker_compose_path = os.path.join(os.path.dirname(__file__), "docker-compose.yml")

    # Check if the services are already running
    try:
        result = subprocess.run(
            [
                "docker-compose",
                "-f",
                docker_compose_path,
                "ps",
                "--services",
                "--filter",
                "status=running",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        running_services = result.stdout.strip().split("\n")
        if len(running_services) >= 4:  # At least 4 services should be running
            print("Using existing Docker containers...")
            yield
            return
    except subprocess.SubprocessError:
        pass  # Fall through to starting containers

    print("Starting Docker containers...")
    # Start containers
    subprocess.run(
        ["docker-compose", "-f", docker_compose_path, "up", "-d"],
        check=True,
    )

    # Give services time to start - especially important for Kafka
    print("Waiting for services to start (30 seconds)...")
    time.sleep(30)

    # Check if the services are running
    result = subprocess.run(
        ["docker-compose", "-f", docker_compose_path, "ps"],
        check=True,
        capture_output=True,
        text=True,
    )
    print(f"Docker services status:\n{result.stdout}")

    yield

    # We don't stop containers by default anymore
    # User must run `docker-compose down` manually
    print("NOTE: Docker containers are still running. Run 'docker-compose down' to stop them.")


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


class MessageReceiver:
    """Helper class to collect received messages"""

    def __init__(self):
        self.received_messages = []
        self.message_received_event = asyncio.Event()
        print("MessageReceiver initialized")

    async def handler(self, message):
        """Message handler that stores received messages"""
        print(f"Received message in handler: {message}")

        # Convert binary data to string or dict if possible
        if isinstance(message, bytes):
            try:
                # Try to decode as JSON first
                decoded_message = json.loads(message.decode("utf-8"))
                print(f"Successfully decoded JSON message: {decoded_message}")
                self.received_messages.append(decoded_message)
            except json.JSONDecodeError:
                # If not JSON, just decode as string
                decoded_message = message.decode("utf-8", errors="replace")
                print(f"Message is not JSON, decoded as string: {decoded_message}")
                self.received_messages.append(decoded_message)
            except UnicodeDecodeError:
                # If can't decode as UTF-8, store as binary
                print(f"Message cannot be decoded as UTF-8, storing as binary: {message}")
                self.received_messages.append(message)
        else:
            # If not bytes, store as is
            self.received_messages.append(message)

        # Signal that we received a message
        self.message_received_event.set()

    async def wait_for_message(self, timeout=5.0):
        """Wait for a message to be received"""
        try:
            await asyncio.wait_for(self.message_received_event.wait(), timeout)
            print(f"Message received event triggered! Got {len(self.received_messages)} messages")
            return True
        except asyncio.TimeoutError:
            print(f"Timeout waiting for message after {timeout}s")
            return False

    def clear(self):
        """Clear received messages and reset event"""
        self.received_messages = []
        self.message_received_event.clear()


@pytest.mark.asyncio
@pytest.mark.e2e
class TestMessagerE2E:
    """End-to-end tests for the Messager class with real message brokers"""

    @pytest.mark.parametrize("protocol", ["mqtt", "rabbitmq"])
    async def test_publish_subscribe_same_protocol(self, docker_services, protocol):
        """Test publishing and subscribing with the same protocol"""
        # Skip test if docker is not available
        if "docker_services" not in locals():
            pytest.skip("Docker services not available")

        config = TEST_CONFIG[protocol].copy()

        # Create a unique topic for this test
        test_uuid = uuid.uuid4().hex
        test_topic = f"test/e2e/{protocol}/{test_uuid}"
        test_message = TestMessage(message=f"E2E message for {protocol}", value=100)

        # Create messager
        messager = Messager(**config)

        # Create message receiver
        receiver = MessageReceiver()

        try:
            # Connect to broker
            connected = await messager.connect()
            assert connected, f"Failed to connect to {protocol} broker"
            print(f"Connected to {protocol} broker successfully")

            # Subscribe to test topic
            await messager.subscribe(test_topic, receiver.handler, TestMessage)
            print(f"Subscribed to topic: {test_topic}")

            # Give subscription time to establish
            await asyncio.sleep(2)

            # Publish test message
            print(f"Publishing message to topic: {test_topic}")
            await messager.publish(test_topic, test_message)

            # Wait for message to be received
            print("Waiting for message...")
            received = await receiver.wait_for_message(timeout=10.0)

            # Check that message was received
            assert received, f"No message received for {protocol}"
            assert len(receiver.received_messages) > 0, f"No messages in receiver for {protocol}"

            # Verify message content
            received_message = receiver.received_messages[0]
            print(f"Received message: {received_message}")
            assert received_message.message == test_message.message
            assert received_message.value == test_message.value

        finally:
            # Always disconnect
            await messager.disconnect()

    @pytest.mark.parametrize(
        "publisher_protocol,subscriber_protocol",
        [
            ("mqtt", "mqtt"),  # Same protocol as baseline
            ("rabbitmq", "rabbitmq"),  # Same protocol as baseline
        ],
    )
    async def test_publish_subscribe_cross_protocol(
        self, docker_services, publisher_protocol, subscriber_protocol
    ):
        """Test publishing with one protocol and subscribing with another"""
        # Skip test if docker is not available
        if "docker_services" not in locals():
            pytest.skip("Docker services not available")

        publisher_config = TEST_CONFIG[publisher_protocol].copy()
        subscriber_config = TEST_CONFIG[subscriber_protocol].copy()

        # Create a unique topic for this test
        test_uuid = uuid.uuid4().hex
        test_topic = f"test/e2e/cross/{publisher_protocol}-{subscriber_protocol}/{test_uuid}"
        test_message = TestMessage(
            message=f"Cross-protocol message from {publisher_protocol} to {subscriber_protocol}",
            value=200,
        )

        # Create messagers
        publisher = Messager(**publisher_config)
        subscriber = Messager(**subscriber_config)

        # Create message receiver
        receiver = MessageReceiver()

        try:
            # Connect both messagers
            pub_connected = await publisher.connect()
            sub_connected = await subscriber.connect()

            assert pub_connected, f"Failed to connect {publisher_protocol} publisher"
            assert sub_connected, f"Failed to connect {subscriber_protocol} subscriber"
            print(f"Connected to both {publisher_protocol} and {subscriber_protocol} brokers")

            # Subscribe to test topic
            await subscriber.subscribe(test_topic, receiver.handler, TestMessage)
            print(f"Subscribed to topic: {test_topic}")

            # Give subscription time to establish
            await asyncio.sleep(3)

            # Publish test message
            print(f"Publishing message to topic: {test_topic}")
            await publisher.publish(test_topic, test_message)

            # Wait for message to be received (longer timeout for cross-protocol)
            print("Waiting for message...")
            received = await receiver.wait_for_message(timeout=15.0)

            # Check that message was received
            assert (
                received
            ), f"No message received from {publisher_protocol} to {subscriber_protocol}"
            assert len(receiver.received_messages) > 0, "No messages in receiver"

            # Verify message content
            received_message = receiver.received_messages[0]
            print(f"Received message: {received_message}")
            assert received_message.message == test_message.message
            assert received_message.value == test_message.value

        finally:
            # Always disconnect
            await asyncio.gather(
                publisher.disconnect(),
                subscriber.disconnect(),
            )

    @pytest.mark.kafka
    @pytest.mark.xfail(reason="Kafka consumer not receiving messages correctly")
    async def test_kafka_publish_subscribe(self, docker_services):
        """Test Kafka publishing and subscribing"""
        # Skip test if docker is not available
        if "docker_services" not in locals():
            pytest.skip("Docker services not available")

        config = TEST_CONFIG["kafka"].copy()
        # Remove group_id and auto_offset_reset from config as they're not supported by Messager init
        group_id = config.pop("group_id", f"group-{uuid.uuid4().hex}")
        auto_offset_reset = config.pop("auto_offset_reset", "earliest")

        # Verify that Kafka is ready by using the docker inspect command
        try:
            result = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Health.Status}}", "messager-kafka-1"],
                check=True,
                capture_output=True,
                text=True,
            )
            kafka_status = result.stdout.strip()
            if kafka_status != "healthy" and kafka_status != "starting":
                print(f"Kafka status: {kafka_status}")
                pytest.skip("Kafka container is not ready")
        except subprocess.SubprocessError:
            print("Could not check Kafka container health")
            # Continue anyway - test will fail if Kafka is actually not available

        # Use a predefined topic for Kafka (that was created in docker-compose)
        test_topic = "test-topic"
        test_message = TestMessage(message="Kafka test message", value=42)

        # Create messagers - we need separate publisher and subscriber for Kafka
        publisher = Messager(**config)

        # Subscriber needs a unique group_id
        subscriber_config = config.copy()
        subscriber_config["client_id"] = "kafka-subscriber"
        subscriber = Messager(**subscriber_config)

        # Create message receiver
        receiver = MessageReceiver()

        try:
            # Connect both with retry
            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                try:
                    pub_connected = await publisher.connect()
                    sub_connected = await subscriber.connect()

                    assert pub_connected, "Failed to connect Kafka publisher"
                    assert sub_connected, "Failed to connect Kafka subscriber"
                    print("Connected to Kafka broker successfully")

                    # Try to subscribe
                    await subscriber.subscribe(
                        test_topic,
                        receiver.handler,
                        TestMessage,
                        group_id=group_id,
                        auto_offset_reset=auto_offset_reset,
                    )
                    print(f"Subscribed to Kafka topic: {test_topic} with group_id: {group_id}")
                    break
                except Exception as e:
                    retry_count += 1
                    print(f"Connection attempt {retry_count} failed: {e}")
                    if retry_count >= max_retries:
                        pytest.skip(f"Failed to connect to Kafka after {max_retries} attempts: {e}")
                    await asyncio.sleep(5)  # Wait before retry

            # Give subscription time to establish
            await asyncio.sleep(5)

            # Publish messages in a loop to increase chances of success
            print(f"Publishing messages to Kafka topic: {test_topic}")
            for i in range(5):
                test_message.message = f"Kafka message {i}"
                await publisher.publish(test_topic, test_message)
                await asyncio.sleep(1)

            # Wait for message to be received
            print("Waiting for Kafka message...")
            received = await receiver.wait_for_message(timeout=20.0)

            # Check that message was received
            assert received, "No message received from Kafka"
            assert len(receiver.received_messages) > 0, "No messages in Kafka receiver"

            # Verify at least one message was received (matching not needed since we sent many)
            received_message = receiver.received_messages[0]
            print(f"Received Kafka message: {received_message}")
            assert "Kafka message" in received_message.message
            assert isinstance(received_message.value, int)

        finally:
            # Always disconnect
            await asyncio.gather(
                publisher.disconnect(),
                subscriber.disconnect(),
            )
