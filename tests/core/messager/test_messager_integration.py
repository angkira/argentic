import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import sys
import os

# Add src to path to fix import issues
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from argentic.core.messager.messager import Messager
from argentic.core.messager.protocols import MessagerProtocol
from argentic.core.protocol.message import BaseMessage


# Use model_config instead of inheritance to avoid pytest collection
class TestMessage(BaseMessage):
    """Test message for integration tests"""

    type: str = "TestMessage"  # Required field
    message: str = "test_message"
    value: int = 42

    # Ensure this class is not collected as a test
    __test__ = False


@pytest.mark.asyncio
class TestMessagerIntegration:
    """Integration tests for using multiple Messager instances with different protocols"""

    @patch("argentic.core.messager.messager.create_driver")
    async def test_message_passing_between_protocols(self, mock_create_driver):
        """Test that messages can be passed between different protocol drivers"""
        # Create a mock for each driver type
        mqtt_driver = AsyncMock()
        mqtt_driver.is_connected = MagicMock(return_value=True)

        kafka_driver = AsyncMock()
        kafka_driver.is_connected = MagicMock(return_value=True)

        # Map protocols to their drivers
        protocol_to_driver = {
            MessagerProtocol.MQTT: mqtt_driver,
            MessagerProtocol.KAFKA: kafka_driver,
        }

        # Make create_driver return the right driver for each protocol
        mock_create_driver.side_effect = lambda protocol, cfg: protocol_to_driver[protocol]

        # Create messagers for different protocols
        mqtt_messager = Messager(
            broker_address="mqtt.example.com",
            protocol=MessagerProtocol.MQTT,
            client_id="mqtt-client",
        )

        kafka_messager = Messager(
            broker_address="kafka.example.com",
            protocol=MessagerProtocol.KAFKA,
            client_id="kafka-client",
            port=9092,
        )

        # Test message to be passed between the two
        test_topic = "test/integration"
        test_message = TestMessage(type="TestMessage", message="Hello between protocols", value=100)

        # Set up the captured handler on the MQTT driver
        test_handler = AsyncMock()
        captured_handler = None

        async def capture_mqtt_handler(topic, handler):
            nonlocal captured_handler
            captured_handler = handler

        mqtt_driver.subscribe.side_effect = capture_mqtt_handler

        # Subscribe to the topic with MQTT
        await mqtt_messager.connect()
        await mqtt_messager.subscribe(test_topic, test_handler, TestMessage)

        # Publish to the topic with Kafka
        await kafka_messager.connect()
        await kafka_messager.publish(test_topic, test_message)

        # Simulate the MQTT driver receiving the message from Kafka
        # This would be the encoded message coming through the broker
        encoded_message = test_message.model_dump_json().encode("utf-8")
        await captured_handler(encoded_message)

        # Verify the message was received and parsed correctly
        test_handler.assert_called_once()
        received_message = test_handler.call_args[0][0]
        assert received_message.message == "Hello between protocols"
        assert received_message.value == 100

    @patch("argentic.core.messager.messager.create_driver")
    async def test_multiple_subscribers_different_protocols(self, mock_create_driver):
        """Test multiple subscribers with different protocols"""
        # Create mocks for three different protocol drivers
        mqtt_driver = AsyncMock()
        mqtt_driver.is_connected = MagicMock(return_value=True)

        redis_driver = AsyncMock()
        redis_driver.is_connected = MagicMock(return_value=True)

        rabbitmq_driver = AsyncMock()
        rabbitmq_driver.is_connected = MagicMock(return_value=True)

        # Map protocols to their drivers
        protocol_to_driver = {
            MessagerProtocol.MQTT: mqtt_driver,
            MessagerProtocol.REDIS: redis_driver,
            MessagerProtocol.RABBITMQ: rabbitmq_driver,
        }

        # Make create_driver return the right driver for each protocol
        mock_create_driver.side_effect = lambda protocol, cfg: protocol_to_driver[protocol]

        # Create messagers for different protocols
        mqtt_messager = Messager(broker_address="mqtt.example.com", protocol=MessagerProtocol.MQTT)

        redis_messager = Messager(
            broker_address="redis.example.com", protocol=MessagerProtocol.REDIS, port=6379
        )

        rabbitmq_messager = Messager(
            broker_address="rabbitmq.example.com", protocol=MessagerProtocol.RABBITMQ, port=5672
        )

        # Connect all messagers
        await mqtt_messager.connect()
        await redis_messager.connect()
        await rabbitmq_messager.connect()

        # Test topic and handlers
        test_topic = "test/multi-protocol"
        mqtt_handler = AsyncMock(name="mqtt_handler")
        redis_handler = AsyncMock(name="redis_handler")
        rabbitmq_handler = AsyncMock(name="rabbitmq_handler")

        # Capture the handlers
        mqtt_captured_handler = None
        redis_captured_handler = None
        rabbitmq_captured_handler = None

        async def capture_mqtt_handler(topic, handler):
            nonlocal mqtt_captured_handler
            mqtt_captured_handler = handler

        async def capture_redis_handler(topic, handler):
            nonlocal redis_captured_handler
            redis_captured_handler = handler

        async def capture_rabbitmq_handler(topic, handler):
            nonlocal rabbitmq_captured_handler
            rabbitmq_captured_handler = handler

        mqtt_driver.subscribe.side_effect = capture_mqtt_handler
        redis_driver.subscribe.side_effect = capture_redis_handler
        rabbitmq_driver.subscribe.side_effect = capture_rabbitmq_handler

        # Subscribe with all messagers
        await mqtt_messager.subscribe(test_topic, mqtt_handler, TestMessage)
        await redis_messager.subscribe(test_topic, redis_handler, TestMessage)
        await rabbitmq_messager.subscribe(test_topic, rabbitmq_handler, TestMessage)

        # Create test message
        test_message = TestMessage(type="TestMessage", message="Multi-protocol message", value=999)
        encoded_message = test_message.model_dump_json().encode("utf-8")

        # Simulate each handler receiving the message
        await mqtt_captured_handler(encoded_message)
        await redis_captured_handler(encoded_message)
        await rabbitmq_captured_handler(encoded_message)

        # Verify all handlers received the message
        mqtt_handler.assert_called_once()
        redis_handler.assert_called_once()
        rabbitmq_handler.assert_called_once()

        # Verify message content in each handler
        for handler in [mqtt_handler, redis_handler, rabbitmq_handler]:
            received = handler.call_args[0][0]
            assert received.message == "Multi-protocol message"
            assert received.value == 999
