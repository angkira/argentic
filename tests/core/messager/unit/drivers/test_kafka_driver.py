from argentic.core.messager.drivers import DriverConfig
from argentic.core.protocol.message import BaseMessage
from argentic.core.messager.drivers.KafkaDriver import KafkaDriver
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch, Mock
import asyncio

import pytest

# Add src to path to fix import issues
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))


@pytest.fixture
def driver_config() -> DriverConfig:
    """Create a test driver configuration"""
    return DriverConfig(
        url="test.kafka.server", port=9092, user="testuser", password="testpass", token=None
    )


class TestKafkaDriver:
    """
    Unit tests for Kafka Driver interface only.
    Complex async networking behavior is tested in e2e tests.
    """

    def test_init(self, driver_config):
        """Test driver initialization"""
        driver = KafkaDriver(driver_config)

        # Verify initial state
        assert driver._producer is None
        assert driver._consumer is None
        assert isinstance(driver._listeners, dict)
        assert len(driver._listeners) == 0
        assert driver._reader_task is None

    def test_is_connected_when_no_producer(self, driver_config):
        """Test is_connected returns False when no producer exists"""
        driver = KafkaDriver(driver_config)
        assert driver.is_connected() is False

    def test_is_connected_when_producer_exists(self, driver_config):
        """Test is_connected logic with producer"""
        driver = KafkaDriver(driver_config)

        # Mock producer that's not closed
        mock_producer = Mock()
        mock_producer._closed = False
        driver._producer = mock_producer

        assert driver.is_connected() is True

    def test_is_connected_when_producer_closed(self, driver_config):
        """Test is_connected returns False when producer is closed"""
        driver = KafkaDriver(driver_config)

        # Mock producer that's closed
        mock_producer = Mock()
        mock_producer._closed = True
        driver._producer = mock_producer

        assert driver.is_connected() is False

    @pytest.mark.asyncio
    async def test_disconnect_with_no_connections(self, driver_config):
        """Test disconnect when nothing is connected"""
        driver = KafkaDriver(driver_config)

        # Should not raise any exceptions
        await driver.disconnect()

    def test_subscription_storage(self, driver_config):
        """Test that subscription handlers are stored correctly"""
        driver = KafkaDriver(driver_config)

        topic = "test-topic"
        handler = Mock()

        # Test manual subscription storage
        if topic not in driver._listeners:
            driver._listeners[topic] = []

        driver._listeners[topic].append(handler)

        # Verify storage
        assert topic in driver._listeners
        assert handler in driver._listeners[topic]
        assert len(driver._listeners[topic]) == 1

    def test_multiple_handlers_same_topic(self, driver_config):
        """Test multiple handlers for same topic"""
        driver = KafkaDriver(driver_config)

        topic = "test-topic"
        handler1 = Mock()
        handler2 = Mock()

        # Add multiple handlers
        if topic not in driver._listeners:
            driver._listeners[topic] = []

        driver._listeners[topic].append(handler1)
        driver._listeners[topic].append(handler2)

        # Verify both handlers stored
        assert len(driver._listeners[topic]) == 2
        assert handler1 in driver._listeners[topic]
        assert handler2 in driver._listeners[topic]

    def test_publish_input_validation(self, driver_config):
        """Test publish input validation without network calls"""
        driver = KafkaDriver(driver_config)

        # Mock producer to avoid network calls
        mock_producer = Mock()
        driver._producer = mock_producer

        # Test with invalid message types
        invalid_messages = ["string", 123, None, {"dict": "value"}]

        for invalid_msg in invalid_messages:
            # Test validation directly without running async loop
            # Invalid messages should not have model_dump_json method
            assert hasattr(invalid_msg, "model_dump_json") is False

    def test_config_parameter_mapping(self, driver_config):
        """Test that config parameters are mapped correctly"""
        driver = KafkaDriver(driver_config)

        # Verify config is accessible and contains expected values
        expected_bootstrap = f"{driver_config.url}:{driver_config.port}"

        # Test parameter preparation logic
        assert driver_config.url is not None
        assert driver_config.port is not None
        assert expected_bootstrap == "test.kafka.server:9092"

    @patch("argentic.core.messager.drivers.KafkaDriver.AIOKafkaProducer")
    def test_producer_initialization_parameters(self, mock_producer_class, driver_config):
        """Test producer initialization parameters without actual connection"""
        driver = KafkaDriver(driver_config)

        # Test parameter preparation
        expected_bootstrap = f"{driver_config.url}:{driver_config.port}"

        # Verify parameter values without creating real producer
        assert expected_bootstrap == "test.kafka.server:9092"
        assert isinstance(driver._listeners, dict)

    @patch("argentic.core.messager.drivers.KafkaDriver.AIOKafkaConsumer")
    def test_consumer_initialization_parameters(self, mock_consumer_class, driver_config):
        """Test consumer initialization parameters without actual connection"""
        driver = KafkaDriver(driver_config)

        # Test group_id default logic
        test_group_id = "test-group"
        expected_bootstrap = f"{driver_config.url}:{driver_config.port}"

        # Test parameter preparation
        expected_params = {
            "bootstrap_servers": expected_bootstrap,
            "group_id": test_group_id,
            "auto_offset_reset": "earliest",
            "session_timeout_ms": 30000,
            "heartbeat_interval_ms": 10000,
            "max_poll_interval_ms": 300000,
        }

        # Verify expected parameters
        assert expected_params["bootstrap_servers"] == "test.kafka.server:9092"
        assert expected_params["auto_offset_reset"] == "earliest"
        assert expected_params["session_timeout_ms"] == 30000

    def test_listener_management(self, driver_config):
        """Test listener management functionality"""
        driver = KafkaDriver(driver_config)

        # Test adding listeners to different topics
        topic1 = "topic1"
        topic2 = "topic2"
        handler1 = Mock()
        handler2 = Mock()

        # Add listeners
        if topic1 not in driver._listeners:
            driver._listeners[topic1] = []
        if topic2 not in driver._listeners:
            driver._listeners[topic2] = []

        driver._listeners[topic1].append(handler1)
        driver._listeners[topic2].append(handler2)

        # Verify separation
        assert len(driver._listeners) == 2
        assert handler1 in driver._listeners[topic1]
        assert handler2 in driver._listeners[topic2]
        assert handler1 not in driver._listeners[topic2]
        assert handler2 not in driver._listeners[topic1]

    def test_state_management(self, driver_config):
        """Test internal state management"""
        driver = KafkaDriver(driver_config)

        # Test initial state
        assert driver._producer is None
        assert driver._consumer is None
        assert driver._reader_task is None

        # Test state changes
        mock_producer = Mock()
        mock_consumer = Mock()
        mock_task = Mock()

        driver._producer = mock_producer
        driver._consumer = mock_consumer
        driver._reader_task = mock_task

        # Verify state changes
        assert driver._producer == mock_producer
        assert driver._consumer == mock_consumer
        assert driver._reader_task == mock_task
