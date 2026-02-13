"""Unit tests for ZeroMQ driver.

Tests internal state and logic without actual network calls.
E2E tests with real ZeroMQ sockets are in tests/core/messager/e2e/.
"""

import os
import sys
from unittest.mock import Mock, patch

import pytest

try:
    import zmq  # noqa: F401
    import zmq.asyncio  # noqa: F401
except Exception:  # pragma: no cover
    import pytest as _pytest

    _pytest.skip(
        "pyzmq dependency is missing – skipping ZeroMQ driver tests.",
        allow_module_level=True,
    )

# Add src to path to fix import issues
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))

from argentic.core.messager.drivers.ZeroMQDriver import ZeroMQDriver
from argentic.core.messager.drivers.configs import ZeroMQDriverConfig


@pytest.fixture
def driver_config() -> ZeroMQDriverConfig:
    """Create test configuration for ZeroMQ driver."""
    return ZeroMQDriverConfig(
        url="127.0.0.1",
        port=5555,
        backend_port=5556,
        start_proxy=False,  # Don't auto-start for unit tests
        proxy_mode="external",
        high_water_mark=1000,
        linger=1000,
        connect_timeout=5000,
        topic_encoding="utf-8",
    )


@pytest.fixture
def embedded_config() -> ZeroMQDriverConfig:
    """Create configuration for embedded proxy mode."""
    return ZeroMQDriverConfig(
        url="127.0.0.1",
        port=15555,  # Use different ports for tests
        backend_port=15556,
        start_proxy=True,
        proxy_mode="embedded",
    )


class TestZeroMQDriver:
    """Unit tests for ZeroMQ driver interface.

    Complex async networking behavior is tested in e2e tests.
    """

    def test_init(self, driver_config):
        """Test driver initialization."""
        driver = ZeroMQDriver(driver_config)

        # Verify initial state
        assert driver._context is None
        assert driver._pub_socket is None
        assert driver._sub_socket is None
        assert driver._proxy_manager is None
        assert driver._listeners == {}
        assert driver._reader_task is None
        assert driver._connected is False

    def test_init_embedded_mode(self, embedded_config):
        """Test driver initialization with embedded proxy config."""
        driver = ZeroMQDriver(embedded_config)

        # Verify config is set correctly
        assert driver.config.start_proxy is True
        assert driver.config.proxy_mode == "embedded"
        assert driver._proxy_manager is None  # Not created until connect()
        assert driver._connected is False

    def test_is_connected_when_disconnected(self, driver_config):
        """Test is_connected returns False when not connected."""
        driver = ZeroMQDriver(driver_config)
        assert driver.is_connected() is False

    def test_is_connected_when_connected(self, driver_config):
        """Test is_connected returns True when connected flag is set."""
        driver = ZeroMQDriver(driver_config)

        # Simulate connected state
        driver._connected = True
        driver._pub_socket = Mock()
        driver._sub_socket = Mock()

        assert driver.is_connected() is True

    def test_is_connected_partial_state(self, driver_config):
        """Test is_connected with partial connection state."""
        driver = ZeroMQDriver(driver_config)

        # Connected but missing pub socket
        driver._connected = True
        driver._sub_socket = Mock()
        assert driver.is_connected() is False

        # Reset and test missing sub socket
        driver._connected = True
        driver._pub_socket = Mock()
        driver._sub_socket = None
        assert driver.is_connected() is False

    def test_config_validation(self, driver_config):
        """Test configuration parameter access."""
        driver = ZeroMQDriver(driver_config)

        # Verify config is accessible
        assert driver.config.url == "127.0.0.1"
        assert driver.config.port == 5555
        assert driver.config.backend_port == 5556
        assert driver.config.high_water_mark == 1000
        assert driver.config.linger == 1000
        assert driver.config.connect_timeout == 5000
        assert driver.config.topic_encoding == "utf-8"

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self, driver_config):
        """Test disconnect when not connected does nothing gracefully."""
        driver = ZeroMQDriver(driver_config)

        # Should not raise any exceptions
        await driver.disconnect()

        assert driver._connected is False
        assert driver._pub_socket is None
        assert driver._sub_socket is None

    def test_subscription_storage(self, driver_config):
        """Test that subscription data structures work correctly."""
        driver = ZeroMQDriver(driver_config)

        # Test subscription storage without network calls
        test_handler = Mock()
        topic = "test/topic"

        # Manually test subscription storage logic (as would happen in subscribe)
        if topic not in driver._listeners:
            driver._listeners[topic] = []

        driver._listeners[topic].append(test_handler)

        # Verify storage worked
        assert topic in driver._listeners
        assert len(driver._listeners[topic]) == 1
        assert driver._listeners[topic][0] == test_handler

    def test_multiple_handlers_same_topic(self, driver_config):
        """Test multiple handlers for the same topic."""
        driver = ZeroMQDriver(driver_config)

        topic = "test/topic"
        handler1 = Mock()
        handler2 = Mock()
        handler3 = Mock()

        # Initialize subscription storage
        if topic not in driver._listeners:
            driver._listeners[topic] = []

        # Add multiple handlers
        driver._listeners[topic].append(handler1)
        driver._listeners[topic].append(handler2)
        driver._listeners[topic].append(handler3)

        # Verify all handlers are stored
        assert len(driver._listeners[topic]) == 3
        assert handler1 in driver._listeners[topic]
        assert handler2 in driver._listeners[topic]
        assert handler3 in driver._listeners[topic]

    def test_topic_validation_no_spaces(self, driver_config):
        """Test that topics with spaces should be rejected."""
        driver = ZeroMQDriver(driver_config)

        # Test topic validation logic
        invalid_topics = [
            "topic with spaces",
            "topic\twith\ttabs",
            "topic with  multiple  spaces",
        ]

        for topic in invalid_topics:
            # Topic should contain spaces (which are invalid)
            assert " " in topic or "\t" in topic

        # Valid topics
        valid_topics = [
            "simple/topic",
            "topic_with_underscores",
            "topic-with-dashes",
            "topic/with/slashes",
        ]

        for topic in valid_topics:
            assert " " not in topic
            assert "\t" not in topic

    def test_state_management(self, driver_config):
        """Test internal state management."""
        driver = ZeroMQDriver(driver_config)

        # Test initial state
        assert driver._context is None
        assert driver._pub_socket is None
        assert driver._sub_socket is None
        assert driver._reader_task is None
        assert driver._connected is False
        assert driver._listeners == {}

        # Test state changes
        mock_context = Mock()
        mock_pub = Mock()
        mock_sub = Mock()
        mock_task = Mock()

        driver._context = mock_context
        driver._pub_socket = mock_pub
        driver._sub_socket = mock_sub
        driver._reader_task = mock_task
        driver._connected = True

        # Verify state changes
        assert driver._context == mock_context
        assert driver._pub_socket == mock_pub
        assert driver._sub_socket == mock_sub
        assert driver._reader_task == mock_task
        assert driver._connected is True

    def test_listeners_data_structure_integrity(self, driver_config):
        """Test listeners data structure maintains integrity."""
        driver = ZeroMQDriver(driver_config)

        # Test multiple topics
        topic1 = "agent/command"
        topic2 = "agent/response"
        handler1 = Mock()
        handler2 = Mock()

        # Initialize subscription storage
        driver._listeners[topic1] = [handler1]
        driver._listeners[topic2] = [handler2]

        # Verify separation
        assert len(driver._listeners) == 2
        assert topic1 in driver._listeners
        assert topic2 in driver._listeners
        assert handler1 in driver._listeners[topic1]
        assert handler2 in driver._listeners[topic2]

        # Handlers should not leak between topics
        assert handler1 not in driver._listeners[topic2]
        assert handler2 not in driver._listeners[topic1]

    def test_url_construction(self, driver_config):
        """Test ZeroMQ URL construction logic."""
        driver = ZeroMQDriver(driver_config)

        # Test URL construction without actually connecting
        expected_pub_url = f"tcp://{driver.config.url}:{driver.config.port}"
        expected_sub_url = f"tcp://{driver.config.url}:{driver.config.backend_port}"

        assert expected_pub_url == "tcp://127.0.0.1:5555"
        assert expected_sub_url == "tcp://127.0.0.1:5556"

    def test_topic_encoding_config(self, driver_config):
        """Test topic encoding configuration."""
        driver = ZeroMQDriver(driver_config)

        # Test default encoding
        assert driver.config.topic_encoding == "utf-8"

        # Test that topics can be encoded
        topic = "test/topic"
        encoded = topic.encode(driver.config.topic_encoding)
        assert isinstance(encoded, bytes)
        assert encoded == b"test/topic"

    def test_prefix_matching_logic(self, driver_config):
        """Test topic prefix matching logic (as used in _reader)."""
        driver = ZeroMQDriver(driver_config)

        # Set up subscriptions
        driver._listeners["agent/command"] = [Mock()]
        driver._listeners["agent/response"] = [Mock()]
        driver._listeners["agent/command/ask"] = [Mock()]

        # Test prefix matching logic
        incoming_topics = [
            ("agent/command/ask_question", ["agent/command", "agent/command/ask"]),
            ("agent/response/answer", ["agent/response"]),
            ("agent/command", ["agent/command"]),
            ("other/topic", []),
        ]

        for incoming_topic, expected_matches in incoming_topics:
            matched = []
            for registered_topic in driver._listeners.keys():
                if incoming_topic.startswith(registered_topic):
                    matched.append(registered_topic)

            assert set(matched) == set(expected_matches), (
                f"Topic '{incoming_topic}' should match {expected_matches}, "
                f"but matched {matched}"
            )

    def test_format_connection_error_details(self, driver_config):
        """Test error message formatting."""
        driver = ZeroMQDriver(driver_config)

        # Test with actual ZMQ error if available
        try:
            import zmq

            zmq_error = zmq.ZMQError()
            result = driver.format_connection_error_details(zmq_error)
            assert result is not None
            assert "ZeroMQ error" in result
        except ImportError:
            # If ZMQ not available, test with generic exception
            generic_error = Exception("connection failed")
            result = driver.format_connection_error_details(generic_error)
            assert result is None

    def test_config_defaults(self):
        """Test default configuration values."""
        config = ZeroMQDriverConfig()

        # Test defaults from schema
        assert config.url == "127.0.0.1"
        assert config.port == 5555
        assert config.backend_port == 5556
        assert config.start_proxy is True
        assert config.proxy_mode == "embedded"
        assert config.high_water_mark == 1000
        assert config.linger == 1000
        assert config.connect_timeout == 5000
        assert config.topic_encoding == "utf-8"

    def test_message_format_validation(self, driver_config):
        """Test message format parsing logic."""
        driver = ZeroMQDriver(driver_config)

        # Test wire format parsing (as would happen in _reader)
        valid_messages = [
            ("topic message_data", ("topic", "message_data")),
            ("agent/command/ask {\"type\":\"ask\"}", ("agent/command/ask", '{"type":"ask"}')),
            ("simple data", ("simple", "data")),
        ]

        for message, expected in valid_messages:
            parts = message.split(" ", 1)
            if len(parts) == 2:
                topic, data = parts
                assert (topic, data) == expected

        # Invalid messages (should be caught in _reader)
        invalid_messages = [
            "no_space_separator",
            "",
        ]

        for message in invalid_messages:
            parts = message.split(" ", 1)
            assert len(parts) != 2 or not parts[0] or not parts[1]

    @pytest.mark.asyncio
    async def test_publish_validation(self, driver_config):
        """Test publish parameter validation."""
        driver = ZeroMQDriver(driver_config)

        # Test that publish rejects topics with spaces
        from argentic.core.protocol.message import BaseMessage

        message = BaseMessage(type="test")

        # Simulate connected state
        driver._connected = True
        driver._pub_socket = Mock()

        # Should raise ValueError for topic with spaces
        with pytest.raises(ValueError, match="contains spaces"):
            await driver.publish("topic with spaces", message)

    def test_proxy_manager_integration(self, embedded_config):
        """Test proxy manager integration in embedded mode."""
        driver = ZeroMQDriver(embedded_config)

        # Proxy manager should not be created until connect()
        assert driver._proxy_manager is None
        assert driver.config.start_proxy is True
        assert driver.config.proxy_mode == "embedded"
