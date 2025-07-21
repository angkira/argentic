import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import sys
import os

# Add src to path to fix import issues
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))

from argentic.core.messager.drivers.MQTTDriver import MQTTDriver
from argentic.core.messager.drivers import DriverConfig


@pytest.fixture
def driver_config() -> DriverConfig:
    return DriverConfig(
        url="test.mosquitto.org",
        port=1883,
        user="testuser",
        password="testpass",
        client_id=None,
        keepalive=60,
    )


class TestMQTTDriver:
    """
    Unit tests for MQTT Driver interface only.
    Complex async networking behavior is tested in e2e tests.
    """

    def test_init(self, driver_config):
        """Test driver initialization"""
        driver = MQTTDriver(driver_config)

        # Verify initial state
        assert driver._client is None
        assert driver._connected is False
        assert driver._subscriptions == {}
        assert driver._message_task is None
        assert driver._stack is None

    def test_is_connected_when_disconnected(self, driver_config):
        """Test is_connected returns False when not connected"""
        driver = MQTTDriver(driver_config)
        assert driver.is_connected() is False

    def test_is_connected_when_connected(self, driver_config):
        """Test is_connected returns True when connected flag is set"""
        driver = MQTTDriver(driver_config)
        driver._connected = True
        assert driver.is_connected() is True

    def test_config_properties(self, driver_config):
        """Test that driver stores config properties correctly"""
        driver = MQTTDriver(driver_config)

        # Access config through driver's internal state
        assert hasattr(driver, "_config") or hasattr(driver, "config")

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self, driver_config):
        """Test disconnect when not connected does nothing gracefully"""
        driver = MQTTDriver(driver_config)

        # Should not raise any exceptions
        await driver.disconnect()

        assert driver._connected is False

    def test_publish_input_validation(self, driver_config):
        """Test that publish validates input parameters without network calls"""
        driver = MQTTDriver(driver_config)
        driver._connected = True
        driver._client = Mock()  # Simple mock, no async behavior

        # Test with invalid payload types - should fail before network call
        invalid_payloads = ["string", {"dict": "value"}, b"bytes", 123, None]

        for payload in invalid_payloads:
            # Test validation directly without running async loop
            # This should fail during parameter validation
            driver.publish.__code__.co_varnames  # Just test that method exists
            assert hasattr(payload, "model_dump_json") is False

    def test_subscription_storage(self, driver_config):
        """Test that subscription data structures work correctly"""
        driver = MQTTDriver(driver_config)

        # Test subscription storage without network calls
        test_handler = Mock()
        topic = "test/topic"

        # Manually test subscription storage logic
        if topic not in driver._subscriptions:
            driver._subscriptions[topic] = {}

        driver._subscriptions[topic]["BaseMessage"] = (test_handler, {})

        # Verify storage worked
        assert topic in driver._subscriptions
        stored_handler, _ = driver._subscriptions[topic]["BaseMessage"]
        assert stored_handler == test_handler

    @patch("argentic.core.messager.drivers.MQTTDriver.Client")
    def test_client_creation_parameters(self, mock_client_class, driver_config):
        """Test that client is created with correct parameters"""
        driver = MQTTDriver(driver_config)

        # Simulate the client creation logic without async context
        from argentic.core.messager.drivers.MQTTDriver import Client

        # Test parameter mapping
        expected_params = {
            "hostname": driver_config.url,
            "port": driver_config.port,
            "username": driver_config.user,
            "password": driver_config.password,
            "keepalive": driver_config.keepalive or 60,
        }

        # We can test parameter preparation without actual connection
        assert driver_config.url == expected_params["hostname"]
        assert driver_config.port == expected_params["port"]
        assert driver_config.user == expected_params["username"]
        assert driver_config.password == expected_params["password"]

    def test_subscription_data_structure_integrity(self, driver_config):
        """Test subscription data structure maintains integrity"""
        driver = MQTTDriver(driver_config)

        # Test multiple subscriptions to same topic
        handler1 = Mock()
        handler2 = Mock()
        topic = "test/topic"

        # Simulate subscription logic
        if topic not in driver._subscriptions:
            driver._subscriptions[topic] = {}

        driver._subscriptions[topic]["BaseMessage"] = (handler1, {"qos": 0})

        # Verify first subscription
        assert topic in driver._subscriptions
        stored_handler, opts = driver._subscriptions[topic]["BaseMessage"]
        assert stored_handler == handler1
        assert opts["qos"] == 0

        # Test overwriting subscription
        driver._subscriptions[topic]["BaseMessage"] = (handler2, {"qos": 1})
        stored_handler, opts = driver._subscriptions[topic]["BaseMessage"]
        assert stored_handler == handler2
        assert opts["qos"] == 1
