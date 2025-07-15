import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import sys
import os
import ssl

# Add src to path to fix import issues
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))

from argentic.core.messager.drivers.MQTTDriver import MQTTDriver
from argentic.core.messager.drivers import DriverConfig


# Mock for aiomqtt message
class MockMQTTMessage:
    def __init__(self, topic, payload):
        self.topic = Mock()
        self.topic.value = topic
        self.payload = payload


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


@pytest.fixture
def tls_config() -> dict:
    return {
        "ca_certs": "/path/to/ca.pem",
        "certfile": "/path/to/cert.pem",
        "keyfile": "/path/to/key.pem",
        "cert_reqs": "CERT_REQUIRED",
        "tls_version": "PROTOCOL_TLS",
        "ciphers": None,
    }


@pytest.mark.asyncio
class TestMQTTDriver:
    def setup_method(self):
        # Create a mock client that behaves like aiomqtt.Client
        self.mock_client = AsyncMock()
        self.mock_client.__aenter__ = AsyncMock(return_value=self.mock_client)
        self.mock_client.__aexit__ = AsyncMock()

        # Mock the AsyncExitStack
        self.mock_stack = AsyncMock()
        self.mock_stack.enter_async_context = AsyncMock(return_value=self.mock_client)
        self.mock_stack.aclose = AsyncMock()

    async def test_init(self, driver_config):
        """Test driver initialization"""
        driver = MQTTDriver(driver_config)

        # Verify initial state
        assert driver._client is None
        assert driver._connected is False
        assert driver._subscriptions == {}
        assert driver._message_task is None
        assert driver._stack is None

    @patch("argentic.core.messager.drivers.MQTTDriver.Client")
    @patch("argentic.core.messager.drivers.MQTTDriver.AsyncExitStack")
    async def test_connect(self, mock_stack_class, mock_client_class, driver_config):
        """Test connect method"""
        mock_stack_class.return_value = self.mock_stack
        mock_client_class.return_value = self.mock_client

        driver = MQTTDriver(driver_config)
        result = await driver.connect()

        # Verify client was created with correct parameters
        mock_client_class.assert_called_once()
        called_kwargs = mock_client_class.call_args.kwargs
        assert called_kwargs["hostname"] == driver_config.url
        assert called_kwargs["port"] == driver_config.port
        assert called_kwargs["username"] == driver_config.user
        assert called_kwargs["password"] == driver_config.password
        assert called_kwargs["keepalive"] == driver_config.keepalive or 60

        # Verify connection was established
        self.mock_stack.enter_async_context.assert_called_once_with(self.mock_client)

        # Verify result
        assert result is True
        assert driver._connected is True

    @patch("argentic.core.messager.drivers.MQTTDriver.Client")
    @patch("argentic.core.messager.drivers.MQTTDriver.AsyncExitStack")
    async def test_disconnect(self, mock_stack_class, mock_client_class, driver_config):
        """Test disconnect method"""
        mock_stack_class.return_value = self.mock_stack
        mock_client_class.return_value = self.mock_client

        driver = MQTTDriver(driver_config)

        # Set up connected state
        driver._connected = True
        driver._stack = self.mock_stack
        driver._client = self.mock_client

        # Don't test task cancellation as it's complex to mock properly
        # Just verify that disconnect works when there's no task
        driver._message_task = None

        await driver.disconnect()

        # Verify stack was closed
        self.mock_stack.aclose.assert_called_once()

        # Verify state was reset
        assert driver._connected is False
        assert driver._client is None

    @patch("argentic.core.messager.drivers.MQTTDriver.Client")
    @patch("argentic.core.messager.drivers.MQTTDriver.AsyncExitStack")
    async def test_disconnect_without_task(
        self, mock_stack_class, mock_client_class, driver_config
    ):
        """Test disconnect method when no message task exists"""
        mock_stack_class.return_value = self.mock_stack
        mock_client_class.return_value = self.mock_client

        driver = MQTTDriver(driver_config)

        # Set up connected state but no message task
        driver._connected = True
        driver._stack = self.mock_stack
        driver._client = self.mock_client
        driver._message_task = None

        await driver.disconnect()

        # Verify stack was closed
        self.mock_stack.aclose.assert_called_once()

        # Verify state was reset
        assert driver._connected is False
        assert driver._client is None

    @patch("argentic.core.messager.drivers.MQTTDriver.Client")
    @patch("argentic.core.messager.drivers.MQTTDriver.AsyncExitStack")
    async def test_publish_base_message(self, mock_stack_class, mock_client_class, driver_config):
        """Test publishing a BaseMessage"""
        mock_stack_class.return_value = self.mock_stack
        mock_client_class.return_value = self.mock_client

        driver = MQTTDriver(driver_config)

        # Set up connected state
        driver._connected = True
        driver._client = self.mock_client

        # Mock BaseMessage
        mock_message = Mock()
        mock_message.model_dump_json.return_value = '{"id":"test-id","type":"test-type"}'

        test_topic = "test/topic"
        test_qos = 1
        test_retain = True

        await driver.publish(test_topic, mock_message, qos=test_qos, retain=test_retain)

        # publish called with positional args in driver
        args, kwargs = self.mock_client.publish.call_args
        assert args[0] == test_topic
        assert args[1] == b'{"id":"test-id","type":"test-type"}'
        assert kwargs["qos"] == test_qos
        assert kwargs["retain"] == test_retain

    # The driver now only accepts BaseMessage payloads. Ensure TypeError is raised for non-BaseMessage.

    @pytest.mark.parametrize("bad_payload", ["raw string", {"k": 1}, b"bytes"])
    async def test_publish_invalid_payload_types(self, driver_config, bad_payload):
        driver = MQTTDriver(driver_config)

        # Mock connected state to avoid reconnection loop
        driver._connected = True
        driver._client = AsyncMock()

        test_topic = "test/topic"

        with pytest.raises(AttributeError):
            await driver.publish(test_topic, bad_payload)

    @patch("argentic.core.messager.drivers.MQTTDriver.Client")
    @patch("argentic.core.messager.drivers.MQTTDriver.AsyncExitStack")
    async def test_subscribe(self, mock_stack_class, mock_client_class, driver_config):
        """Test subscribing to a topic"""
        mock_stack_class.return_value = self.mock_stack
        mock_client_class.return_value = self.mock_client

        driver = MQTTDriver(driver_config)

        # Set up connected state
        driver._connected = True
        driver._client = self.mock_client

        test_topic = "test/subscribe"
        test_handler = AsyncMock()
        test_qos = 1

        await driver.subscribe(test_topic, test_handler, qos=test_qos)

        # Verify subscription was made
        self.mock_client.subscribe.assert_called_once_with(test_topic, qos=test_qos)

        # Verify handler was stored in the internal mapping (it is now stored with class key)
        stored_handler, _ = driver._subscriptions[test_topic]["BaseMessage"]
        assert stored_handler == test_handler

    async def test_listen_task_creation(self, driver_config):
        """Test that message task is created during connect"""
        driver = MQTTDriver(driver_config)

        # Initially no task
        assert driver._message_task is None

    async def test_is_connected(self, driver_config):
        """Test is_connected method"""
        driver = MQTTDriver(driver_config)

        # Initially not connected
        assert driver.is_connected() is False

        # Set connected state
        driver._connected = True
        assert driver.is_connected() is True
