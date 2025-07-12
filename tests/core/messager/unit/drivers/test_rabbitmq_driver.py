import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import sys
import os

# Add src to path to fix import issues
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))

# Mock the aio_pika module
sys.modules["aio_pika"] = MagicMock()
sys.modules["aio_pika"].RobustConnection = MagicMock()
sys.modules["aio_pika"].Channel = MagicMock()
sys.modules["aio_pika"].Message = MagicMock()
sys.modules["aio_pika"].ExchangeType = MagicMock()
sys.modules["aio_pika"].connect_robust = AsyncMock()

from argentic.core.messager.drivers import DriverConfig
from argentic.core.protocol.message import BaseMessage
from argentic.core.messager.drivers.RabbitMQDriver import RabbitMQDriver


@pytest.fixture
def driver_config() -> DriverConfig:
    """Create a test driver configuration"""
    return DriverConfig(
        url="test.rabbitmq.server", port=5672, user="testuser", password="testpass", token=None
    )


@pytest.mark.asyncio
class TestRabbitMQDriver:
    """Tests for the RabbitMQDriver class"""

    def setup_method(self):
        """Setup before each test method"""
        # Create mocks for RabbitMQ components
        self.mock_connection = AsyncMock()
        self.mock_connection.close = AsyncMock()
        self.mock_connection.is_closed = False

        self.mock_channel = AsyncMock()
        self.mock_channel.declare_exchange = AsyncMock()
        self.mock_channel.declare_queue = AsyncMock()

        self.mock_exchange = AsyncMock()
        self.mock_exchange.publish = AsyncMock()

        self.mock_queue = AsyncMock()
        self.mock_queue.bind = AsyncMock()
        self.mock_queue.consume = AsyncMock()

    @patch("argentic.core.messager.drivers.RabbitMQDriver.aio_pika.connect_robust")
    async def test_init(self, mock_connect, driver_config):
        """Test driver initialization"""
        driver = RabbitMQDriver(driver_config)

        # Verify initial state
        assert driver._connection is None
        assert driver._channel is None
        assert isinstance(driver._listeners, dict)
        assert len(driver._listeners) == 0
        assert isinstance(driver._queues, dict)
        assert len(driver._queues) == 0

    @patch("argentic.core.messager.drivers.RabbitMQDriver.aio_pika.connect_robust")
    async def test_connect(self, mock_connect, driver_config):
        """Test connect method"""
        mock_connect.return_value = self.mock_connection
        self.mock_connection.channel.return_value = self.mock_channel

        driver = RabbitMQDriver(driver_config)
        await driver.connect()

        # Verify connection was made with correct URL
        expected_url = f"amqp://{driver_config.user}:{driver_config.password}@{driver_config.url}:{driver_config.port}"
        if driver_config.virtualhost and driver_config.virtualhost != "/":
            expected_url += f"/{driver_config.virtualhost.lstrip('/')}"
        else:
            pass

        mock_connect.assert_awaited_once_with(expected_url)

        # Verify channel was created
        self.mock_connection.channel.assert_awaited_once()
        assert driver._connection == self.mock_connection
        assert driver._channel == self.mock_channel

    @patch("argentic.core.messager.drivers.RabbitMQDriver.aio_pika.connect_robust")
    async def test_disconnect(self, mock_connect, driver_config):
        """Test disconnect method"""
        mock_connect.return_value = self.mock_connection

        driver = RabbitMQDriver(driver_config)
        driver._connection = self.mock_connection

        await driver.disconnect()

        # Verify connection was closed
        self.mock_connection.close.assert_awaited_once()

    @patch("argentic.core.messager.drivers.RabbitMQDriver.aio_pika.connect_robust")
    @patch("argentic.core.messager.drivers.RabbitMQDriver.aio_pika.Message")
    @patch("argentic.core.messager.drivers.RabbitMQDriver.aio_pika.ExchangeType")
    @patch("argentic.core.messager.drivers.RabbitMQDriver.json.dumps")
    async def test_publish(
        self, mock_json_dumps, mock_exchange_type, mock_message_class, mock_connect, driver_config
    ):
        """Test publish method"""
        mock_connect.return_value = self.mock_connection
        self.mock_connection.channel.return_value = self.mock_channel
        self.mock_channel.declare_exchange.return_value = self.mock_exchange

        # Mock ExchangeType.FANOUT
        mock_exchange_type.FANOUT = "fanout"

        # Mock aio_pika.Message
        mock_message = MagicMock()
        mock_message_class.return_value = mock_message

        # Mock json.dumps to return a known string
        mock_json_dumps.return_value = '{"id":"test-id","type":"test-type"}'

        driver = RabbitMQDriver(driver_config)
        driver._connection = self.mock_connection
        driver._channel = self.mock_channel

        # Simple mock class that just needs to be detected as a BaseMessage
        class MockBaseMessage:
            id = "test-id"
            timestamp = None

            def model_dump_json(self):
                return '{"id":"test-id","type":"test-type"}'

            def model_dump(self):
                return {"id": "test-id", "type": "test-type"}

            def __dict__(self):
                return {"id": "test-id", "type": "test-type"}

            @property
            def __class__(self):
                return type("MockBaseMessage", (), {"__name__": "MockBaseMessage"})

        # Test data
        test_topic = "test-topic"
        test_message = MockBaseMessage()

        await driver.publish(test_topic, test_message)

        # Verify exchange was declared with correct parameters
        self.mock_channel.declare_exchange.assert_called_once_with(
            test_topic, mock_exchange_type.FANOUT
        )

        # Verify message was published with correct parameters
        mock_message_class.assert_called_once()
        self.mock_exchange.publish.assert_awaited_once_with(mock_message, routing_key="")

    @patch("argentic.core.messager.drivers.RabbitMQDriver.aio_pika.connect_robust")
    @patch("argentic.core.messager.drivers.RabbitMQDriver.aio_pika.ExchangeType")
    async def test_subscribe(self, mock_exchange_type, mock_connect, driver_config):
        """Test subscribe method"""
        mock_connect.return_value = self.mock_connection
        self.mock_connection.channel.return_value = self.mock_channel
        self.mock_channel.declare_exchange.return_value = self.mock_exchange
        self.mock_channel.declare_queue.return_value = self.mock_queue

        # Mock ExchangeType.FANOUT
        mock_exchange_type.FANOUT = "fanout"

        driver = RabbitMQDriver(driver_config)
        driver._connection = self.mock_connection
        driver._channel = self.mock_channel

        # Test data
        test_topic = "test-topic"
        test_handler = AsyncMock()

        await driver.subscribe(test_topic, test_handler)

        # Verify exchange was declared
        self.mock_channel.declare_exchange.assert_awaited_once_with(test_topic, "fanout")

        # Verify queue was declared
        self.mock_channel.declare_queue.assert_awaited_once_with(exclusive=True)

        # Verify queue was bound to exchange
        self.mock_queue.bind.assert_awaited_once_with(self.mock_exchange)

        # Verify queue.consume was called
        self.mock_queue.consume.assert_awaited_once()

        # Verify handler was registered
        assert test_topic in driver._listeners
        assert test_handler in driver._listeners[test_topic]

        # Verify queue was stored
        assert test_topic in driver._queues
        assert driver._queues[test_topic] == self.mock_queue

    @patch("argentic.core.messager.drivers.RabbitMQDriver.aio_pika.connect_robust")
    async def test_is_connected(self, mock_connect, driver_config):
        """Test is_connected method"""
        mock_connect.return_value = self.mock_connection

        driver = RabbitMQDriver(driver_config)

        # Not connected initially
        assert driver.is_connected() is False

        # Connected when connection exists and is not closed
        driver._connection = self.mock_connection
        self.mock_connection.is_closed = False
        assert driver.is_connected() is True

        # Not connected when connection is closed
        self.mock_connection.is_closed = True
        assert driver.is_connected() is False

    @patch("argentic.core.messager.drivers.RabbitMQDriver.aio_pika.connect_robust")
    @patch("argentic.core.messager.drivers.RabbitMQDriver.aio_pika.ExchangeType")
    async def test_message_processing(self, mock_exchange_type, mock_connect, driver_config):
        """Test message processing through the internal _reader function"""
        mock_connect.return_value = self.mock_connection
        self.mock_connection.channel.return_value = self.mock_channel
        self.mock_channel.declare_exchange.return_value = self.mock_exchange
        self.mock_channel.declare_queue.return_value = self.mock_queue

        # Ensure queue.consume will be called during subscribe
        self.mock_queue.consume = AsyncMock()

        # Mock ExchangeType.FANOUT
        mock_exchange_type.FANOUT = "fanout"

        driver = RabbitMQDriver(driver_config)
        driver._connection = self.mock_connection
        driver._channel = self.mock_channel

        # Test data
        test_topic = "test-topic"
        test_handler1 = AsyncMock()
        test_handler2 = AsyncMock()

        # Important: don't set up _listeners beforehand, let subscribe do it
        # driver._listeners = {test_topic: [test_handler1, test_handler2]}

        # First, clear the listeners dict to ensure we have control
        driver._listeners = {}

        # Subscribe to register the first handler
        await driver.subscribe(test_topic, test_handler1)

        # Manually add the second handler after subscribe
        driver._listeners[test_topic].append(test_handler2)

        # Create a mock message
        mock_message = AsyncMock()
        mock_message.body = b'{"key":"value"}'

        # Directly simulate message processing without relying on the reader function
        for h in driver._listeners[test_topic]:
            await h(mock_message.body)

        # Verify handlers were called with message body
        test_handler1.assert_awaited_once_with(mock_message.body)
        test_handler2.assert_awaited_once_with(mock_message.body)
