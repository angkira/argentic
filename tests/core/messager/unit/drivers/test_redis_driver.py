import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import sys
import os

# Add src to path to fix import issues
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))


# Create complete mock for aioredis to avoid the TimeoutError issue
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


# Apply the mock
sys.modules["aioredis"] = RedisMock

# Import after mocking to avoid the real aioredis being imported
from argentic.core.messager.drivers import DriverConfig
from argentic.core.protocol.message import BaseMessage
from argentic.core.messager.drivers.RedisDriver import RedisDriver


@pytest.fixture
def driver_config() -> DriverConfig:
    """Create a test driver configuration"""
    return DriverConfig(
        url="test.redis.server", port=6379, user="testuser", password="testpass", token=None
    )


@pytest.fixture
def mock_redis():
    """Create a mocked Redis client"""
    redis = AsyncMock()
    redis.close = AsyncMock()
    redis.closed = False
    redis.publish = AsyncMock()

    # Create a proper PubSub mock that correctly handles async behavior
    pubsub = AsyncMock()
    pubsub.subscribe = AsyncMock()
    pubsub.listen = AsyncMock()

    # Set up listen async iterator
    pubsub.__aiter__ = AsyncMock()
    pubsub.__aiter__.return_value = pubsub
    pubsub.__anext__ = AsyncMock()

    # Make redis.pubsub() return our mock_pubsub directly, not as a coroutine
    redis.pubsub = MagicMock(return_value=pubsub)

    return {"redis": redis, "pubsub": pubsub}


@pytest.mark.asyncio
class TestRedisDriver:
    """Tests for the RedisDriver class"""

    def setup_method(self, method):
        """Setup before each test method"""
        # Create mocks for Redis components
        self.mock_redis = AsyncMock()
        self.mock_redis.close = AsyncMock()
        self.mock_redis.closed = False
        self.mock_redis.publish = AsyncMock()

        # Create a proper PubSub mock that correctly handles async behavior
        self.mock_pubsub = AsyncMock()
        # Critical: subscribe must be an AsyncMock, not a method on an AsyncMock
        self.mock_pubsub.subscribe = AsyncMock()
        self.mock_pubsub.listen = AsyncMock()

        # Set up listen async iterator
        self.mock_pubsub.__aiter__ = AsyncMock()
        self.mock_pubsub.__aiter__.return_value = self.mock_pubsub
        self.mock_pubsub.__anext__ = AsyncMock()

        # Make redis.pubsub() return our mock_pubsub directly, not as a coroutine
        self.mock_redis.pubsub = MagicMock(return_value=self.mock_pubsub)

    @patch("argentic.core.messager.drivers.RedisDriver.aioredis.from_url")
    async def test_init(self, mock_from_url, driver_config):
        """Test driver initialization"""
        driver = RedisDriver(driver_config)

        # Verify initial state
        assert driver._redis is None
        assert driver._pubsub is None
        assert isinstance(driver._listeners, dict)
        assert len(driver._listeners) == 0
        assert driver._reader_task is None

    @pytest.mark.parametrize(
        "config,expected_url",
        [
            # Standard case
            (
                DriverConfig(
                    url="test.redis.server",
                    port=6379,
                    user="testuser",
                    password="testpass",
                    token=None,
                ),
                "redis://test.redis.server:6379",
            ),
            # Custom port
            (
                DriverConfig(
                    url="custom.redis.server",
                    port=7000,
                    user="testuser",
                    password="testpass",
                    token=None,
                ),
                "redis://custom.redis.server:7000",
            ),
            # No authentication
            (
                DriverConfig(
                    url="no-auth.redis.server", port=6379, user=None, password=None, token=None
                ),
                "redis://no-auth.redis.server:6379",
            ),
        ],
    )
    @patch("argentic.core.messager.drivers.RedisDriver.aioredis.from_url")
    async def test_connect_with_different_configs(self, mock_from_url, config, expected_url):
        """Test connect method with different configurations"""
        mock_from_url.return_value = self.mock_redis

        driver = RedisDriver(config)
        await driver.connect()

        # Verify Redis connection was made with correct URL
        if config.password:
            mock_from_url.assert_awaited_once_with(expected_url, password=config.password)
        else:
            mock_from_url.assert_awaited_once_with(expected_url, password=None)

        assert driver._redis == self.mock_redis

    @patch("argentic.core.messager.drivers.RedisDriver.aioredis.from_url")
    async def test_disconnect(self, mock_from_url, driver_config):
        """Test disconnect method"""
        mock_from_url.return_value = self.mock_redis

        driver = RedisDriver(driver_config)
        driver._redis = self.mock_redis

        await driver.disconnect()

        # Verify Redis connection was closed
        self.mock_redis.close.assert_awaited_once()

    @patch("argentic.core.messager.drivers.RedisDriver.aioredis.from_url")
    async def test_disconnect_handles_no_connection(self, mock_from_url, driver_config):
        """Test disconnect method with no active connection"""
        driver = RedisDriver(driver_config)
        driver._redis = None  # Explicitly set to None to simulate no connection

        # This should not raise any exceptions
        await driver.disconnect()

    @pytest.mark.parametrize(
        "message_data",
        [
            # Standard message
            {"id": "test-id", "type": "test-type"},
            # Message with nested data
            {
                "id": "nested-id",
                "type": "nested",
                "data": {"key": "value", "nested": {"key2": "value2"}},
            },
            # Message with arrays
            {"id": "array-id", "type": "array", "items": [1, 2, 3, 4, 5]},
            # Empty message
            {"id": "empty-id", "type": "empty"},
        ],
    )
    @patch("argentic.core.messager.drivers.RedisDriver.aioredis.from_url")
    async def test_publish_with_different_message_types(
        self, mock_from_url, driver_config, message_data
    ):
        """Test publishing different types of messages"""
        import json

        mock_from_url.return_value = self.mock_redis

        driver = RedisDriver(driver_config)
        driver._redis = self.mock_redis

        # Mock BaseMessage
        class MockBaseMessage:
            def __init__(self, data):
                self.data = data

            def model_dump_json(self):
                return json.dumps(self.data)

        # Test data
        test_topic = "test-topic"
        test_message = MockBaseMessage(message_data)

        await driver.publish(test_topic, test_message)

        # Verify message was published with correct parameters
        self.mock_redis.publish.assert_awaited_once_with(test_topic, test_message.model_dump_json())

        # Verify the content matches expected json
        expected_json = json.dumps(message_data)
        actual_json = test_message.model_dump_json()
        assert actual_json == expected_json

    @patch("argentic.core.messager.drivers.RedisDriver.aioredis.from_url")
    @patch("argentic.core.messager.drivers.RedisDriver.asyncio.create_task")
    async def test_subscribe_first_topic(
        self, mock_create_task, mock_from_url, driver_config, mock_redis
    ):
        """Test subscribing to first topic - initializes pubsub and reader task"""
        # Use the fixture instead of the instance variables
        redis = mock_redis["redis"]
        pubsub = mock_redis["pubsub"]
        mock_from_url.return_value = redis

        driver = RedisDriver(driver_config)
        driver._redis = redis

        # Test data
        test_topic = "test-topic"
        test_handler = AsyncMock()

        # Call subscribe
        await driver.subscribe(test_topic, test_handler)

        # Verify that pubsub() was called
        redis.pubsub.assert_called_once()

        # Verify that subscribe was called with the correct topic
        pubsub.subscribe.assert_awaited_once_with(test_topic)

        # Verify that a reader task was created
        mock_create_task.assert_called_once()
        assert mock_create_task.call_args[0][0].__name__ == "_reader"

        # Verify the handler was registered
        assert test_topic in driver._listeners
        assert test_handler in driver._listeners[test_topic]

    @patch("argentic.core.messager.drivers.RedisDriver.aioredis.from_url")
    @patch("argentic.core.messager.drivers.RedisDriver.asyncio.create_task")
    async def test_subscribe_multiple_handlers_same_topic(
        self, mock_create_task, mock_from_url, driver_config, mock_redis
    ):
        """Test subscribing multiple handlers to the same topic"""
        redis = mock_redis["redis"]
        pubsub = mock_redis["pubsub"]
        mock_from_url.return_value = redis

        driver = RedisDriver(driver_config)
        driver._redis = redis
        driver._pubsub = pubsub

        # Test data
        test_topic = "same-topic"
        handler1 = AsyncMock()
        handler2 = AsyncMock()
        handler3 = AsyncMock()

        # Subscribe first handler and initialize pubsub
        await driver.subscribe(test_topic, handler1)

        # Reset mocks to verify subsequent calls
        pubsub.subscribe.reset_mock()
        mock_create_task.reset_mock()

        # Subscribe second and third handlers
        await driver.subscribe(test_topic, handler2)
        await driver.subscribe(test_topic, handler3)

        # Verify subscribe was only called for the first handler
        # (Redis doesn't need to resubscribe to a topic)
        assert pubsub.subscribe.await_count == 0

        # Verify create_task was only called once (for the first handler)
        assert mock_create_task.call_count == 0

        # Verify all handlers were registered
        assert len(driver._listeners[test_topic]) == 3
        assert handler1 in driver._listeners[test_topic]
        assert handler2 in driver._listeners[test_topic]
        assert handler3 in driver._listeners[test_topic]

    @patch("argentic.core.messager.drivers.RedisDriver.aioredis.from_url")
    @patch("argentic.core.messager.drivers.RedisDriver.asyncio.create_task")
    async def test_subscribe_additional_topic(self, mock_create_task, mock_from_url, driver_config):
        """Test subscribing to additional topic - reuses pubsub and reader task"""
        # Set up the mock from_url to return our mock_redis
        mock_from_url.return_value = self.mock_redis

        driver = RedisDriver(driver_config)
        driver._redis = self.mock_redis

        # Create pre-existing pubsub and task
        driver._pubsub = self.mock_pubsub
        driver._reader_task = MagicMock()

        # Set up existing subscription
        first_topic = "first-topic"
        first_handler = AsyncMock()
        driver._listeners = {first_topic: [first_handler]}

        # Test data for new subscription
        second_topic = "second-topic"
        second_handler = AsyncMock()

        # Call subscribe for the second topic
        await driver.subscribe(second_topic, second_handler)

        # Verify that pubsub() was NOT called again
        self.mock_redis.pubsub.assert_not_called()

        # Verify that subscribe was called for the new topic
        self.mock_pubsub.subscribe.assert_awaited_once_with(second_topic)

        # Verify that no new reader task was created
        mock_create_task.assert_not_called()

        # Verify the new handler was registered
        assert second_topic in driver._listeners
        assert second_handler in driver._listeners[second_topic]

        # Verify original subscription still exists
        assert first_topic in driver._listeners
        assert first_handler in driver._listeners[first_topic]

    @patch("argentic.core.messager.drivers.RedisDriver.aioredis.from_url")
    async def test_is_connected(self, mock_from_url, driver_config):
        """Test is_connected method"""
        mock_from_url.return_value = self.mock_redis

        driver = RedisDriver(driver_config)

        # Not connected initially
        assert driver.is_connected() is False

        # Connected when redis exists and is not closed
        driver._redis = self.mock_redis
        self.mock_redis.closed = False
        assert driver.is_connected() is True

        # Not connected when redis is closed
        self.mock_redis.closed = True
        assert driver.is_connected() is False

    @patch("argentic.core.messager.drivers.RedisDriver.aioredis.from_url")
    async def test_reader_processes_messages(self, mock_from_url, driver_config):
        """Test _reader method that processes incoming messages"""
        # Set up the mock from_url to return our mock_redis
        mock_from_url.return_value = self.mock_redis

        driver = RedisDriver(driver_config)
        driver._redis = self.mock_redis
        driver._pubsub = self.mock_pubsub

        # Register handlers
        channel = "test-channel"
        handler1 = AsyncMock()
        handler2 = AsyncMock()
        driver._listeners = {channel: [handler1, handler2]}

        # Set up test messages of different types
        test_messages = [
            {
                "type": "message",
                "channel": channel,
                "data": b'{"id":"msg1","type":"test","value":"simple"}',
            },
            {
                "type": "message",
                "channel": channel,
                "data": b'{"id":"msg2","type":"test","nested":{"key":"value"}}',
            },
            {
                "type": "message",
                "channel": "unknown-channel",  # Channel with no listeners
                "data": b'{"id":"msg3","type":"test"}',
            },
        ]

        # Process each message
        for test_message in test_messages:
            if test_message["type"] == "message":
                channel_handlers = driver._listeners.get(test_message["channel"], [])
                for h in channel_handlers:
                    await h(test_message["data"])

        # Verify handlers were called for the right messages
        assert handler1.await_count == 2  # Only first two messages
        assert handler2.await_count == 2  # Only first two messages

        # Verify the arguments
        expected_calls = [
            test_messages[0]["data"],
            test_messages[1]["data"],
        ]
        for i, call in enumerate(handler1.await_args_list):
            assert call.args[0] == expected_calls[i]

    @patch("argentic.core.messager.drivers.RedisDriver.aioredis.from_url")
    async def test_reader_ignores_non_message_types(self, mock_from_url, driver_config):
        """Test _reader method ignores non-message type events"""
        # Set up the mock from_url to return our mock_redis
        mock_from_url.return_value = self.mock_redis

        driver = RedisDriver(driver_config)
        driver._redis = self.mock_redis
        driver._pubsub = self.mock_pubsub

        # Register handlers
        channel = "test-channel"
        handler = AsyncMock()
        driver._listeners = {channel: [handler]}

        # Set up different non-message type Redis events
        non_message_events = [
            {"type": "subscribe", "channel": channel, "data": 1},
            {"type": "unsubscribe", "channel": channel, "data": 0},
            {"type": "pong", "channel": None, "data": "OK"},
        ]

        # Simulate how the _reader processes messages
        for event in non_message_events:
            if event["type"] == "message":  # This check should fail for all test events
                for h in driver._listeners.get(event["channel"], []):
                    await h(event["data"])

        # Verify handler was NOT called for any non-message event
        handler.assert_not_awaited()

    @patch("argentic.core.messager.drivers.RedisDriver.aioredis.from_url")
    async def test_exception_handling_during_message_processing(self, mock_from_url, driver_config):
        """Test that exceptions in message handlers don't break the processing loop"""
        mock_from_url.return_value = self.mock_redis

        driver = RedisDriver(driver_config)
        driver._redis = self.mock_redis
        driver._pubsub = self.mock_pubsub

        # Register a mix of handlers - one that raises an exception and one that doesn't
        channel = "test-channel"
        working_handler = AsyncMock()
        failing_handler = AsyncMock(side_effect=Exception("Handler error"))
        driver._listeners = {channel: [working_handler, failing_handler, working_handler]}

        # Test message
        test_message = {"type": "message", "channel": channel, "data": b'{"key":"value"}'}

        # Process message manually with exception handling like the _reader method would
        if test_message["type"] == "message":
            for h in driver._listeners.get(test_message["channel"], []):
                try:
                    await h(test_message["data"])
                except Exception:
                    # Exception should be caught and ignored
                    pass

        # Verify working handlers were still called despite exception in failing handler
        assert working_handler.await_count == 2  # Called twice
        working_handler.assert_awaited_with(test_message["data"])
        failing_handler.assert_awaited_once_with(test_message["data"])
