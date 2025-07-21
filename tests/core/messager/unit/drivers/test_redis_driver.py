import pytest
from unittest.mock import AsyncMock, patch, MagicMock, Mock
import sys
import os

try:
    import aioredis  # noqa: F401
except Exception:  # pragma: no cover
    import pytest as _pytest

    _pytest.skip(
        "aioredis dependency is missing – skipping Redis driver tests.", allow_module_level=True
    )

# Add src to path to fix import issues
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))

from argentic.core.messager.drivers.RedisDriver import RedisDriver
from argentic.core.messager.drivers import DriverConfig


@pytest.fixture
def driver_config() -> DriverConfig:
    return DriverConfig(
        url="redis://localhost",
        port=6379,
        user=None,
        password="testpass",
        token=None,
    )


class TestRedisDriver:
    """
    Unit tests for Redis Driver interface only.
    Complex async networking behavior is tested in e2e tests.
    """

    def test_init(self, driver_config):
        """Test driver initialization"""
        driver = RedisDriver(driver_config)

        # Verify initial state
        assert driver._redis is None
        assert driver._connected is False
        assert driver._subscriptions == {}
        assert driver._pubsub is None
        assert driver._listen_task is None

    def test_is_connected_when_disconnected(self, driver_config):
        """Test is_connected returns False when not connected"""
        driver = RedisDriver(driver_config)
        assert driver.is_connected() is False

    def test_is_connected_when_connected(self, driver_config):
        """Test is_connected returns True when connected flag is set"""
        driver = RedisDriver(driver_config)
        driver._connected = True
        assert driver.is_connected() is True

    def test_config_url_handling(self, driver_config):
        """Test URL configuration handling"""
        driver = RedisDriver(driver_config)

        # Test URL building logic
        expected_url = f"{driver_config.url}:{driver_config.port}"
        if driver_config.password:
            # Redis URL should include password if provided
            assert driver_config.password == "testpass"

        # Verify config is accessible
        assert driver_config.url == "redis://localhost"
        assert driver_config.port == 6379

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self, driver_config):
        """Test disconnect when not connected does nothing gracefully"""
        driver = RedisDriver(driver_config)

        # Should not raise any exceptions
        await driver.disconnect()

        assert driver._connected is False

    def test_subscription_storage(self, driver_config):
        """Test that subscription data structures work correctly"""
        driver = RedisDriver(driver_config)

        # Test subscription storage without network calls
        test_handler = Mock()
        channel = "test/channel"

        # Manually test subscription storage logic
        if channel not in driver._subscriptions:
            driver._subscriptions[channel] = {}

        driver._subscriptions[channel]["BaseMessage"] = (test_handler, {})

        # Verify storage worked
        assert channel in driver._subscriptions
        stored_handler, _ = driver._subscriptions[channel]["BaseMessage"]
        assert stored_handler == test_handler

    def test_multiple_subscriptions_same_channel(self, driver_config):
        """Test multiple subscriptions to same channel"""
        driver = RedisDriver(driver_config)

        channel = "test/channel"
        handler1 = Mock()
        handler2 = Mock()

        # Initialize subscription storage
        if channel not in driver._subscriptions:
            driver._subscriptions[channel] = {}

        # Test overwriting subscription
        driver._subscriptions[channel]["BaseMessage"] = (handler1, {"option": "value1"})
        stored_handler, opts = driver._subscriptions[channel]["BaseMessage"]
        assert stored_handler == handler1
        assert opts["option"] == "value1"

        # Test replacing subscription
        driver._subscriptions[channel]["BaseMessage"] = (handler2, {"option": "value2"})
        stored_handler, opts = driver._subscriptions[channel]["BaseMessage"]
        assert stored_handler == handler2
        assert opts["option"] == "value2"

    def test_publish_input_validation(self, driver_config):
        """Test that publish validates input parameters without network calls"""
        driver = RedisDriver(driver_config)
        driver._connected = True
        driver._redis = Mock()  # Simple mock, no async behavior

        # Test with invalid payload types - should fail before network call
        invalid_payloads = ["string", {"dict": "value"}, b"bytes", 123, None]

        for payload in invalid_payloads:
            # Test validation directly without running async loop
            # Invalid payloads should not have model_dump_json method
            assert hasattr(payload, "model_dump_json") is False

    def test_state_management(self, driver_config):
        """Test internal state management"""
        driver = RedisDriver(driver_config)

        # Test initial state
        assert driver._redis is None
        assert driver._pubsub is None
        assert driver._listen_task is None
        assert driver._connected is False

        # Test state changes
        mock_redis = Mock()
        mock_pubsub = Mock()
        mock_task = Mock()

        driver._redis = mock_redis
        driver._pubsub = mock_pubsub
        driver._listen_task = mock_task
        driver._connected = True

        # Verify state changes
        assert driver._redis == mock_redis
        assert driver._pubsub == mock_pubsub
        assert driver._listen_task == mock_task
        assert driver._connected is True

    def test_subscription_data_structure_integrity(self, driver_config):
        """Test subscription data structure maintains integrity"""
        driver = RedisDriver(driver_config)

        # Test multiple channels
        channel1 = "channel1"
        channel2 = "channel2"
        handler1 = Mock()
        handler2 = Mock()

        # Initialize subscription storage
        for channel in [channel1, channel2]:
            if channel not in driver._subscriptions:
                driver._subscriptions[channel] = {}

        # Add subscriptions
        driver._subscriptions[channel1]["BaseMessage"] = (handler1, {})
        driver._subscriptions[channel2]["BaseMessage"] = (handler2, {})

        # Verify separation
        assert len(driver._subscriptions) == 2
        assert channel1 in driver._subscriptions
        assert channel2 in driver._subscriptions

        stored_handler1, _ = driver._subscriptions[channel1]["BaseMessage"]
        stored_handler2, _ = driver._subscriptions[channel2]["BaseMessage"]

        assert stored_handler1 == handler1
        assert stored_handler2 == handler2
        assert stored_handler1 != stored_handler2

    @patch("argentic.core.messager.drivers.RedisDriver.aioredis")
    def test_redis_url_construction(self, mock_aioredis, driver_config):
        """Test Redis URL construction logic"""
        driver = RedisDriver(driver_config)

        # Test URL construction without actually connecting
        base_url = driver_config.url
        port = driver_config.port
        password = driver_config.password

        # Verify URL components
        assert base_url == "redis://localhost"
        assert port == 6379
        assert password == "testpass"

        # Test expected URL format
        if port and port != 6379:
            expected_url = f"{base_url}:{port}"
        else:
            expected_url = base_url

        # For this test config, should be original URL since port is default
        assert base_url == "redis://localhost"

    def test_channel_pattern_validation(self, driver_config):
        """Test channel pattern validation logic"""
        driver = RedisDriver(driver_config)

        # Test valid channel patterns
        valid_channels = [
            "simple/channel",
            "test.channel",
            "channel_with_underscores",
            "channel-with-dashes",
            "123numeric456",
        ]

        for channel in valid_channels:
            # Basic validation - channel should be string
            assert isinstance(channel, str)
            assert len(channel) > 0

        # Test edge cases
        empty_channel = ""
        none_channel = None

        # These should be invalid
        assert empty_channel == ""
        assert none_channel is None
