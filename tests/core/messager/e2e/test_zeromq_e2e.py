"""End-to-end tests for ZeroMQ driver.

Tests real ZeroMQ socket communication with embedded proxy.
Uses unique test ports to avoid conflicts with production systems.
"""

import asyncio
import os
import sys
from typing import List

import pytest

try:
    import zmq  # noqa: F401
    import zmq.asyncio  # noqa: F401

    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False

if not ZMQ_AVAILABLE:
    pytest.skip(
        "pyzmq dependency is missing – skipping ZeroMQ E2E tests.",
        allow_module_level=True,
    )

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))

from argentic.core.messager.drivers.ZeroMQDriver import ZeroMQDriver
from argentic.core.messager.drivers.configs import ZeroMQDriverConfig
from argentic.core.protocol.message import BaseMessage


class TestZeroMQE2E:
    """End-to-end tests for ZeroMQ driver with real sockets."""

    @pytest.fixture
    async def embedded_driver(self):
        """Create ZeroMQ driver with embedded proxy on unique test ports."""
        config = ZeroMQDriverConfig(
            url="127.0.0.1",
            port=15555,  # Test ports to avoid conflicts
            backend_port=15556,
            start_proxy=True,
            proxy_mode="embedded",
            high_water_mark=1000,
        )
        driver = ZeroMQDriver(config)
        await driver.connect()

        # Wait for proxy to stabilize
        await asyncio.sleep(0.3)

        yield driver

        await driver.disconnect()

    @pytest.fixture
    async def subscriber_driver(self, embedded_driver):
        """Create a second driver instance to act as subscriber."""
        config = ZeroMQDriverConfig(
            url="127.0.0.1",
            port=15555,
            backend_port=15556,
            start_proxy=False,  # Use existing proxy
            proxy_mode="external",
        )
        driver = ZeroMQDriver(config)
        await driver.connect()

        # Wait for connection to stabilize
        await asyncio.sleep(0.2)

        yield driver

        await driver.disconnect()

    @pytest.mark.asyncio
    async def test_basic_pub_sub(self, embedded_driver):
        """Test basic publish/subscribe functionality."""
        topic = "test/basic"
        received_messages: List[BaseMessage] = []

        async def handler(message: BaseMessage):
            received_messages.append(message)

        # Subscribe
        await embedded_driver.subscribe(topic, handler)
        await asyncio.sleep(0.1)  # Wait for subscription to propagate

        # Publish
        test_message = BaseMessage(type="test_message")
        await embedded_driver.publish(topic, test_message)

        # Wait for message delivery
        await asyncio.sleep(0.2)

        # Verify
        assert len(received_messages) == 1
        assert received_messages[0].type == "test_message"

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self, embedded_driver, subscriber_driver):
        """Test multiple subscribers receiving same message."""
        topic = "test/multi_sub"
        received_1: List[BaseMessage] = []
        received_2: List[BaseMessage] = []

        async def handler1(message: BaseMessage):
            received_1.append(message)

        async def handler2(message: BaseMessage):
            received_2.append(message)

        # Subscribe both drivers
        await embedded_driver.subscribe(topic, handler1)
        await subscriber_driver.subscribe(topic, handler2)
        await asyncio.sleep(0.2)

        # Publish from embedded driver
        test_message = BaseMessage(type="multi_test")
        await embedded_driver.publish(topic, test_message)

        # Wait for delivery
        await asyncio.sleep(0.3)

        # Both should receive
        assert len(received_1) == 1
        assert len(received_2) == 1
        assert received_1[0].type == "multi_test"
        assert received_2[0].type == "multi_test"

    @pytest.mark.asyncio
    async def test_multiple_topics(self, embedded_driver):
        """Test routing messages to correct topic handlers."""
        topic1 = "test/topic1"
        topic2 = "test/topic2"
        received_1: List[BaseMessage] = []
        received_2: List[BaseMessage] = []

        async def handler1(message: BaseMessage):
            received_1.append(message)

        async def handler2(message: BaseMessage):
            received_2.append(message)

        # Subscribe to different topics
        await embedded_driver.subscribe(topic1, handler1)
        await embedded_driver.subscribe(topic2, handler2)
        await asyncio.sleep(0.1)

        # Publish to topic1
        msg1 = BaseMessage(type="message_1")
        await embedded_driver.publish(topic1, msg1)

        # Publish to topic2
        msg2 = BaseMessage(type="message_2")
        await embedded_driver.publish(topic2, msg2)

        # Wait for delivery
        await asyncio.sleep(0.2)

        # Verify correct routing
        assert len(received_1) == 1
        assert len(received_2) == 1
        assert received_1[0].type == "message_1"
        assert received_2[0].type == "message_2"

    @pytest.mark.asyncio
    async def test_topic_prefix_matching(self, embedded_driver):
        """Test ZeroMQ prefix-based topic matching."""
        base_topic = "test/prefix"
        specific_topic = "test/prefix/specific"

        received_base: List[BaseMessage] = []
        received_specific: List[BaseMessage] = []

        async def base_handler(message: BaseMessage):
            received_base.append(message)

        async def specific_handler(message: BaseMessage):
            received_specific.append(message)

        # Subscribe to base topic (should match all prefixed topics)
        await embedded_driver.subscribe(base_topic, base_handler)
        await embedded_driver.subscribe(specific_topic, specific_handler)
        await asyncio.sleep(0.1)

        # Publish to specific topic
        msg = BaseMessage(type="prefix_test")
        await embedded_driver.publish(specific_topic, msg)

        # Wait for delivery
        await asyncio.sleep(0.2)

        # Base handler should receive (prefix match)
        assert len(received_base) == 1
        # Specific handler should also receive
        assert len(received_specific) == 1

    @pytest.mark.asyncio
    async def test_message_integrity(self, embedded_driver):
        """Test that message content is preserved during transmission."""
        topic = "test/integrity"
        received_messages: List[BaseMessage] = []

        async def handler(message: BaseMessage):
            received_messages.append(message)

        await embedded_driver.subscribe(topic, handler)
        await asyncio.sleep(0.1)

        # Publish message with specific type
        test_message = BaseMessage(type="integrity_test")
        await embedded_driver.publish(topic, test_message)

        await asyncio.sleep(0.2)

        # Verify message integrity
        assert len(received_messages) == 1
        received = received_messages[0]
        assert received.type == "integrity_test"
        assert received.message_id == test_message.message_id
        assert received.timestamp == test_message.timestamp

    @pytest.mark.asyncio
    async def test_high_throughput(self, embedded_driver):
        """Test handling of many messages in quick succession."""
        topic = "test/throughput"
        received_messages: List[BaseMessage] = []

        async def handler(message: BaseMessage):
            received_messages.append(message)

        await embedded_driver.subscribe(topic, handler)
        await asyncio.sleep(0.1)

        # Send many messages
        num_messages = 100
        for i in range(num_messages):
            msg = BaseMessage(type=f"msg_{i}")
            await embedded_driver.publish(topic, msg)

        # Wait for all messages to be processed
        await asyncio.sleep(0.5)

        # Verify all received
        assert len(received_messages) == num_messages

    @pytest.mark.asyncio
    async def test_unsubscribe(self, embedded_driver):
        """Test unsubscribing from topics."""
        topic = "test/unsub"
        received_messages: List[BaseMessage] = []

        async def handler(message: BaseMessage):
            received_messages.append(message)

        # Subscribe
        await embedded_driver.subscribe(topic, handler)
        await asyncio.sleep(0.1)

        # Publish first message
        msg1 = BaseMessage(type="before_unsub")
        await embedded_driver.publish(topic, msg1)
        await asyncio.sleep(0.2)

        assert len(received_messages) == 1

        # Unsubscribe
        await embedded_driver.unsubscribe(topic)
        await asyncio.sleep(0.1)

        # Publish second message (should not be received)
        msg2 = BaseMessage(type="after_unsub")
        await embedded_driver.publish(topic, msg2)
        await asyncio.sleep(0.2)

        # Should still only have first message
        assert len(received_messages) == 1
        assert received_messages[0].type == "before_unsub"

    @pytest.mark.asyncio
    async def test_multiple_handlers_same_topic(self, embedded_driver):
        """Test multiple handlers on the same topic."""
        topic = "test/multi_handler"
        received_1: List[BaseMessage] = []
        received_2: List[BaseMessage] = []

        async def handler1(message: BaseMessage):
            received_1.append(message)

        async def handler2(message: BaseMessage):
            received_2.append(message)

        # Subscribe both handlers to same topic
        await embedded_driver.subscribe(topic, handler1)
        await embedded_driver.subscribe(topic, handler2)
        await asyncio.sleep(0.1)

        # Publish message
        msg = BaseMessage(type="multi_handler_test")
        await embedded_driver.publish(topic, msg)
        await asyncio.sleep(0.2)

        # Both handlers should receive
        assert len(received_1) == 1
        assert len(received_2) == 1
        assert received_1[0].type == "multi_handler_test"
        assert received_2[0].type == "multi_handler_test"

    @pytest.mark.asyncio
    async def test_connection_state(self):
        """Test connection state management."""
        config = ZeroMQDriverConfig(
            url="127.0.0.1",
            port=15557,
            backend_port=15558,
            start_proxy=True,
            proxy_mode="embedded",
        )
        driver = ZeroMQDriver(config)

        # Initially not connected
        assert not driver.is_connected()

        # Connect
        await driver.connect()
        await asyncio.sleep(0.2)
        assert driver.is_connected()

        # Disconnect
        await driver.disconnect()
        await asyncio.sleep(0.1)
        assert not driver.is_connected()

    @pytest.mark.asyncio
    async def test_publish_without_connection(self):
        """Test that publishing without connection raises error."""
        config = ZeroMQDriverConfig(
            url="127.0.0.1",
            port=15559,
            backend_port=15560,
            start_proxy=False,
            proxy_mode="external",
        )
        driver = ZeroMQDriver(config)

        # Try to publish without connecting
        msg = BaseMessage(type="test")
        with pytest.raises(ConnectionError, match="not connected"):
            await driver.publish("test/topic", msg)

    @pytest.mark.asyncio
    async def test_topic_with_spaces_rejected(self, embedded_driver):
        """Test that topics with spaces are rejected."""
        msg = BaseMessage(type="test")

        with pytest.raises(ValueError, match="contains spaces"):
            await embedded_driver.publish("topic with spaces", msg)

    @pytest.mark.asyncio
    async def test_reconnect_scenario(self):
        """Test driver can reconnect after disconnect."""
        config = ZeroMQDriverConfig(
            url="127.0.0.1",
            port=15561,
            backend_port=15562,
            start_proxy=True,
            proxy_mode="embedded",
        )
        driver = ZeroMQDriver(config)

        # First connection
        await driver.connect()
        await asyncio.sleep(0.2)
        assert driver.is_connected()

        # Disconnect
        await driver.disconnect()
        await asyncio.sleep(0.1)
        assert not driver.is_connected()

        # Reconnect
        await driver.connect()
        await asyncio.sleep(0.2)
        assert driver.is_connected()

        # Verify functionality after reconnect
        topic = "test/reconnect"
        received: List[BaseMessage] = []

        async def handler(message: BaseMessage):
            received.append(message)

        await driver.subscribe(topic, handler)
        await asyncio.sleep(0.1)

        msg = BaseMessage(type="after_reconnect")
        await driver.publish(topic, msg)
        await asyncio.sleep(0.2)

        assert len(received) == 1
        assert received[0].type == "after_reconnect"

        await driver.disconnect()
