"""Integration test for ZeroMQ driver with real messaging.

Tests complete pub/sub flow using the Messager interface over ZeroMQ driver.
"""

import asyncio
from typing import List

import pytest

try:
    import zmq  # noqa: F401

    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False

if not ZMQ_AVAILABLE:
    pytest.skip(
        "pyzmq dependency is missing – skipping ZeroMQ integration tests.",
        allow_module_level=True,
    )

from argentic.core.messager.messager import Messager
from argentic.core.messager.protocols import MessagerProtocol
from argentic.core.protocol.message import BaseMessage


@pytest.mark.asyncio
@pytest.mark.integration
async def test_zeromq_pub_sub_via_messager():
    """Test ZeroMQ pub/sub through Messager interface."""
    # Create publisher with embedded proxy
    publisher = Messager(
        protocol=MessagerProtocol.ZEROMQ,
        broker_address="127.0.0.1",
        port=18555,  # Integration test ports
        backend_port=18556,
        start_proxy=True,
        proxy_mode="embedded",
        client_id="integration_publisher",
    )
    await publisher.connect()
    await asyncio.sleep(0.3)  # Let proxy stabilize

    # Create subscriber
    subscriber = Messager(
        protocol=MessagerProtocol.ZEROMQ,
        broker_address="127.0.0.1",
        port=18555,
        backend_port=18556,
        start_proxy=False,
        proxy_mode="external",
        client_id="integration_subscriber",
    )
    await subscriber.connect()
    await asyncio.sleep(0.2)

    # Set up message collection
    received: List[BaseMessage] = []

    async def handler(message: BaseMessage):
        received.append(message)

    # Subscribe with BaseMessage (works with all message types)
    await subscriber.subscribe("integration/test", handler, BaseMessage)
    await asyncio.sleep(0.5)

    # Publish messages
    for i in range(5):
        msg = BaseMessage(type=f"test_msg_{i}")
        await publisher.publish("integration/test", msg)
        await asyncio.sleep(0.15)

    # Wait for delivery
    await asyncio.sleep(1.0)

    # Cleanup
    await subscriber.stop()
    await publisher.stop()

    # Verify
    assert len(received) == 5, f"Expected 5 messages, got {len(received)}"
    for i, msg in enumerate(received):
        assert msg.type == f"test_msg_{i}"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_zeromq_multiple_topics():
    """Test routing messages to different topics."""
    publisher = Messager(
        protocol=MessagerProtocol.ZEROMQ,
        broker_address="127.0.0.1",
        port=18557,
        backend_port=18558,
        start_proxy=True,
        proxy_mode="embedded",
    )
    await publisher.connect()
    await asyncio.sleep(0.3)

    subscriber = Messager(
        protocol=MessagerProtocol.ZEROMQ,
        broker_address="127.0.0.1",
        port=18557,
        backend_port=18558,
        start_proxy=False,
    )
    await subscriber.connect()
    await asyncio.sleep(0.2)

    topic1_msgs: List[BaseMessage] = []
    topic2_msgs: List[BaseMessage] = []

    async def topic1_handler(msg: BaseMessage):
        topic1_msgs.append(msg)

    async def topic2_handler(msg: BaseMessage):
        topic2_msgs.append(msg)

    await subscriber.subscribe("topic/one", topic1_handler)
    await subscriber.subscribe("topic/two", topic2_handler)
    await asyncio.sleep(0.5)

    # Publish to different topics
    await publisher.publish("topic/one", BaseMessage(type="msg_one"))
    await publisher.publish("topic/two", BaseMessage(type="msg_two"))
    await publisher.publish("topic/one", BaseMessage(type="msg_one_2"))

    await asyncio.sleep(0.8)

    await subscriber.stop()
    await publisher.stop()

    # Verify correct routing
    assert len(topic1_msgs) == 2
    assert len(topic2_msgs) == 1
